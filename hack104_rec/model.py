import datetime as dt
import itertools as itt
import logging
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import attr

import pyspark
from hack104_rec.template import env
from pyspark.sql import functions as f
from pyspark.sql.types import LongType, StringType

from .core import auto_spark, udfy
from .data import Data, DataModelMixin
from .job import JobDateRange, JobProcessed
from .train_action import TrainActionUnstacked
from .train_click import TrainClickCTR, TrainClickExploded

logger = logging.getLogger(__name__)


@contextmanager
def print_duration(prefix=''):
    now = dt.datetime.now()
    yield
    print(prefix, dt.datetime.now() - now)


class Features(DataModelMixin):

    data = Data('features.pq')

    def __init__(self, start_date, stop_date):
        self.start_date = start_date
        self.stop_date = stop_date

    @auto_spark
    def query(self, populate_if_empty=False, spark=None):
        error_occured = False
        try:
            sdf = (super().query(spark)
                   .filter((f.col('start_date') == self.start_date) &
                           (f.col('stop_date') == self.stop_date)))
        except pyspark.sql.utils.AnalysisException:
            error_occured = True

        if populate_if_empty and (error_occured or sdf.limit(1).count() == 0):

            logger.warning(
                f'Populating {type(self).__name__}'
                f'(start_date={self.start_date!r}, '
                f'stop_date={self.stop_date!r})')

            self.populate(spark=spark)
            return self.query(populate_if_empty=False)
        else:
            return sdf

    @auto_spark
    def populate(self, spark=None):
        open_days_sdf = (
            JobDateRange.query(spark=spark)
            .select(
                'jobno',
                f.when(f.col('date_stop') < self.stop_date, f.col('date_stop'))
                .otherwise(self.stop_date).alias('date_stop'),

                f.when(f.col('date_start') >
                       self.start_date, f.col('date_start'))
                .otherwise(self.start_date).alias('date_start'),
            )
            .withColumn('open_days',
                        f.datediff(f.col('date_stop'),
                                   f.col('date_start')) + 1)
        )

        features_sdf = (
            TrainActionUnstacked.query(spark=spark)
            .filter(
                (f.col('date') >= self.start_date) &
                (f.col('date') <= self.stop_date))
            .groupby('jobno')
            .sum()
            .drop('sum(jobno)')
        )

        count_columns = TrainActionUnstacked.query().columns
        count_columns.remove('jobno')
        count_columns.remove('date')

        features_sdf = (
            features_sdf
            .join(open_days_sdf, 'jobno', how='left')
        )

        for column in count_columns:
            features_sdf = (
                features_sdf
                .withColumnRenamed(f'sum({column})', column)
                .withColumn(f'{column}_per_day',
                            f.col(column) / f.col('open_days'))
                .withColumn(f'{column}_per_day',
                            f.col(column) / f.col('open_days'))
            )

        features_sdf = (
            features_sdf
            .withColumn('saveJob_per_viewJob',
                        f.col('viewJob_total') / f.col('saveJob_total'))
            .withColumn('applyJob_per_viewJob',
                        f.col('viewJob_total') / f.col('applyJob_total'))
        )

        job_sdf = (JobProcessed.query()
                   .withColumn('custno', uuid_to_long.udf('custno'))
                   .drop('description', 'job', 'others'))

        ctr_sdf = (TrainClickCTR.query()
                   .filter(
                       (f.col('date') >= self.start_date) &
                       (f.col('date') <= self.stop_date))
                   .groupby('job', 'action')
                   .agg((f.sum('click') / f.sum('impr')).alias('CTR'))
                   .withColumn('CTR_clickJob',
                               f.when(f.col('action') == 'clickJob',
                                      f.col('CTR')).otherwise(0))
                   .withColumn('CTR_clickSave',
                               f.when(f.col('action') == 'clickSave',
                                      f.col('CTR')).otherwise(0))
                   .withColumn('CTR_clickApply',
                               f.when(f.col('action') == 'clickApply',
                                      f.col('CTR')).otherwise(0))
                   .groupby('job')
                   .agg(f.sum('CTR_clickApply').alias('CTR_clickApply'),
                        f.sum('CTR_clickSave').alias('CTR_clickSave'),
                        f.sum('CTR_clickJob').alias('CTR_clickJob')
                        )
                   .withColumnRenamed('job', 'jobno')
                   )

        # features_sdf = features_sdf.join(ctr_sdf, 'jobno', how='left')
        features_sdf = features_sdf.join(job_sdf, 'jobno', how='left')
        features_sdf = (
            features_sdf
            .withColumn('start_date', f.lit(self.start_date))
            .withColumn('stop_date', f.lit(self.stop_date)))

        self.write(features_sdf,
                   partitionBy=('start_date', 'stop_date'),
                   compression='snappy')


@udfy(return_type=LongType())
def uuid_to_long(the_uuid):
    return hash(uuid.UUID('b1049129-a9f5-4b59-9258-784d2b1bf5d1').int)


class Dataset(DataModelMixin):
    data = Data('dataset.pq')

    def __init__(self, start_date, stop_date,
                 features_start_date=None, features_stop_date=None):
        if features_start_date is None and features_stop_date is None:
            features_start_date = start_date
            features_stop_date = stop_date
        self.start_date = start_date
        self.stop_date = stop_date
        self.features_start_date = features_start_date
        self.features_stop_date = features_stop_date

    @auto_spark
    def query(self, populate_if_empty=False, spark=None):

        error_occured = False
        try:
            sdf = (
                super().query(spark)
                .filter(
                    (f.col('start_date') == self.start_date) &
                    (f.col('stop_date') == self.stop_date) &
                    (f.col('features_start_date') ==
                     self.features_start_date) &
                    (f.col('features_stop_date') == self.features_stop_date)))
        except pyspark.sql.utils.AnalysisException:
            error_occured = True

        if populate_if_empty and (error_occured or sdf.limit(1).count() == 0):
            logger.warning(
                f'Populating {type(self).__name__}'
                f'(start_date={self.start_date!r}, '
                f'stop_date={self.stop_date!r}, '
                f'features_start_date={self.features_start_date!r}, '
                f'features_stop_date={self.features_stop_date!r})'
            )

            self.populate(spark=spark)
            return self.query(populate_if_empty=False)
        else:
            return sdf

    @auto_spark(('spark.driver.memory', '10g'),
                ('spark.executor.memory', '10g'))
    def populate(
            self,
            spark=None):

        features_sdf = (
            Features(self.features_start_date, self.features_stop_date)
            .query(
                populate_if_empty=True,
                spark=spark)
            .drop('start_date', 'stop_date')
        )

        label_sdf = (
            TrainClickExploded.query(spark=spark)
            .filter(
                (f.col('date') >= self.start_date) &
                (f.col('date') <= self.stop_date))
            .drop('source', 'date', 'datetime', 'action')
            .select('*', f.col('query_params.*'))
            .drop('query_params')
            .drop('keyword')  # String
            .drop('area', 'dep', 'edu', 'expcat', 'incat',
                  'isnew', 'jobcat', 'cat', 'jobexp', 'ro',
                  'rostatus', 's9', 'wf', 'wt', 'zone')  # List[Int]
            .fillna(0)
        )

        trainset_sdf = (
            label_sdf.join(
                features_sdf,
                label_sdf.job == features_sdf.jobno,
                how='left'
            )
            # .drop('jobno', 'job')
            .drop('job')
            .repartition(16*4, 'gid')
            .sortWithinPartitions('gid', 'pos_in_list')
        )

        trainset_sdf = (
            trainset_sdf
            .withColumn('start_date', f.lit(self.start_date))
            .withColumn('stop_date', f.lit(self.stop_date))
            .withColumn('features_start_date', f.lit(self.features_start_date))
            .withColumn('features_stop_date', f.lit(self.features_stop_date))
        )

        self.write(trainset_sdf,
                   partitionBy=('start_date', 'stop_date',
                                'features_start_date', 'features_stop_date'),
                   compression='snappy')
        return

        groups_df = (trainset_sdf
                     .groupby('gid').count()
                     .toPandas())
        trainset_df = trainset_sdf.toPandas()

        return trainset_df, groups_df

    @auto_spark
    def to_libsvm(self, prefix, spark=None):

        base_path = self.get_base_path(prefix)

        dataset_path = base_path.with_suffix('.txt')
        dataset_spark_path = base_path.with_suffix('.txt.spark')
        query_path = base_path.with_suffix('.txt.query')
        query_spark_path = base_path.with_suffix('.txt.query.spark')

        shutil.rmtree(dataset_spark_path, ignore_errors=True)
        shutil.rmtree(query_spark_path, ignore_errors=True)

        sdf = (self.query(spark=spark, populate_if_empty=True)
               .repartition(16*4, 'gid')
               .sortWithinPartitions('gid', 'pos_in_list'))

        (sdf
         .drop('start_date', 'stop_date', 'features_start_date',
               'features_stop_date', 'date_stop', 'date_start')
         .rdd.map(row_to_libsvm)
         # .coalesce(1, False)
         .saveAsTextFile(dataset_spark_path.absolute().as_posix())
         )

        concat_files(sorted(dataset_spark_path.glob('part-*')), dataset_path)

        (sdf
         .rdd.mapPartitions(count_group_length)
         # .coalesce(1, False)
         .saveAsTextFile(query_spark_path.absolute().as_posix())
         )
        concat_files(sorted(query_spark_path.glob('part-*')), query_path)

    def get_base_path(self, prefix):
        base_path = Path(
            f'data/{prefix}-{self.start_date}_{self.stop_date}_'
            f'{self.features_start_date}_{self.features_stop_date}')
        return base_path


class Model:
    def __init__(self,
                 train_start_date, train_stop_date,
                 train_features_start_date, train_features_stop_date,

                 valid_start_date, valid_stop_date,
                 valid_features_start_date, valid_features_stop_date,
                 ):
        self.train_start_date = train_start_date
        self.train_stop_date = train_stop_date
        self.train_features_start_date = train_features_start_date
        self.train_features_stop_date = train_features_stop_date
        self.trainset = Dataset(
            start_date=train_start_date,
            stop_date=train_stop_date,
            features_start_date=train_features_start_date,
            features_stop_date=train_features_stop_date,
        )

        self.valid_start_date = valid_start_date
        self.valid_stop_date = valid_stop_date
        self.valid_features_start_date = valid_features_start_date
        self.valid_features_stop_date = valid_features_stop_date
        self.validset = Dataset(
            start_date=valid_start_date,
            stop_date=valid_stop_date,
            features_start_date=valid_features_start_date,
            features_stop_date=valid_features_stop_date,
        )

    @auto_spark
    def build_libSVM_files(self, spark=None):
        with print_duration():
            self.validset.to_libsvm('validset')

        with print_duration():
            self.trainset.to_libsvm('trainset')

    def generate_lightGBM_conf(self):
        train_conf_template = env.get_template('train.conf.j2')

        train_base_path = self.trainset.get_base_path('trainset')
        valid_base_path = self.validset.get_base_path('validset')

        trainset_path = train_base_path.with_suffix('.txt')
        validset_path = valid_base_path.with_suffix('.txt')

        train_conf_text = train_conf_template.render(
            trainset_path=trainset_path.as_posix(),
            validset_path=validset_path.as_posix())

        conf_path = Path(
            f'data/lightGBM-train-{self.train_start_date}-'
            f'valid-{self.valid_start_date}').with_suffix('.conf')

        conf_path.write_text(train_conf_text)
        print('config:', conf_path)


def concat_files(input_paths, output_path):
    with output_path.open('wb') as wfd:
        for input_path in input_paths:
            with input_path.open('rb') as rfd:
                shutil.copyfileobj(rfd, wfd, 1024*1024*10)


def count_group_length(rows):
    for gid, grp_rows in itt.groupby(rows, key=lambda row: row.gid):
        yield len(list(grp_rows))


@udfy(return_type=StringType())
def row_to_libsvm(row):
    # return (f'{row["rel"]} gid:{row["gid"]} pos:{row["pos_in_list"]} ' +
    #         ' '.join(f'{i}:{v!r}' for i, v in enumerate(row[3:]) if v != 0))
    features_strs = []
    for i, v in enumerate(row[3:]):
        if v == 0 or v is None or v is False:
            continue
        features_strs.append(f'{i}:{float(v)!r}')

    return (f'{row["rel"]} ' + ' '.join(features_strs))


def get_y_X(trainset_df):
    return (trainset_df['rel'],
            trainset_df.drop(
                columns=['gid', 'pos_in_list', 'rel',
                         'date_start', 'date_stop']))


@attr.s(auto_attribs=True)
class CVFold:
    train_start: int
    train_stop: int
    test_start: int
    test_stop: int


def series_cv(length, n_splits=3, step_ratio=0.2, train_ratio=0.7):

    step_size = round(step_ratio*length /
                      (1+step_ratio*(n_splits-1)))
    chunk_size = length - step_size*(n_splits-1)
    train_size = round(chunk_size * train_ratio)
    test_size = chunk_size - train_size
    for i in range(n_splits):
        shift = i*step_size
        yield CVFold(shift, shift + train_size - 1, shift + train_size,
                     shift + train_size + test_size-1)
