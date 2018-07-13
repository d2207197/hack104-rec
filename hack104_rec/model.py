import logging
import shutil
from pathlib import Path

import attr

import pyspark
from pyspark.sql import functions as f
from pyspark.sql.types import StringType

from .core import auto_spark, udfy
from .data import Data, DataModelMixin
from .job import JobDateRange, JobProcessed
from .train_action import TrainActionUnstacked
from .train_click import TrainClickExploded

logger = logging.getLogger(__name__)


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
            .join(open_days_sdf, 'jobno')
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
                   .drop('custno', 'description', 'job', 'others'))

        features_sdf = features_sdf.join(job_sdf, 'jobno')
        features_sdf = (
            features_sdf
            .withColumn('start_date', f.lit(self.start_date))
            .withColumn('stop_date', f.lit(self.stop_date)))

        self.write(features_sdf,
                   partitionBy=('start_date', 'stop_date'),
                   compression='snappy')


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
            .drop('source', 'date', 'datetime')
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
            .drop('jobno', 'job')
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

        base_path = Path(
            f'data/{prefix}-{self.start_date}_{self.stop_date}_'
            f'{self.features_start_date}_{self.features_stop_date}')

        dataset_path = base_path.with_suffix('.txt')
        dataset_spark_path = base_path.with_suffix('.txt.spark')
        query_path = base_path.with_suffix('.txt.query')
        query_spark_path = base_path.with_suffix('.txt.query.spark')

        shutil.rmtree(dataset_spark_path, ignore_errors=True)
        shutil.rmtree(query_spark_path, ignore_errors=True)

        sdf = self.query(spark=spark)
        (sdf
         .drop('start_date', 'stop_date', 'features_start_date',
               'features_stop_date', 'date_stop', 'date_start')
         .rdd.map(row_to_libsvm)
         .coalesce(1, False)
         .saveAsTextFile(dataset_spark_path.absolute().as_posix())
         )
        (dataset_spark_path / 'part-00000').replace(dataset_path)

        (sdf
         .groupby('gid')
         .count()
         .select(f.col('count').astype(StringType()))
         .rdd.map(lambda row: row['count'])
         .coalesce(1, False)
         .saveAsTextFile(query_spark_path.absolute().as_posix())
         )
        (query_spark_path / 'part-00000').replace(query_path)


@udfy(return_type=StringType())
def row_to_libsvm(row):
    # return (f'{row["rel"]} gid:{row["gid"]} pos:{row["pos_in_list"]} ' +
    #         ' '.join(f'{i}:{v!r}' for i, v in enumerate(row[3:]) if v != 0))
    return (f'{row["rel"]} ' +
            ' '.join(f'{i}:{v!r}' for i, v in enumerate(row[3:]) if v != 0))


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
