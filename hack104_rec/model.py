import attr

from pyspark.sql import functions as f

from .core import auto_spark
from .data import Data, DataFormat, DataModelMixin
from .job import JobDateRange, JobProcessed
from .train_action import TrainActionUnstacked
from .train_click import TrainClickExploded


class Features(DataModelMixin):

    data = Data('features.pq')

    @classmethod
    @auto_spark
    def populate(cls, start_date, stop_date, spark=None):
        open_days_sdf = (
            JobDateRange.query(spark=spark)
            .select(
                'jobno',
                f.when(f.col('date_stop') < stop_date, f.col('date_stop'))
                .otherwise(stop_date).alias('date_stop'),

                f.when(f.col('date_start') > start_date, f.col('date_start'))
                .otherwise(start_date).alias('date_start'),
            )
            .withColumn('open_days',
                        f.datediff(f.col('date_stop'),
                                   f.col('date_start')) + 1)
        )

        features_sdf = (
            TrainActionUnstacked.query(spark=spark)
            .filter(
                (f.col('date') >= start_date) &
                (f.col('date') <= stop_date))
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
            .withColumn('start_date', f.lit(start_date))
            .withColumn('stop_date', f.lit(stop_date)))

        (features_sdf.write
         .mode('overwrite')
         .partitionBy('start_date', 'stop_date')
         .option("compression", "snappy")
         .parquet('data/features.pq'))


@auto_spark(('spark.driver.memory', '10g'),
            ('spark.executor.memory', '10g'))
def query_dataset(start_date, stop_date,
                  features_start_date=None, features_stop_date=None,
                  spark=None):

    if features_start_date is None and features_stop_date is None:
        features_start_date = start_date
        features_stop_date = stop_date

    features_sdf = query_job_features(
        features_start_date, features_stop_date, spark=spark)

    label_sdf = (
        TrainClickExploded.query(spark=spark)
        .filter(
            (f.col('date') >= start_date) &
            (f.col('date') <= stop_date))
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
        .repartition(16*4, 'gid').sortWithinPartitions('gid', 'pos_in_list')
    )

    groups_df = (trainset_sdf
                 .groupby('gid').count()
                 .toPandas())
    trainset_df = trainset_sdf.toPandas()

    return trainset_df, groups_df


def get_y_X(trainset_df):
    return (trainset_df['rel'],
            trainset_df.drop(
                columns=['gid', 'pos_in_list', 'rel', 'date_start', 'date_stop']))


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
        yield CVFold(shift, shift + train_size - 1, shift + train_size, shift + train_size + test_size-1)
