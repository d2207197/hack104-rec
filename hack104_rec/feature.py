from pyspark.sql import functions as f

from .job import JobDateRange
from .train_action import TrainActionUnstacked
from .train_click import TrainClickExploded


def query_features(start_date, stop_date):
    open_days_sdf = (
        JobDateRange.query()
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
        TrainActionUnstacked.query()
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

    return features_sdf


def query_trainset(start_date, stop_date):
    label_sdf = (
        TrainClickExploded.query()
        .filter(
            (f.col('date') >= start_date) &
            (f.col('date') <= stop_date))
        .repartition(16*4, 'gid').sortWithinPartitions('gid', 'pos_in_list')
    )
    features_sdf = query_features(start_date, stop_date)
    trainset = (
        label_sdf.join(
            features_sdf,
            label_sdf.job == features_sdf.jobno,
            how='left'
        )
        .drop('source', 'date', 'datetime', 'query_params')
        .drop('gid', 'jobno', 'job')
        .sortWithinPartitions('gid', 'pos_in_list')
    )

    return trainset
