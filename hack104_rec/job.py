
from pyspark.sql import functions as f

from .core import auto_spark
from .data import Data, DataModelMixin
from .train_action import TrainActionProcessed
from .train_click import TrainClickExploded


class JobDateRange(DataModelMixin):
    data = Data('train-action-date-range.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        action_sdf = TrainActionProcessed.query()
        click_sdf = TrainClickExploded.query()

        action_sdf = (
            action_sdf
            .groupby('jobno')
            .agg(f.min('date').alias('action_date_start'),
                 f.max('date').alias('action_date_stop'))
        )
        click_sdf = (
            click_sdf
            .groupby('job_in_list')
            .agg(f.min('date').alias('click_date_start'),
                 f.max('date').alias('click_date_stop'))
        )
        sdf = (action_sdf
               .join(click_sdf,
                     action_sdf.jobno == click_sdf.job_in_list)
               .drop('job_in_list'))
        sdf = (sdf
               .withColumn(
                   'date_start',
                   f.when(
                       f.col('action_date_start') > f.col('click_date_start'),
                       f.col('click_date_start'))
                   .otherwise(f.col('action_date_start')))
               .withColumn(
                   'date_stop',
                   f.when(f.col('action_date_stop') < f.col('click_date_stop'),
                          f.col('click_date_stop'))
                   .otherwise(f.col('action_date_stop')))
               )
        cls.write(sdf)
