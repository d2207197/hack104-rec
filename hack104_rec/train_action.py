
from pyspark.sql import functions as f
from pyspark.sql.types import (LongType, StringType, StructField, StructType,
                               TimestampType)

from .core import auto_spark
from .data import Data, DataFormat, DataModelMixin


class TrainAction(DataModelMixin):
    data = Data('train-action.json',
                data_format=DataFormat.JSON,
                spark_schema=StructType(
                    [
                        StructField("jobno", StringType(), True),
                        StructField("date", StringType(), True),
                        StructField("action", StringType(), True),
                        StructField("source", StringType(), True),
                        StructField("device", StringType(), True),
                    ]
                )
                )


class TrainActionProcessed(DataModelMixin):
    data = Data('train-action-processed.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        sdf = (
            TrainAction.query()
            .withColumn('id', f.monotonically_increasing_id())
            .withColumn('datetime',
                        (f.col('date').cast(
                            LongType()) / 1000)
                        .cast(TimestampType()))
            .withColumn('date', f.to_date('datetime'))
            .withColumn('jobno', f.col('jobno').cast(LongType()))
        )
        cls.write(sdf)


class TrainActionCount(DataModelMixin):
    data = Data('train-action-count.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        sdf = (TrainActionProcessed.query()
               .groupby('date', 'source', 'action', 'jobno')
               .count()
               )
        cls.write(sdf, mode='overwrite')


class TrainActionUnstacked(DataModelMixin):
    data = Data('train-action-unstacked.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        sdf = TrainActionCount.query()
        SOURCES = ['web', 'mobileWeb', 'app']
        ACTIONS = ['viewJob', 'saveJob', 'applyJob']
        sdf = (sdf
               .groupby('date', 'jobno', 'source')
               .pivot('action', ACTIONS)
               .sum('count')
               .fillna(0)
               .withColumn('total',
                           sum(f.col(action) for action in ACTIONS))
               .groupby('date', 'jobno')
               .pivot('source', SOURCES)
               .sum('total', *ACTIONS)
               .fillna(0)
               )
        for action in ACTIONS + ['total']:
            for source in SOURCES:
                sdf = (sdf
                       .withColumnRenamed(f'{source}_sum({action})',
                                          f'{source}_{action}'))

            sdf = (sdf
                   .withColumn(f'{action}_total',
                               sum(f.col(f'{source}_{action}')
                                   for source in SOURCES)))

        cls.write(sdf)
