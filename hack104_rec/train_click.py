# %cd / home/joe/hack104-rec


import pyspark
from pyspark.sql import functions as f
from pyspark.sql.types import (ArrayType, BooleanType, IntegerType, LongType,
                               MapType, StringType, StructField, StructType,
                               TimestampType)

from hack104_rec.query_param import query_param_processor

from .core import tokenize
from .data import Data, DataFormat, DataModelMixin


class TrainClick(DataModelMixin):
    data = Data('train-click.json',
                data_format=DataFormat.JSON,
                spark_schema=StructType(
                    [
                        StructField('action', StringType(), False),
                        StructField('date', StringType(), False),
                        StructField('joblist', ArrayType(
                            StringType(), False), False),
                        StructField('jobno', StringType(), False),
                        StructField('querystring', StringType(), False),
                        StructField('source', StringType(), False)]
                )
                )


class TrainClickProcessed(DataModelMixin):
    data = Data('train-click-processed.pq',
                data_format=DataFormat.PARQUET
                )

    @classmethod
    def populate(cls):
        spark = (
            pyspark.sql.SparkSession
            .builder
            .appName(f'{cls.__name__}.populate()')
            # .config('spark.driver.memory', '5g')
            # .config('spark.executor.memory', '5g')
            .getOrCreate())

        train_click_processed_sdf = (
            TrainClick.query(spark)
            .withColumn('id', f.monotonically_increasing_id())
            .withColumn('datetime',
                        (f.col('date').cast(LongType()) / 1000)
                        .cast(TimestampType()))
            .withColumn('jobno', f.col('jobno').cast(LongType()))
            .withColumn('joblist',
                        f.col('joblist').cast(ArrayType(LongType())))
            .withColumn('query_params',
                        query_param_processor.udf('querystring'))
            .withColumn('tokens', tokenize.udf('query_params.keyword'))
            .withColumn('date', f.to_date('datetime'))
        )
        cls.write(train_click_processed_sdf, mode='overwrite')
        spark.stop()
