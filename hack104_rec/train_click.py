

from pyspark.sql import functions as f
from pyspark.sql.types import (ArrayType, BooleanType, IntegerType, LongType,
                               MapType, StringType, StructField, StructType,
                               TimestampType)

from hack104_rec import query_string

from .core import auto_spark
from .data import Data, DataFormat, DataModelMixin
from .metric import ndcg_at_k, score_relevance
from .misc import tokenize


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
                data_format=DataFormat.PARQUET)

    @classmethod
    @auto_spark
    def populate(cls, spark=None):

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
                        query_string.process_udf('querystring'))
            .withColumn('tokens', tokenize.udf('query_params.keyword'))
            .withColumn('date', f.to_date('datetime'))
        )
        cls.write(train_click_processed_sdf, mode='overwrite')

    @classmethod
    @auto_spark
    def query_ndcg(cls, spark=None):
        return (cls.query()
                .withColumn('rel_list', score_relevance.udf('jobno', 'joblist'))
                .withColumn('ndcg', ndcg_at_k.udf('rel_list'))
                .select(f.mean('ndcg').alias('ndcg')))


class TrainClickExploded(DataModelMixin):
    data = Data('train-click-exploded.p',
                data_format=DataFormat.PARQUET)

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        train_click_exploded_sdf = (
            TrainClickProcessed.query(spark)
            .withColumn('job_in_list', f.explode('joblist'))
        )

        cls.write(train_click_exploded_sdf,
                  mode='overwrite', compression='snappy')
