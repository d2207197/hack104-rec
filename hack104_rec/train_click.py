

from pyspark.sql import functions as f
from pyspark.sql.types import (ArrayType, LongType, StringType, StructField,
                               StructType, TimestampType)

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
    data = Data('train-click-processed.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        sdf = (
            TrainClick.query(spark)
            .withColumn('id', f.monotonically_increasing_id())
            .withColumn('datetime',
                        (f.col('date').cast(LongType()) / 1000)
                        .cast(TimestampType()))
            .withColumn('date', f.to_date('datetime'))
            .withColumn('jobno', f.col('jobno').cast(LongType()))
            .withColumn('joblist',
                        f.col('joblist').cast(ArrayType(LongType())))
            .withColumn('query_params',
                        query_string.process_udf('querystring'))
            .withColumn('tokens', tokenize.udf('query_params.keyword'))
        )
        cls.write(sdf)

    @classmethod
    @auto_spark
    def query_ndcg(cls, spark=None):
        return (cls.query()
                .withColumn('rel_list',
                            score_relevance.udf('jobno', 'joblist'))
                .withColumn('ndcg', ndcg_at_k.udf('rel_list'))
                .select(f.mean('ndcg').alias('ndcg')))


class TrainClickExploded(DataModelMixin):
    data = Data('train-click-exploded.p',
                data_format=DataFormat.PARQUET)

    @classmethod
    @auto_spark(('spark.driver.memory', '5g'),
                ('spark.executor.memory', '5g'))
    def populate(cls, spark=None):
        sdf = (
            TrainClickProcessed.query(spark)
            .select('*',
                    f.posexplode('joblist')
                    .alias('pos_in_list', 'job_in_list'))
            .drop('joblist')
        )

        cls.write(sdf,
                  compression='snappy')


class TrainClickCTR(DataModelMixin):
    data = Data('train-click-ctr.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        sdf = (
            TrainClickExploded.query()
            .withColumn('click',
                        f.when(f.col('jobno') ==
                               f.col('job_in_list'), 1)
                        .otherwise(0))
            .groupby('date', 'source', 'action',
                     'query_params.keyword', 'job_in_list')
            .agg(f.count(f.lit(1)).alias('impr'),
                 f.sum('click').alias('click'))
            .withColumn('CTR', f.col('click') / f.col('impr'))
            .sort('action', 'impr', 'CTR', ascending=[False, False, False])
        )
        cls.write(sdf)
