

from pyspark.sql import functions as f
from pyspark.sql.types import (ArrayType, LongType, StringType, StructField,
                               StructType, TimestampType)

from hack104_rec import query_string

from .core import auto_spark, udfy
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
            .withColumn('rel_list',
                        score_relevance.udf('joblist', 'jobno', 'action'))
        )
        cls.write(sdf)

    @classmethod
    @auto_spark
    def calc_ndcg(cls, spark=None):
        return (
            cls.query()
            .withColumn('ndcg', ndcg_at_k.udf('rel_list'))
            .select(f.mean('ndcg').alias('ndcg'))
            .first()
        ).ndcg


@udfy(return_type=ArrayType(
    StructType([
        StructField('job', LongType()),
        StructField('rel', LongType()),
    ])))
def zip_job_rel(joblist, rel_list):
    return [(j, r) for j, r in zip(joblist, rel_list)]


class TrainClickGrouped(DataModelMixin):
    data = Data('train-click-grouped.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        sdf = (
            TrainClickProcessed.query()
            .groupby('source', 'date', 'datetime', 'query_params', 'joblist')
            .agg(f.collect_list('id').alias('id_list'),
                 f.collect_list('jobno').alias('jobno_list'),
                 f.collect_list('action').alias('action_list'))
            .withColumn('rel_list',
                        score_relevance.udf(
                            'joblist', 'jobno_list', 'action_list'))
            .withColumn('job_rel_list', zip_job_rel.udf('joblist', 'rel_list'))
            .withColumn('gid', f.monotonically_increasing_id())
        )
        cls.write(sdf)

    @classmethod
    @auto_spark
    def calc_ndcg(cls, spark=None):
        return (
            cls.query()
            .withColumn('ndcg', ndcg_at_k.udf('rel_list'))
            .select(f.mean('ndcg').alias('ndcg'))
            .first()
        ).ndcg


class TrainClickExploded(DataModelMixin):
    data = Data('train-click-exploded.pq')

    @classmethod
    @auto_spark(('spark.driver.memory', '5g'),
                ('spark.executor.memory', '5g'))
    def populate(cls, spark=None):
        sdf = (
            TrainClickGrouped.query(spark)
            .drop('joblist', 'rel_list', 'id_list', 'jobno_list', 'action_list')
            .select('*',
                    f.posexplode('job_rel_list')
                    .alias('pos_in_list', 'job_rel_in_list'))
            .drop('job_rel_list')
            .select('*', 'job_rel_in_list.*')
            .drop('job_rel_in_list')
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
            .withColumn('click', f.col('rel'))
            .groupby('date', 'source', 'action',
                     'query_params.keyword', 'job_in_list')
            .agg(f.count(f.lit(1)).alias('impr'),
                 f.sum('click').alias('click'))
            .withColumn('CTR', f.col('click') / f.col('impr'))
            .sort('action', 'impr', 'CTR', ascending=[False, False, False])
        )
        cls.write(sdf)
