
from hack104_rec import query_string
from pyspark.sql import functions as f
from pyspark.sql.types import (ArrayType, LongType, StringType, StructField,
                               StructType)

from .core import auto_spark
from .data import Data, DataFormat, DataModelMixin
from .misc import tokenize


class TestsetClick(DataModelMixin):
    data = Data('testset-click.json',
                data_format=DataFormat.JSON,
                spark_schema=StructType(
                    [
                        StructField('id', LongType(), False),
                        StructField('joblist', ArrayType(
                            StringType(), False), False),
                        StructField('querystring', StringType(), False),
                    ]
                )
                )


class TestsetClickProcessed(DataModelMixin):
    data = Data('testset-click-processed.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        sdf = (
            TestsetClick.query(spark)
            .withColumnRenamed('id', 'gid')
            .withColumn('joblist',
                        f.col('joblist').cast(ArrayType(LongType())))
            .withColumn('query_params',
                        query_string.process_udf('querystring'))
            .withColumn('tokens', tokenize.udf('query_params.keyword'))
        )
        cls.write(sdf)


# class TestClickGrouped(DataModelMixin):
#     data = Data('test-click-grouped.pq')
#
#     @classmethod
#     @auto_spark
#     def populate(cls, spark=None):
#         sdf = (
#             TrainClickProcessed.query()
#             .groupby('query_params', 'joblist')
#             # .groupby('source', 'date', 'datetime', 'query_params', 'joblist')
#             .agg(f.collect_list('id').alias('id_list'),
#                  f.collect_list('jobno').alias('jobno_list'),
#                  f.collect_list('action').alias('action_list'))
#             .withColumn('rel_list',
#                         score_relevance.udf(
#                             'joblist', 'jobno_list', 'action_list'))
#             .withColumn('job_rel_list', zip_job_rel.udf('joblist', 'rel_list'))
#             .withColumn('gid', f.monotonically_increasing_id())
#         )
#         cls.write(sdf)


class TestsetClickExploded(DataModelMixin):
    data = Data('test-click-exploded.pq')

    @classmethod
    @auto_spark(('spark.driver.memory', '5g'),
                ('spark.executor.memory', '5g'))
    def populate(cls, spark=None):
        sdf = (
            TestsetClickProcessed.query(spark)
            # .drop('joblist', 'rel_list', 'id_list', 'jobno_list', 'action_list')
            .select('*',
                    f.posexplode('joblist')
                    .alias('pos_in_list', 'job')
                    )
            .withColumn('rel', f.lit(0))
            .drop('joblist')
            # .select('*', 'job_in_list.*')
            # .drop('job_in_list')
        )

        cls.write(sdf,
                  compression='snappy')
