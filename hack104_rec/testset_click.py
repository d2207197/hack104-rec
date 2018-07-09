
from pyspark.sql import functions as f
from pyspark.sql.types import (ArrayType, LongType, StringType, StructField,
                               StructType)

from hack104_rec import query_string

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
            .withColumn('joblist',
                        f.col('joblist').cast(ArrayType(LongType())))
            .withColumn('query_params',
                        query_string.process_udf('querystring'))
            .withColumn('tokens', tokenize.udf('query_params.keyword'))
        )
        cls.write(sdf)
