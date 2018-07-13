
from pyspark.sql import functions as f
from pyspark.sql.types import (ArrayType, LongType, StringType, StructField,
                               StructType, TimestampType, IntegerType)

from .core import auto_spark, udfy
from .misc import tokenize
from .data import Data, DataFormat, DataModelMixin
from .train_action import TrainActionProcessed
from .train_click import TrainClickExploded
from .utils import rm_repeat, to_halfwidth, tokenize_to_struct


class Company(DataModelMixin):
    data = Data('company.json', data_format=DataFormat.JSON)


class CompanyProcessed(DataModelMixin):
    data = Data('company-processed.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        sdf = (
            Company.query()
            .select('*')
            .withColumn('name', tokenize_to_struct.udf('name'))
            .withColumn('management', tokenize_to_struct.udf('management'))
            .withColumn('product', tokenize_to_struct.udf('product'))
            .withColumn('profile', tokenize_to_struct.udf('profile'))
            .withColumn('welfare', tokenize_to_struct.udf('welfare'))
        )

        cls.write(sdf)

