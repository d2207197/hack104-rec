# %cd / home/joe/hack104-rec


from enum import Enum, auto
from pathlib import Path

import attr
from pyspark.sql.types import (ArrayType, BooleanType, IntegerType, LongType,
                               MapType, StringType, StructField, StructType,
                               TimestampType)

from .core import with_spark


class DataFormat(Enum):
    JSON = auto()
    PARQUET = auto()


BASE_PATH: Path = Path('data')


@attr.s(auto_attribs=True)
class Data:
    path: str
    data_format: DataFormat = DataFormat.PARQUET
    spark_schema: StructType = None

    @property
    def full_path(self):
        return str(BASE_PATH / self.path)


class DataModelMixin():
    @property
    def data(self) -> Data:
        raise NotImplementedError

    @classmethod
    @with_spark
    def query(cls, spark=None):
        if cls.data.data_format is DataFormat.JSON:
            return spark.read.json(
                cls.data.full_path,
                schema=cls.data.spark_schema)
        elif cls.data.data_format is DataFormat.PARQUET:
            return spark.read.parquet(cls.data.full_path)
        else:
            raise ValueError('Unknown DataFormat')

    @classmethod
    def write(cls, sdf, **write_kwargs):
        if cls.data.data_format is DataFormat.JSON:
            sdf.write.json(cls.data.full_path, **write_kwargs)
        elif cls.data.data_format is DataFormat.PARQUET:
            sdf.write.parquet(cls.data.full_path, **write_kwargs)
        else:
            raise ValueError('Unknown DataFormat')
