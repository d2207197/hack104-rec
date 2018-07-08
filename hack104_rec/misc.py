
import jieba
from carriage import Stream, X
from opencc import OpenCC
from pyspark.sql.types import (ArrayType, BooleanType, IntegerType, LongType,
                               MapType, ShortType, StringType, StructField,
                               StructType, TimestampType)

from .core import udfy

openCC = OpenCC('t2s')


@udfy(return_type=ArrayType(StringType()))
def tokenize(text):
    if text is None:
        return None

    return (Stream(jieba.cut_for_search(openCC.convert(text)))
            .filter(X != ' ').to_list())
