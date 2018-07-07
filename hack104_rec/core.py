# %cd / home/joe/hack104-rec
import os

import findspark
import jieba
from carriage import Stream, X
from opencc import OpenCC
from pyspark.sql import functions as f
from pyspark.sql.types import (ArrayType, BooleanType, IntegerType, LongType,
                               MapType, StringType, StructField, StructType,
                               TimestampType)


def init_spark_env():
    os.environ['SPARK_HOME'] = '/opt/spark'
    findspark.init()


def set_trace():
    from pudb.remote import set_trace
    set_trace(term_size=(170, 45))


def udfy(func=None, return_type=StringType()):
    def wrapper(func):
        func.udf = f.udf(func, returnType=return_type)
        return func

    if func is None:
        return wrapper
    else:
        return wrapper(func)


openCC = OpenCC('t2s')


@udfy(return_type=ArrayType(StringType()))
def tokenize(text):
    try:
        return (Stream(jieba.cut_for_search(openCC.convert(text)))
                .filter(X != ' ').to_list())
    except:
        print('text', text)
