import os
from contextlib import contextmanager

import findspark
import pyspark
from pyspark.conf import SparkConf
from pyspark.sql import functions as f
from pyspark.sql.session import SparkSession
from pyspark.sql.types import (ArrayType, BooleanType, IntegerType, LongType,
                               MapType, ShortType, StringType, StructField,
                               StructType, TimestampType)


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


def auto_spark(f=None, *configs):
    spark_conf = SparkConf()

    def wrapper(*args, **kwargs):
        if (all(not isinstance(arg, SparkSession)
                for arg in args) and
                'spark' not in kwargs):

            spark = (
                SparkSession
                .builder
                .config(conf=spark_conf)
                .getOrCreate())

            ret = wrapper.func(*args, spark=spark, **kwargs)

        else:
            ret = f(*args, **kwargs)

        return ret

    if callable(f):
        wrapper.func = f
        return wrapper

    if f is not None:
        configs = (f,) + configs

    spark_conf.setAll(configs)

    def deco(f):
        wrapper.func = f
        return wrapper
    return deco
