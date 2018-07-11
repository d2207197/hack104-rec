# %cd / home/joe/hack104-rec
import urllib
import urllib.parse

import attr
from pyspark.sql import functions as f
from pyspark.sql.types import (ArrayType, BooleanType, IntegerType, LongType,
                               MapType, StringType, StructField, StructType,
                               TimestampType)


@attr.s(slots=True)
class QueryParamProcessor:
    param_name = attr.ib()
    type_ = attr.ib()
    process_param = attr.ib()


class QueryStringProcessor:
    name_to_qpp_map = {}

    @classmethod
    def register(cls, param_name, type_, process_param_func=None):
        return cls.register_many([param_name], type_, process_param_func)

    @classmethod
    def register_many(cls, param_names, type_, process_param_func=None):
        if process_param_func is None:
            def deco(process_param_func):
                cls._register_many(param_names, type_, process_param_func)
            return deco

        else:
            cls._register_many(param_names, type_, process_param_func)

    @classmethod
    def _register_many(cls, param_names, type_, process_param_func):
        for param_name in param_names:
            qpp = QueryParamProcessor(
                param_name,
                type_, process_param_func)
            cls.name_to_qpp_map[param_name] = qpp

    @classmethod
    def process(cls, query_string):
        query_string = urllib.parse.urldefrag(query_string).url
        query_string_d = dict(urllib.parse.parse_qsl(query_string))

        def _process_param(value, qpp):
            # if value is None:
            #     return value

            return qpp.process_param(value)

        query_string_processed = tuple(
            _process_param(query_string_d.get(param_name), qpp)
            for param_name, qpp in cls.name_to_qpp_map.items()
        )

        return query_string_processed

    @classmethod
    def get_spark_type(cls):
        return StructType([StructField(qpp.param_name, qpp.type_)
                           for name, qpp in cls.name_to_qpp_map.items()])


@QueryStringProcessor.register('keyword', IntegerType())
def keyword_processor(k):
    return k


@QueryStringProcessor.register('sctp', IntegerType())
def sctp_processor(v):
    if v == 'P':
        return 1
    elif v == 'S':
        return 2
    else:
        return 0


@QueryStringProcessor.register('mode', IntegerType())
def mode_processor(v):
    if v == 's':
        return 1
    elif v == 'l':
        return 2
    else:
        return 0


@QueryStringProcessor.register('wktm', IntegerType())
def wktm_processor(v):
    if v == '週休二日':
        return 1
    elif v == '隔週休':
        return 2
    else:
        return 0


QueryStringProcessor.register_many(
    [
        'area', 'dep', 'edu', 'expcat', 'incat',
        'isnew', 'jobcat', 'cat', 'jobexp', 'ro',
        'rostatus', 's9', 'wf', 'wt', 'zone'
    ],
    ArrayType(IntegerType()),
    lambda intarr: [int(e) for e in intarr.split(',')
                    ] if intarr is not None else []
)


QueryStringProcessor.register_many(
    ['order', 'asc', 'scmax', 'scmin'],
    IntegerType(),
    lambda int_str: int(int_str) if int_str is not None else -1
)


@QueryStringProcessor.register('kwop', BooleanType())
def kwop_processor(v):
    if v == '1':
        return True
    return False


@QueryStringProcessor.register('m', BooleanType())
def m_processor(v):
    if v == '1':
        return True
    return False


@QueryStringProcessor.register('s5', IntegerType())
def s5_processor(v):
    if v == '256':
        return True
    return False


@QueryStringProcessor.register('sr', IntegerType())
def sr_processor(v):
    if v == '99':
        return True
    return False


process = QueryStringProcessor.process
process_udf = f.udf(
    QueryStringProcessor.process,
    returnType=QueryStringProcessor.get_spark_type())


if __name__ == '__main__':
    print(process(
        'ro=0&isnew=3&kwop=7&keyword=軟體&indcat=1001000000&'
        'edu=2%2C3&order=1&asc=0&zone=1%2C21&s9=4%2C8&'
        'page=1&jobexp=3%2C10&wt=2%2C8&mode=s&jobsource=n104bank1')
    )
