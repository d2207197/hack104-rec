# %cd / home/joe/hack104-rec
import urllib
import urllib.parse

import attr
from carriage import Row
from icecream import ic
from pyspark.sql import Row
from pyspark.sql import functions as f
from pyspark.sql.types import (ArrayType, BooleanType, IntegerType, LongType,
                               MapType, StringType, StructField, StructType,
                               TimestampType)

# class type_processor:
#     type_to_processor_map = {}

#     def __init__(self, type_cls):
#         self.type_cls = type_cls

#     def __call__(self, f):
#         self.type_to_processor_map[self.type_cls] = f
#         return f

#     @classmethod
#     def process(cls, type_, value_str):
#         type_cls = type(type_)
#         if type_cls is ArrayType:
#             value_strs = value_str.split(',')
#             elem_type = type_.elementType
#             return [cls.process(elem_type, value_str)
#                     for value_str in value_strs]

#         else:
#             processor = cls.type_to_processor_map[type_cls]
#             return processor(value_str)


# @type_processor(IntegerType)
# def int_processor(int_str):
#     return int(int_str)


# @type_processor(StringType)
# def str_processor(s):
#     return s


@attr.s(slots=True)
class QueryParamProcessor:
    param_name = attr.ib()
    type_ = attr.ib()
    process_param = attr.ib()


class query_param_processor:
    name_to_qpp_map = {}

    def __init__(self, param_name, type_):
        self.param_name = param_name
        self.register(param_name, type_)

    @classmethod
    def register(cls, param_name, type_, process_param_func=None):
        qpp = QueryParamProcessor(
            param_name,
            type_, process_param_func)
        cls.name_to_qpp_map[param_name] = qpp

    def __call__(self, process_param_func):
        qpp = self.name_to_qpp_map[self.param_name]
        qpp.process_param = process_param_func

    @classmethod
    def process(cls, query_string) -> Row:
        query_string = urllib.parse.urldefrag(query_string).url
        query_string_d = dict(urllib.parse.parse_qsl(query_string))

        def _process_param(value, qpp):
            if value is None:
                return value

            return qpp.process_param(value)

        try:
            query_string_processed_d = {
                param_name: _process_param(query_string_d.get(param_name), qpp)
                for param_name, qpp in cls.name_to_qpp_map.items()
            }
        except:
            from icecream import ic
            print(query_string_d, query_string)
            ic(query_string_d, query_string)
            raise

        return Row(**query_string_processed_d)

    @classmethod
    def udf(cls, *args, **kwargs):
        return f.udf(cls.process, returnType=cls.get_type())(*args, **kwargs)

    @classmethod
    def get_type(cls):
        return StructType([StructField(qpp.param_name, qpp.type_)
                           for name, qpp in cls.name_to_qpp_map.items()])


def str_processor(s):
    return s


def int_processor(i):
    ic(i)

    return int(i)


def intarr_processor(intarr):
    return [int_processor(e) for e in intarr.split(',')]


STR_PARAMS = ['mode', 'keyword', 'sctp']

for param in STR_PARAMS:
    query_param_processor.register(
        param,
        StringType(),
        str_processor
    )


INTARR_PARAMS = [
    'area', 'dep', 'edu', 'expcat', 'incat', 'isnew', 'jobcat', 'cat', 'jobexp', 'ro', 'rostatus', 's9', 'wf', 'wt', 'zone'
]
for param in INTARR_PARAMS:
    query_param_processor.register(
        param,
        ArrayType(IntegerType()),
        intarr_processor
    )

INT_PARAMS = ['order', 'asc', 'scmax', 'scmin', 'wktm']
for param in INT_PARAMS:
    query_param_processor.register(
        param,
        IntegerType(),
        int_processor
    )


@query_param_processor('kwop', BooleanType())
def kwop_processor(v):
    if v == '1':
        return True
    return False


@query_param_processor('m', BooleanType())
def m_processor(v):
    if v == '1':
        return True
    return False


@query_param_processor('s5', IntegerType())
def s5_processor(v):
    if v == '256':
        return True
    return False


@query_param_processor('sr', IntegerType())
def sr_processor(v):
    if v == '99':
        return True
    return False


if __name__ == '__main__':
    print(query_param_processor.process(
        'ro=0&isnew=3&kwop=7&keyword=軟體&indcat=1001000000&'
        'edu=2%2C3&order=1&asc=0&zone=1%2C21&s9=4%2C8&'
        'page=1&jobexp=3%2C10&wt=2%2C8&mode=s&jobsource=n104bank1')
    )
