import re
from pyspark.sql.types import (ArrayType, BooleanType, IntegerType, LongType,
                               MapType, ShortType, StringType, StructField,
                               StructType, TimestampType)
from .core import udfy
from .misc import tokenize

CHAR_REPEAT_REGEX = r'(.)\1+'
WORD_REPEAT_REGEX = r'(\S{2,}?)\1+'


@udfy(return_type=StructType([StructField('keyword', StringType()),
                             StructField('token', ArrayType(StringType()))]))
def tokenize_to_struct(text):
    tokens = tokenize(simple_clean(text))
    return (text, [tok for tok in tokens if len(tok) > 0] if text is not None else None)


def simple_clean(text):
    return (
        rm_repeat(to_halfwidth(text.replace('\n', ' ').replace('\r', ''))).lower().strip()
        if text is not None else None
    )


def to_halfwidth(query: str) -> str:
    """Convert the query string to halfwidth."""
    """
    全形字符 unicode 編碼從 65281 ~ 65374(十六進制 0xFF01 ~ 0xFF5E)
    半形字符 unicode 編碼從 33 ~ 126(十六進制 0x21~ 0x7E)
    空格比較特殊, 全形為12288(0x3000), 半形為32(0x20)
    而且除空格外, 全形/半形按 unicode 編碼排序在順序上是對應的
    所以可以直接通過用+-法來處理非空格字元, 對空格單獨處理.
    """
    rstring = ""
    for char in query:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xfee0
        if code < 0x0020 or code > 0x7e:  # fallback check
            rstring += char
        else:
            rstring += chr(code)
    return rstring


def to_lower(query: str) -> str:
    """Convert the query string to lowercase."""
    return query.lower()


def rm_repeat(query: str) -> str:
    return re.sub(
        WORD_REPEAT_REGEX, r'\1',
        re.sub(CHAR_REPEAT_REGEX, r'\1\1', query)
    )


if __name__ == '__main__':

    text = 'AAABBBCCCC安安安安哈啊哈哈哈~~~~'

    print(simple_clean(text))

