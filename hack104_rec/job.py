
from pyspark.sql import functions as f
from pyspark.sql.types import (ArrayType, LongType, StringType, StructField,
                               StructType, TimestampType, IntegerType)

from .core import auto_spark, udfy
from .misc import tokenize
from .data import Data, DataFormat, DataModelMixin
from .train_action import TrainActionProcessed
from .train_click import TrainClickExploded
from .utils import rm_repeat, to_halfwidth


class JobDateRange(DataModelMixin):
    data = Data('train-action-date-range.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        action_sdf = TrainActionProcessed.query()
        click_sdf = TrainClickExploded.query()

        action_sdf = (
            action_sdf
            .groupby('jobno')
            .agg(f.min('date').alias('action_date_start'),
                 f.max('date').alias('action_date_stop'))
        )
        click_sdf = (
            click_sdf
            .groupby('job_in_list')
            .agg(f.min('date').alias('click_date_start'),
                 f.max('date').alias('click_date_stop'))
        )
        sdf = (action_sdf
               .join(click_sdf,
                     action_sdf.jobno == click_sdf.job_in_list)
               .drop('job_in_list'))
        sdf = (sdf
               .withColumn(
                   'date_start',
                   f.when(
                       f.col('action_date_start') > f.col('click_date_start'),
                       f.col('click_date_start'))
                   .otherwise(f.col('action_date_start')))
               .withColumn(
                   'date_stop',
                   f.when(f.col('action_date_stop') < f.col('click_date_stop'),
                          f.col('click_date_stop'))
                   .otherwise(f.col('action_date_stop')))
               )
        cls.write(sdf)


class Job(DataModelMixin):
    data = Data('job.json', data_format=DataFormat.JSON)


class JobProcessed(DataModelMixin):
    data = Data('job-processed.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        sdf = (
            Job.query(spark)
            .select('custno',
                    f.col('jobno').cast(IntegerType()),
                    'worktime',
                    'description',
                    'job',
                    'others')
            .withColumn('description', tokenize_to_struct.udf('description'))
            .withColumn('job', tokenize_to_struct.udf('job'))
            .withColumn('others', tokenize_to_struct.udf('others'))
        )
        cls.write(sdf)


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



if __name__ == '__main__':

    text = 'AAABBBCCCC安安安安哈啊哈哈哈~~~~'

    print(simple_clean(text))
