

from carriage import Stream, Row, X
from hack104_rec import query_string
from pyspark.sql import functions as f
from pyspark.sql.types import (ArrayType, LongType, StringType, StructField,
                               StructType, TimestampType, FloatType)
from elasticsearch_dsl import Search, connections, Q
from .core import auto_spark, udfy
from .data import Data, DataFormat, DataModelMixin
from .metric import ndcg_at_k, score_relevance
from .misc import tokenize
from .es.mapping import Job


class TrainClick(DataModelMixin):
    data = Data('train-click.json',
                data_format=DataFormat.JSON,
                spark_schema=StructType(
                    [
                        StructField('action', StringType(), False),
                        StructField('date', StringType(), False),
                        StructField('joblist', ArrayType(
                            StringType(), False), False),
                        StructField('jobno', StringType(), False),
                        StructField('querystring', StringType(), False),
                        StructField('source', StringType(), False)]
                )
                )


class TrainClickProcessed(DataModelMixin):
    data = Data('train-click-processed.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        sdf = (
            TrainClick.query(spark)
            .withColumn('id', f.monotonically_increasing_id())
            .withColumn('datetime',
                        (f.col('date').cast(LongType()) / 1000)
                        .cast(TimestampType()))
            .withColumn('date', f.to_date('datetime'))
            .withColumn('jobno', f.col('jobno').cast(LongType()))
            .withColumn('joblist',
                        f.col('joblist').cast(ArrayType(LongType())))
            .withColumn('query_params',
                        query_string.process_udf('querystring'))
            .withColumn('tokens', tokenize.udf('query_params.keyword'))
            .withColumn('rel_list',
                        score_relevance.udf('joblist', 'jobno', 'action'))
        )
        cls.write(sdf)

    @classmethod
    @auto_spark
    def calc_ndcg(cls, spark=None):
        return (
            cls.query()
            .withColumn('ndcg', ndcg_at_k.udf('rel_list'))
            .select(f.mean('ndcg').alias('ndcg'))
            .first()
        ).ndcg


@udfy(return_type=ArrayType(
    StructType([
        StructField('job', LongType()),
        StructField('rel', LongType()),
        StructField('action', StringType()),
    ])))
def zip_job_rel_act(joblist, rel_list, action_list, jobno_list):
    job_to_act_map = dict(zip(jobno_list, action_list))

    return [(j, r, job_to_act_map.get(j, ''))
            for j, r in zip(joblist, rel_list)]


class TrainClickGrouped(DataModelMixin):
    data = Data('train-click-grouped.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        sdf = (
            TrainClickProcessed.query()
            .groupby('source', 'date', 'datetime', 'query_params', 'joblist', 'tokens')
            .agg(f.collect_list('id').alias('id_list'),
                 f.collect_list('jobno').alias('jobno_list'),
                 f.collect_list('action').alias('action_list'))
            .withColumn('rel_list',
                        score_relevance.udf(
                            'joblist', 'jobno_list', 'action_list'))
            .withColumn('job_rel_list',
                        zip_job_rel_act.udf(
                            'joblist', 'rel_list', 'action_list', 'jobno_list'))
            .withColumn('gid', f.monotonically_increasing_id())
        )
        cls.write(sdf)

    @classmethod
    @auto_spark
    def calc_ndcg(cls, spark=None):
        return (
            cls.query()
            .withColumn('ndcg', ndcg_at_k.udf('rel_list'))
            .select(f.mean('ndcg').alias('ndcg'))
            .first()
        ).ndcg


class TrainClickExploded(DataModelMixin):
    data = Data('train-click-exploded.pq')

    @classmethod
    @auto_spark(('spark.driver.memory', '5g'),
                ('spark.executor.memory', '5g'))
    def populate(cls, spark=None):
        sdf = (
            TrainClickGrouped.query(spark)
            .withColumn('text_score', get_tfidf.udf('joblist', 'tokens'))
            .drop('joblist', 'rel_list', 'id_list', 'jobno_list', 'action_list')
            .select('*',
                    f.posexplode('job_rel_list')
                    .alias('pos_in_list', 'job_rel_in_list'))
            .drop('job_rel_list')
            .select('*', 'job_rel_in_list.*')
            .drop('job_rel_in_list')
        )

        cls.write(sdf,
                  compression='snappy')


class TrainClickCTR(DataModelMixin):
    data = Data('train-click-ctr.pq')

    @classmethod
    @auto_spark
    def populate(cls, spark=None):
        sdf = (
            TrainClickExploded.query()
            .withColumn('click', f.col('rel'))
            .groupby('date', 'source', 'action',
                     'query_params.keyword', 'job')
            .agg(f.count(f.lit(1)).alias('impr'),
                 f.sum('click').alias('click'))
            .withColumn('CTR', f.col('click') / f.col('impr'))

            .sort('action', 'impr', 'CTR', ascending=[False, False, False])
        )
        cls.write(sdf)


@udfy(return_type=ArrayType(
    StructType(
        [StructField('jobno', StringType())] + [StructField(field, FloatType()) for field in Job.tfidf_fields]
    )
))
def get_tfidf(joblist, tokens):
    if not tokens:
        return [tuple([job] + [0.0 for field in Job.tfidf_fields]) for job in joblist]

    joined_tokens = ' '.join(tokens)
    joblist = joblist

    q_multi_match = Q('multi_match',
                  query=joined_tokens,
                  type='most_fields',
                  fields=Job.tfidf_fields,
                  analyzer='whitespace')
    q_ids = Q('ids', values=joblist)
    q_overall = Q('bool', must=q_multi_match, filter=q_ids)
    client = connections.create_connection(hosts=['localhost:9201'], timeout=20)

    search = Search(using=client, index='hack104').query(q_overall).extra(explain=True)

    hits = (h for h in search.execute().hits.hits)

    def ugly_extract_fieldname(description):
        return description[7:description.find(':')]

    def extract_fields_scores(expl):
        return (Stream(expl['_explanation']['details'][0]['details'])
                .map(lambda d: Row(field=ugly_extract_fieldname(d['details'][0]['description']), score=d['value']))
                .to_map())

    scores_of_job = (Stream(hits)
     .map(extract_fields_scores)
    ).to_list()

    return [tuple([job] + [scores.get(field, 0.0) for field in Job.tfidf_fields]) for job, scores in zip(joblist, scores_of_job)]

