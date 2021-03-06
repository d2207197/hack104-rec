
from collections import Counter

import numpy as np
from carriage import Stream
from pyspark.sql.types import (ArrayType, BooleanType, DoubleType, IntegerType,
                               LongType, MapType, ShortType, StringType,
                               StructField, StructType, TimestampType)

from .core import udfy


@udfy(return_type=ArrayType(ShortType()))
def score_relevance(joblist, jobno_list, action_list):
    if not isinstance(jobno_list, list):
        jobno_list = [jobno_list]
    if not isinstance(action_list, list):
        action_list = [action_list]

    job_to_rel_map = Counter()
    for j, a in zip(jobno_list, action_list):
        if a == 'clickJob':
            rel = 1
        elif a == 'clickSave':
            rel = 2
        elif a == 'clickApply':
            rel = 3

        job_to_rel_map[j] += rel

    return [job_to_rel_map.get(j, 0) for j in joblist]


@udfy(return_type=DoubleType())
def dcg_at_k(r, k=None, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    if k is None:
        k = len(r)

    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


@udfy(return_type=DoubleType())
def ndcg_at_k(r, k=None, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return np.asscalar(dcg_at_k(r, k, method) / dcg_max)


def ndcg_score_of_truth(y_truth):
    from carriage import Stream
    return (Stream(y_truth)
            .chunk(20)
            .map(lambda row: row.to_list())
            .map(lambda l: ndcg_at_k(l))
            .mean())


def ndcg_score_of_prediction(y_truth, y_pred):
    pred_groups_stm = (Stream(y_pred)
                       .chunk(20)
                       .map(lambda row: row.to_list()))

    test_y_groups_stm = (
        Stream(y_truth)
        .chunk(20)
        .map(lambda row: row.to_list()))

    return (pred_groups_stm
            .zip(test_y_groups_stm)
            .starmap(
                lambda pred_l, y_l:
                [y_e for pred_e, y_e in
                 sorted(zip(pred_l, y_l), reverse=True)])
            .map(lambda l: ndcg_at_k(l))
            .mean()
            )
