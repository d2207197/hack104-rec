__version__ = '0.1.0'

from .base import init_spark_env
from .core import auto_spark
from .data import Data, DataFormat, DataModelMixin
from .metric import ndcg_at_k, score_relevance
from .misc import tokenize
from .train_click import TrainClick, TrainClickProcessed

__all__ = ['TrainClickProcessed', 'TrainClick', 'auto_spark',
           'tokenize', 'Data', 'DataFormat', 'DataModelMixin',
           'ndcg_at_k', 'score_relevance'
           ]
init_spark_env()
