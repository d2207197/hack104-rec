'''Insert documents from STDIN to elasticsearch.'''

from multiprocessing import Pool
import fileinput
import json
from elasticsearch.helpers import parallel_bulk
from elasticsearch_dsl import connections
import time
from .mapping import Job



POOL_NUM = 8
client = connections.create_connection(hosts=['localhost:9201'], timeout=20)



class JsonPreprocessor:
    def __call__(self, line):
        try:
            return Job.from_dict(json.loads(line)).to_dict(include_meta=True) if len(line.strip()) > 0 else None
        except:
            try:
                print('WTF', line)
            except:
                pass
            return None

class RowPreprocessor:
    def __call__(self, row):
        return Job.from_row(row).to_dict(include_meta=True) if row is not None else None



def bulk_insert(batch, preprocessor=JsonPreprocessor()):
    tic = time.time()
    print('Start preprocessing')
    with Pool(POOL_NUM) as p:
        documents = (r for r in p.map(preprocessor, batch) if isinstance(r, dict))
    toc = time.time()
    print(f'Preprocessing finished, elapsed time: {toc - tic:.3f} sec.')

    # d = list(documents)
    # print(len(d))
    # print(d[0])
    # print(d[-1])
    tic = time.time()
    print('Start `parallel_bulk` insert to es')
    for i, (success, info) in enumerate(parallel_bulk(client, documents), 1):
        if not success:
            print('Doc failed', info)
    toc = time.time()
    print(f'Insert finished, {i} docs inserted, elapsed time: {toc - tic:.3f} sec.')



if __name__ == '__main__':
    bulk_insert(fileinput.input())

