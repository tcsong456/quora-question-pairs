import re
import hashlib
import numpy as np
from tqdm import tqdm
from models.utils.build_vocab import BuildVocab

def normalize(s):
    s = (s or '').strip().lower()
    _ws = re.compile('\s+')
    s = _ws.sub(' ', s)
    return s

def stable_hash(s):
    return hashlib.blake2b(s.encode('utf-8'), digest_size=16).hexdigest()

def row_hash(df, id_col='id'):
    hashes = []
    row_iterator = tqdm(df.iterrows(),
                        total=df.shape[0])
    for _, row in row_iterator:
        q1, q2 = row['question1'], row['question2']
        q1_hash = stable_hash(normalize(q1))
        q2_hash = stable_hash(normalize(q2))
        hashes.append([row[id_col], q1_hash, q2_hash])
    hashes = np.vstack(hashes)
    return hashes
        

if __name__ == '__main__':
    bv = BuildVocab(
            'data/train.csv',
            'data/test.csv'
        )
    train = bv.train_data
    test = bv.test_data
    
    train_hash = row_hash(train)
    test_hash = row_hash(test, id_col='test_id')
    np.save('artifacts/train_hash.npy', train_hash)
    np.save('artifacts/test_hash.npy', test_hash)

#%%
