import re
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from models.utils.build_vocab import BuildVocab
from sklearn.feature_extraction.text import TfidfVectorizer

space_re = re.compile('\s+')

def cosine_sparse(p, q):
    cos = p.multiply(q).sum(axis=1).A1
    return cos

def l2_distance(q1, q2):
    n1 = q1.multiply(q1).sum(axis=1).A1
    n2 = q2.multiply(q2).sum(axis=1).A1
    dot = q1.multiply(q2).sum(axis=1).A1
    return np.sqrt(np.maximum(n1+n2-2*dot, 0))

def mass_overlap(q1, q2):
    ov = q1.multiply(q2).sum(axis=1).A1
    s1 = q1.sum(axis=1).A1
    s2 = q2.sum(axis=1).A1
    ov_ratio_1 = ov / (s1 + 1e-9)
    ov_ratio_2 = ov / (s2 + 1e-9)
    return ov_ratio_1, ov_ratio_2

def length_overlap(q1, q2):
    nnz1 = np.diff(q1.indptr)
    nnz2 = np.diff(q2.indptr)
    nnz1 = np.sqrt(nnz1)
    nnz2 = np.sqrt(nnz2)
    nnz_min = np.minimum(nnz1, nnz2)
    nnz_max = np.maximum(nnz1, nnz2)
    nnz_ratio = nnz_min / (nnz_max + 1e-12)
    return nnz_ratio

def tfidf_features(t1, t2):
    cos = cosine_sparse(t1, t2)
    l2d = l2_distance(t1,t2)
    ov_ratio_1, ov_ratio_2 = mass_overlap(t1, t2)
    nnz_ratio = length_overlap(t1, t2)
    feats = np.stack([cos, l2d, ov_ratio_1, ov_ratio_2, nnz_ratio], axis=1)
    return feats

if __name__ == '__main__':
    bv = BuildVocab(
            'data/train.csv',
            'data/test.csv'
        )
    train = bv.train_data
    test = bv.test_data
    
    all_q = train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + \
        test['question2'].tolist()
    all_q = [space_re.sub(' ', q.lower().strip()) for q in all_q]
    
    vec = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=3, max_df=0.95,
        max_features=150000,
        norm='l2'
        )
    tfidf_vector = vec.fit_transform(all_q)
    
    N = train.shape[0]
    M = test.shape[0]
    q1_tfidf_tr = tfidf_vector[: N]
    q2_tfidf_tr = tfidf_vector[N: 2*N]
    q1_tfidf_te = tfidf_vector[2*N: 2*N+M]
    q2_tfidf_te = tfidf_vector[2*N+M:]
    
    x_tr = tfidf_features(q1_tfidf_tr, q2_tfidf_tr)
    x_te = tfidf_features(q1_tfidf_te, q2_tfidf_te)
    x_tr = np.concatenate([train['id'].values[:, None], x_tr], axis=1)
    x_te = np.concatenate([test['test_id'].values[:, None], x_te], axis=1)
    np.save('artifacts/training/tfidf_features.npy', x_tr.astype(np.float32))
    np.save('artifacts/prediction/tfidf_features.npy', x_te.astype(np.float32))