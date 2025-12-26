import operator
import numpy as np
from functools import reduce
from scipy.sparse.linalg import norm
from models.utils.build_vocab import BuildVocab
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_sparse(p, q, eps=1e-9):
    num = p.multiply(q).sum(axis=1).A1
    den = norm(p, axis=1) * norm(q, axis=1) + eps
    return num / den

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
    l1d = t1.multiply(t2).sum(axis=1).A1
    ov_ratio_1, ov_ratio_2 = mass_overlap(t1, t2)
    nnz_ratio = length_overlap(t1, t2)
    feats = np.stack([cos, l2d, l1d, ov_ratio_1, ov_ratio_2, nnz_ratio], axis=1)
    return feats

if __name__ == '__main__':
    bv = BuildVocab(
            'data/train.csv',
            'data/test.csv'
        )
    train = bv.train_data
    test = bv.test_data
    
    vec = TfidfVectorizer(
    ngram_range=(1,1),
    min_df=3, max_df=0.95,
    max_features=150000,
    )
    
    ids, features_tr, features_te = [], [], []
    y = train['is_duplicate']
    x = train.drop('is_duplicate', axis=1)
    skf = StratifiedKFold(n_splits=5, random_state=7610, shuffle=True)
    for train_idx, val_idx in skf.split(x, y):
        x_tr = x.iloc[train_idx]
        x_val = x.iloc[val_idx]
        
        corpus = x_tr['question1'].values.tolist() + x_tr['question2'].values.tolist()
        vec.fit(corpus)
        t1 = vec.transform(x_val['question1'].values.tolist())
        t2 = vec.transform(x_val['question2'].values.tolist())
        train_features = tfidf_features(t1, t2)
        features_tr.append(train_features)
        ids.append(x_val['id'])
        
        z1 = vec.transform(test['question1'].values.tolist())
        z2 = vec.transform(test['question2'].values.tolist())
        test_features = tfidf_features(z1, z2)
        features_te.append(test_features)
    
    features_tr = np.concatenate(features_tr)
    ids = np.concatenate(ids)[:, None]
    features_tr = np.concatenate([ids, features_tr], axis=1)
    features_tr = features_tr[np.argsort(features_tr[:, 0])].astype(np.float32)
    
    features_te = reduce(operator.add, features_te) / len(features_te)
    test_id = test['test_id'].values[:, None]
    features_te = np.concatenate([test_id, features_te], axis=1)
    np.save('artifacts/training/double_pair_tfidf.npy', features_tr)
    np.save('artifacts/prediction/double_pair_tfidf.npy', features_te)

#%%
