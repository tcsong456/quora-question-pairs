import operator
import numpy as np
from functools import reduce
from sklearn.pipeline import make_pipeline
from models.utils.build_vocab import BuildVocab
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def _entropy(p, eps=1e-9):
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)

def _jsd(p, q, eps=1e-9):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(p) - np.log(m))).sum(axis=1)
    kl_qm = (q * (np.log(q) - np.log(m))).sum(axis=1)
    return 0.5 * (kl_pm + kl_qm)

def _cosine(p, q, eps=1e-9):
    return (p * q).sum(axis=1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1) + eps)

def lda_pairwise_features(q1, q2, eps=1e-9):
    abs_diff = abs(q1 - q2)
    prod = q1 * q2
    
    dot = prod.sum(axis=1)
    l1 = abs_diff.sum(axis=1)
    l2 = np.sqrt((abs_diff**2).sum(axis=1))
    jsd = _jsd(q1, q2, eps)
    cos = _cosine(q1, q2, eps)
    ent1 = _entropy(q1)
    ent2 = _entropy(q2)
    ent_diff = abs(ent1-ent2)
    argmax_equal = (np.argmax(q1, axis=1) == np.argmax(q2, axis=1)).astype(np.float32)
    scalars = np.stack([dot, l1, l2, jsd, cos, ent_diff, argmax_equal], axis=1)
    output = np.concatenate([scalars, abs_diff, prod], axis=1).astype(np.float32)
    return output

if __name__ == '__main__':
    bv = BuildVocab(
            'data/train.csv',
            'data/test.csv'
        )
    train = bv.train_data
    test = bv.test_data
    vectorizer = CountVectorizer(
            ngram_range=(1, 1),
            max_df=0.95,
            min_df=2,
            max_features=20000,
            stop_words='english'
        )
    lda = LatentDirichletAllocation(
            n_components=10,
            learning_method='batch',
            max_iter=10,
            n_jobs=4,
            random_state=951,
            verbose=1
        )
    pl = make_pipeline(vectorizer,
                        lda)
    
    features, test_features, ids = [], [], []
    y = train['is_duplicate']
    x = train.drop('is_duplicate', axis=1)
    skf = StratifiedKFold(n_splits=5, random_state=7610, shuffle=True)
    for train_idx, val_idx in skf.split(x, y):
        x_tr = x.iloc[train_idx]
        x_val = x.iloc[val_idx]
        
        corpus = x_tr['question1'].values.tolist() + x_tr['question2'].values.tolist()
        pl.fit(corpus)
        q1_lda = pl.transform(x_val['question1'].values.tolist())
        q2_lda = pl.transform(x_val['question2'].values.tolist())
        feats = lda_pairwise_features(q1_lda, q2_lda)
        id = x_val['id'].values
        ids.append(id)
        features.append(feats)
        
        q1_lda_test = pl.transform(test['question1'].values.tolist())
        q2_lda_test = pl.transform(test['question2'].values.tolist())
        test_feats = lda_pairwise_features(q1_lda_test, q2_lda_test)
        test_features.append(test_feats)
        
    id = np.concatenate(ids)[:, None]
    lda_feats = np.concatenate(features, axis=0)
    lda_feats = np.concatenate([id, lda_feats], axis=1)
    lda_feats = lda_feats[np.argsort(lda_feats[:, 0])].astype(np.float32)
    
    test_features = reduce(operator.add, test_features) / len(test_features)
    test_id = test['test_id'].values[:, None]
    test_features = np.concatenate([test_id, test_features], axis=1).astype(np.float32)
    np.save('artifacts/training/lda_features.npy', lda_feats)
    np.save('artifacts/prediction/lda_features.npy', test_features)


