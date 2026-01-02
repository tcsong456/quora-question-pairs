import re
import numpy as np
from sklearn.decomposition import NMF
from models.utils.build_vocab import BuildVocab
from sklearn.feature_extraction.text import TfidfVectorizer

space_re = re.compile('\s+')

def fit_nmf(text,
            topics=50,
            ngram_range=(1, 1),
            min_df=5,
            sublinear_tf=False,
            norm='l2',
            random_state=1056):
    vec = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=0.9,
            max_features=300000,
            sublinear_tf=sublinear_tf,
            norm=norm
        )
    X = vec.fit_transform(text)
    
    nmf = NMF(
            n_components=topics,
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            max_iter=100,
            verbose=1,
            alpha_W=0.0,
            l1_ratio=0.0,
            random_state=random_state
                    ) 
    W = nmf.fit_transform(X)
    W = np.asarray(W, dtype=np.float32)
    W_sum = W.sum(axis=1, keepdims=True) + 1e-12
    theta = W / W_sum
    return theta

def nmf_pair_features(t1, t2, eps=1e-12):
    dot = np.sum(t1 * t2, axis=1)
    n1 = np.sqrt(np.sum(t1 * t1, axis=1)) + eps
    n2 = np.sqrt(np.sum(t2 * t2, axis=1)) + eps
    cos = dot / (n1 * n2)

    diff = t1 - t2
    l1 = np.sum(np.abs(diff), axis=1)
    l2 = np.sqrt(np.sum(diff * diff, axis=1) + eps)

    overlap = np.sum(np.minimum(t1, t2), axis=1)

    p1 = np.clip(t1, eps, 1.0)
    p2 = np.clip(t2, eps, 1.0)
    ent1 = -np.sum(p1 * np.log(p1), axis=1)
    ent2 = -np.sum(p2 * np.log(p2), axis=1)
    ent_absdiff = np.abs(ent1 - ent2)

    feats = [cos, l1, l2, overlap, ent1, ent2, ent_absdiff]

    m = 0.5 * (p1 + p2)
    kl1 = np.sum(p1 * np.log(p1 / m), axis=1)
    kl2 = np.sum(p2 * np.log(p2 / m), axis=1)
    js = 0.5 * (kl1 + kl2)
    feats.append(js)

    X = np.vstack(feats).T.astype(np.float32)
    return X

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
    
    nmf_vectors = fit_nmf(all_q,
                          topics=80,
                          ngram_range=(1, 2),
                          min_df=3,
                          sublinear_tf=True)
    
    N = train.shape[0]
    M = test.shape[0]
    q1_nmf_tr = nmf_vectors[: N]
    q2_nmf_tr = nmf_vectors[N: 2*N]
    q1_nmf_te = nmf_vectors[2*N: 2*N+M]
    q2_nmf_te = nmf_vectors[2*N+M:]
    
    x_tr = nmf_pair_features(q1_nmf_tr, q2_nmf_tr)
    x_te = nmf_pair_features(q1_nmf_te, q2_nmf_te)
    x_tr = np.concatenate([train['id'].values[:, None], x_tr], axis=1)
    x_te = np.concatenate([test['test_id'].values[:, None], x_te], axis=1)
    np.save('artifacts/training/nmf_features.npy', x_tr.astype(np.float32))
    np.save('artifacts/prediction/nmf_features.npy', x_te.astype(np.float32))




