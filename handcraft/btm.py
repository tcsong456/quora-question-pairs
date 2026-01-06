import re
import numpy as np
from tqdm import tqdm
from numba import njit
from collections import Counter
from models.utils.build_vocab import BuildVocab
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def build_stopwords_for_btm():
    sw = set(ENGLISH_STOP_WORDS)
    neg = {"no", "not", "nor", "never", "n't"}
    qwords = {"how", "what", "why", "when", "where", "which", "who"}
    sw -= set(neg)
    sw |= set(qwords)
    sw |= {"would", "could", "should", "may", "might", "can", "also"}
    return sw

def tokenize_for_btm(
        text,
        stop_words,
        min_token_len
        ):
    if text is None:
        return []
    
    _NUM_RE = re.compile(r"\b\d+(?:[\.,]\d+)*\b")
    _TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
    s = text.lower()
    s = _NUM_RE.sub(" <num> ", s)
    toks = _TOKEN_RE.findall(s)

    out = []
    for t in toks:
        if t in stop_words:
            continue
        if len(t) < min_token_len:
            continue
        out.append(t)
    return out

def filter_vocab_by_df(
            tokenized_docs,
            min_df,
            max_df_ratio
        ):
    N = len(tokenized_docs)
    df = Counter()
    for doc in tokenized_docs:
        df.update(set(doc))
    
    keep = set()
    max_df = int(N * max_df_ratio)
    for t, c in df.items():
        if c >= min_df and c <= max_df:
            keep.add(t)
    
    filtered_docs = []
    for doc in tokenized_docs:
        filtered_docs.append([t for t in doc if t in keep])
    
    vocab = {t: i for i, t in enumerate(sorted(keep))}
    return filtered_docs, vocab

def docs_to_ids(docs, vocab):
    return [[vocab[t] for t in doc if t in vocab] for doc in docs]

def build_biterm(docs):
    window = 5
    w1s, w2s = [], []
    iter = tqdm(docs, total=len(docs), desc='building word co-occurance pairs')
    for doc in iter:
        L = len(doc)
        if L < 2:
            continue
        for i in range(L):
            r = min(L, i+window+1)
            for j in range(1, r):
                w1 = doc[i]
                w2 = doc[j]
                if w1 == w2:
                    continue
                if w1 < w2:
                    w1s.append(w1); w2s.append(w2)
                else:
                    w1s.append(w2); w2s.append(w1)
    return np.array(w1s, dtype=np.int32), np.array(w2s, dtype=np.int32)

@njit
def _sample_categorical_unnorm(rng_u, probs):
    s = 0.0
    for i in range(probs.shape[0]):
        s += probs[i]
    if s <= 0.0:
        return int(rng_u % probs.shape[0])
    u = rng_u * s
    c = 0.0
    for i in range(probs.shape[0]):
        c += probs[i]
        if c >= u:
            return i
    return probs.shape[0] - 1

@njit
def btm_gibbs_numba(w1, w2, K, V, alpha, beta, iters, seed):
    B = w1.shape[0]
    nz = np.zeros(K, dtype=np.int64)
    nzw = np.zeros((K, V), dtype=np.int64)
    z = np.empty(B, dtype=np.int32)
    state = np.uint64(seed)

    def rand_u01():
        nonlocal state
        state = state * np.uint64(6364136223846793005) + np.uint64(1)
        return ((state >> np.uint64(11)) & np.uint64((1 << 53) - 1)) / float(1 << 53)

    def rand_int(n):
        return int(rand_u01() * n)

    for i in range(B):
        zi = rand_int(K)
        z[i] = zi
        a = w1[i]
        b = w2[i]
        nz[zi] += 1
        nzw[zi, a] += 1
        nzw[zi, b] += 1

    betaV = beta * V
    p = np.empty(K, dtype=np.float64)

    for it in range(iters):
        for i in range(B):
            zi = z[i]
            a = w1[i]
            b = w2[i]
    
            nz[zi] -= 1
            nzw[zi, a] -= 1
            nzw[zi, b] -= 1
    
            for k in range(K):
                denom = 2.0 * nz[k] + betaV
                pa = (nzw[k, a] + beta) / denom
                pb = (nzw[k, b] + beta) / denom
                p[k] = (nz[k] + alpha) * pa * pb
    
            newz = _sample_categorical_unnorm(rand_u01(), p)
    
            z[i] = newz
            nz[newz] += 1
            nzw[newz, a] += 1
            nzw[newz, b] += 1

    return z, nz, nzw

class BTM:
    def __init__(self, K, V, alpha=1.0, beta=0.01, seed=0):
        self.K = int(K)
        self.V = int(V)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.seed = int(seed)
        self.z = None
        self.nz = None
        self.nzw = None

    def fit(self, w1, w2, iters=30):
        w1 = np.asarray(w1, dtype=np.int32)
        w2 = np.asarray(w2, dtype=np.int32)

        K = int(self.K)
        V = int(self.V)
        iters = int(iters)
        alpha = float(self.alpha)
        beta = float(self.beta)
        seed = int(self.seed)

        z, nz, nzw = btm_gibbs_numba(w1, w2, K, V, alpha, beta, iters, seed)

        self.z, self.nz, self.nzw = z, nz, nzw
        return self
    
    def biterm_topic_posterior(self, a: int, b: int) -> np.ndarray:
        if self.nz is None or self.nzw is None:
            raise ValueError("Model is not fit yet. Call fit() first.")
    
        a = int(a); b = int(b)
        if a > b:
            a, b = b, a
    
        nz = self.nz.astype(np.float64, copy=False)
        nzw_a = self.nzw[:, a].astype(np.float64, copy=False)
        nzw_b = self.nzw[:, b].astype(np.float64, copy=False)
    
        beta = float(self.beta)
        alpha = float(self.alpha)
        betaV = beta * float(self.V)
    
        denom = 2.0 * nz + betaV
        p = (nz + alpha) * ((nzw_a + beta) / denom) * ((nzw_b + beta) / denom)
    
        s = p.sum()
        if s <= 0 or not np.isfinite(s):
            return np.full(self.K, 1.0 / self.K, dtype=np.float64)
        return p / s

def infer_theta_docs(model, docs_ids, window=0):
    N = len(docs_ids)
    K = model.K
    theta = np.zeros((N, K), dtype=np.float32)
        
    iter = tqdm(enumerate(docs_ids), total=len(docs_ids), desc='inferring theta for docs')
    for di, doc in iter:
        L = len(doc)
        if L < 2:
            continue

        biterms = []
        if window and window > 0:
            for i in range(L):
                jmax = min(L, i + window + 1)
                for j in range(i + 1, jmax):
                    a, b = doc[i], doc[j]
                    if a == b: 
                        continue
                    if a > b: a, b = b, a
                    biterms.append((a, b))
        else:
            for i in range(L - 1):
                a = doc[i]
                for j in range(i + 1, L):
                    b = doc[j]
                    if a == b: 
                        continue
                    if a > b: a, b = b, a
                    biterms.append((a, b))

        if not biterms:
            continue

        acc = np.zeros(K, dtype=np.float64)
        for a, b in biterms:
            acc += model.biterm_topic_posterior(a, b)

        acc /= acc.sum()
        theta[di] = acc.astype(np.float32)

    return theta

def build_btm_pair_features(t1, t2, eps=1e-12):
    dot = np.sum(t1 * t2, axis=1)
    n1 = np.sqrt(np.sum(t1 * t1, axis=1)) + eps
    n2 = np.sqrt(np.sum(t2 * t2, axis=1)) + eps
    cos = dot / (n1 * n2)

    p = np.clip(t1, eps, 1.0)
    p = p / np.sum(p, axis=1, keepdims=True)
    q = np.clip(t2, eps, 1.0)
    q = q / np.sum(q, axis=1, keepdims=True)
    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m), axis=1)
    kl_qm = np.sum(q * np.log(q / m), axis=1)
    js = 0.5 * (kl_pm + kl_qm)

    top1_1 = np.argmax(t1, axis=1)
    top1_2 = np.argmax(t2, axis=1)
    top1_eq = (top1_1 == top1_2).astype(np.float32)

    top1_p1 = np.max(t1, axis=1)
    top1_p2 = np.max(t2, axis=1)

    X = np.vstack([cos, js, top1_eq, top1_p1, top1_p2]).T.astype(np.float32)
    return X

if __name__ == '__main__':
    bv = BuildVocab(
            'data/train.csv',
            'data/test.csv'
        )
    train = bv.train_data
    test = bv.test_data
    
    docs = []
    all_questions = train['question1'].tolist() + \
        train['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()
    stop_words = build_stopwords_for_btm()
    iter_questions = tqdm(all_questions, total=len(all_questions), desc='tokenizing docs')
    for q in iter_questions:
        docs.append(tokenize_for_btm(q, stop_words, 2))
    
    docs,vocab = filter_vocab_by_df(
            docs,
            min_df=5,
            max_df_ratio=0.3
        )
    doc_ids = docs_to_ids(docs, vocab)
    w1, w2 = build_biterm(doc_ids)
    V = len(vocab)
    K = 50

    btm = BTM(K=K, V=V, alpha=1.0, beta=0.01, seed=42).fit(w1, w2, iters=10)
    theta_all = infer_theta_docs(btm, doc_ids, window=5)
    
    N = train.shape[0]
    M = test.shape[0]
    q1_theta_tr = theta_all[: N]
    q2_theta_tr = theta_all[N: 2*N]
    q1_theta_te = theta_all[2*N: 2*N+M]
    q2_theta_te = theta_all[2*N+M:]
    
    x_tr = build_btm_pair_features(q1_theta_tr, q2_theta_tr)
    x_te = build_btm_pair_features(q1_theta_te, q2_theta_te)
    x_tr = np.concatenate([train['id'].values[:, None], x_tr], axis=1)
    x_te = np.concatenate([test['test_id'].values[:, None], x_te], axis=1)
    np.save('artifacts/training/btm_features.npy', x_tr.astype(np.float32))
    np.save('artifacts/prediction/btm_features.npy', x_te.astype(np.float32))





