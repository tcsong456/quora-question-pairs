import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

def build_weighted_graph(train, test, p_train, p_test):
    G = nx.Graph()
    
    def add_edges(df, p, eps=1e-6):
        q1 = df[:, 1]
        q2 = df[:, 2]
        p = np.clip(p, eps, 1-eps)
        for a, b, w in zip(q1, q2, p):
            if a == b:
                continue
            if G.has_edge(a, b):
                G[a][b]['weight'] = max(G[a][b]['weight'], w)
            else:
                G.add_edge(a, b, weight=float(w))
    add_edges(train, p_train)
    add_edges(test, p_test)
    return G

def fit_node2vec(G, dimensions=128, walk_length=40, num_walks=20,
                 p=1.0, q=2.0, window=10, min_count=1, workers=8, seed=42):
    n2v = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p, q=q,
        weight_key="weight",
        workers=workers,
        seed=seed,
        quiet=False
    )
    w2v = n2v.fit(window=window, min_count=min_count, batch_words=1024)
    return w2v

def get_vec(model, node, dim):
    key = str(node)
    if key in model.wv:
        return model.wv[key]
    return np.zeros(dim, dtype=np.float32)

def pair_features(u_vec, v_vec):
    dot = float(np.dot(u_vec, v_vec))
    nu = float(np.linalg.norm(u_vec) + 1e-12)
    nv = float(np.linalg.norm(v_vec) + 1e-12)
    cos = dot / (nu * nv)

    had = u_vec * v_vec
    diff = np.abs(u_vec - v_vec)
    add  = u_vec + v_vec

    feats = {
        "emb_cos": cos,
        "emb_dot": dot,
        "emb_l2": float(np.linalg.norm(u_vec - v_vec)),
        "emb_norm_u": nu,
        "emb_norm_v": nv,
        "emb_norm_min": min(nu, nv),
        "emb_norm_max": max(nu, nv),
        "emb_norm_ratio": nu / nv if nv > 0 else 0.0,

        "had_mean": float(had.mean()),
        "had_max":  float(had.max()),
        "had_min":  float(had.min()),
        "diff_mean": float(diff.mean()),
        "diff_max":  float(diff.max()),
        "add_mean":  float(add.mean()),
    }
    return feats

def make_embedding_pair_df(df, w2v, dim=128):
    from tqdm import tqdm
    rows = []
    mode = 'training' if df.shape[0] < 500000 else 'testing'
    for _, a, b in tqdm(df, total=len(df), desc=f'building {mode} n2v feats'):
        ua = get_vec(w2v, a, dim)
        vb = get_vec(w2v, b, dim)
        rows.append(pair_features(ua, vb))
    rows = np.concatenate([df[:, 0].astype(np.float32)[:, None], pd.DataFrame(rows).values], axis=1).astype(np.float32)
    return rows
        

if __name__ == '__main__':
    train_hash = np.load('artifacts/train_hash.npy')
    test_hash = np.load('artifacts/test_hash.npy')
    train_probs = np.load('artifacts/training/deberta_features.npy')[:, 1]
    test_probs = np.load('artifacts/prediction/deberta_features.npy')[:, 1]
    G = build_weighted_graph(train_hash, test_hash, train_probs, test_probs)
    keep = [n for n, d in G.degree() if d >= 2]
    H = G.subgraph(keep).copy()
    n2v = Node2Vec(H, dimensions=64, walk_length=20, num_walks=10, p=1.0, q=1.0,
                  workers=4, seed=42)
    w2v = n2v.fit(window=6, min_count=1, batch_words=1024)
    
    n2v_train = make_embedding_pair_df(train_hash[:, :-1], w2v, dim=64)
    n2v_test = make_embedding_pair_df(test_hash, w2v, dim=64)
    np.save('artifacts/training/n2v.npy', n2v_train)
    np.save('artifacts/prediction/n2v.npy', n2v_test)

