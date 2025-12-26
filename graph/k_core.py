import numpy as np
import networkx as nx

def build_graph(hash_data):
    G = nx.Graph()
    if type(hash_data) == list:
        for dhash in hash_data:
            for _, h1, h2 in dhash:
                if h1 != h2:
                    G.add_edge(h1, h2)
    else:
        for _, h1, h2 in hash_data:
            if h1 != h2:
                G.add_edge(h1, h2)
    return G

def compute_kcore(G):
    core = nx.core_number(G)
    return core

def kcore_features(data, core_dict):
    kcores = []
    for _, h1, h2 in data:
        kcore1 = core_dict.get(h1, 0)
        kcore2 = core_dict.get(h2, 0)
        kcore_min = min(kcore1, kcore2)
        kcore_max = max(kcore1, kcore2)
        kcore_diff = kcore_max - kcore_min
        kcore_mean = (kcore_min + kcore_max) / 2
        kcores.append([kcore_min, kcore_max, kcore_mean, kcore_diff])
    kcores = np.vstack(kcores)
    id = data[:, [0]].astype(np.float32)
    kcore_stats = np.concatenate([id, kcores], axis=1).astype(np.float32)
    return kcore_stats

if __name__ == '__main__':
    base_path = 'artifacts/'
    train_hash = np.load(base_path+'train_hash.npy')
    test_hash = np.load(base_path+'test_hash.npy')
    
    G = build_graph([train_hash[:, :-1], test_hash])
    kcore_dict = compute_kcore(G)
    train_kcore = kcore_features(train_hash[:, :-1], kcore_dict)
    trest_kcore = kcore_features(test_hash, kcore_dict)
    np.save('artifacts/training/kcore.npy', train_kcore)
    np.save('artifacts/prediction/kcore.npy', trest_kcore)