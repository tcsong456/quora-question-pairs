import numpy as np
import networkx as nx

def build_graph(hash_data):
    G = nx.Graph()
    if type(hash_data) == list:
        for dhash in hash_data:
            for _, h1, h2 in dhash:
                G.add_edge(h1, h2)
    else:
        for _, h1, h2 in hash_data:
            G.add_edge(h1, h2)
    return G

def precompute_neighbors(G):
    neighbors = {u: set(G.neighbors(u)) for u in G.nodes()}
    degrees = {u: len(neighbors[u]) for u in neighbors}
    return neighbors, degrees

def two_hop_set(u, neigh, deg, deg_cap=200, sample_k=80, n2_cap=5000):
    Nu = neigh.get(u, set())
    du = deg.get(u, 0)
    if du == 0:
        return set()
    if du > deg_cap:
        if sample_k is None or sample_k <= 0:
            return None
        Nu_list = list(Nu)
        Nu_list = Nu_list[:min(sample_k, len(Nu_list))]
    else:
        Nu_list = list(Nu)
    
    n2 = set()
    for w in Nu_list:
        n2.update(neigh.get(w, set()))
        if len(n2) > n2_cap:
            break
    
    n2.discard(u)
    n2.difference_update(Nu)
    return n2

def two_hop_overlap_features(hash_data, neigh, deg,
                             deg_cap=200, sample_k=80, n2_cap=5000):
    feats = []
    cache = {}
    for _, u, v in hash_data:
        if u not in cache:
            cache[u] = two_hop_set(u, neigh, deg, deg_cap, sample_k, n2_cap)
        if v not in cache:
            cache[v] = two_hop_set(v, neigh, deg, deg_cap, sample_k, n2_cap)
        
        u = cache[u]
        v = cache[v]
        if u is None or v is None:
            feats.append([0, 0, 0 ,0])
            continue
        
        inter = len(u & v)
        union = len(u | v)
        jacc = inter / union if union > 0 else 0
        
        common = u & v
        ra,aa = 0, 0
        for w in common:
            dw = deg.get(w, 0)
            if dw <= 1.0:
                continue
            ra += 1.0 / dw
            aa += 1.0 / np.log(dw)
        
        feats.append([inter, jacc, u, v, ra, aa, np.log1p(ra), np.log1p(aa)])
    feats = np.array(feats, dtype=np.float32)
    return feats

if __name__ == '__main__':
    base_path = 'artifacts/'
    train_hash = np.load(base_path+'train_hash.npy')
    test_hash = np.load(base_path+'test_hash.npy')
    
    G = build_graph([train_hash[:, :-1], test_hash])
    neigh, deg = precompute_neighbors(G)
    train_feats = two_hop_overlap_features(train_hash[:, :-1], neigh, deg)
    test_feats = two_hop_overlap_features(test_hash, neigh, deg)

#%%