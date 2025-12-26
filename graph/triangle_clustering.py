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

def triangle_clustring_features(hash_data, tri, clu):
    feats = []
    for _, u, v in hash_data:
        tu = tri.get(u, 0)
        tv = tri.get(v, 0)
        cu = clu.get(u, 0)
        cv = clu.get(v, 0)
        
        feats.append([
                min(tu, tv),
                max(tu, tv),
                abs(tu - tv),
                (tu + tv) / 2,
                np.log1p(tu),
                np.log1p(tv),
                min(cu, cv),
                max(cu, cv),
                abs(cu - cv),
                0.5 * (cv + cu)
                ]
            )
    feats = np.array(feats, dtype=np.float32)
    id = hash_data[:, [0]].astype(np.float32)
    feats = np.concatenate([id, feats], axis=1)
    return feats

if __name__ == '__main__':
    base_path = 'artifacts/'
    train_hash = np.load(base_path+'train_hash.npy')
    test_hash = np.load(base_path+'test_hash.npy')
    
    G = build_graph([train_hash[:, :-1], test_hash])
    tri = nx.triangles(G)
    clu = nx.clustering(G)
    train_feats = triangle_clustring_features(train_hash[:, :-1], tri, clu)
    test_feats = triangle_clustring_features(test_hash, tri, clu)
    np.save('artifacts/training/triangle_clustring.npy', train_feats)
    np.save('artifacts/prediction/triangle_clustring.npy', test_feats)