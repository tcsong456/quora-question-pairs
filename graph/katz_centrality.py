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

def largest_cc_subgraph(G):
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()

def compute_katz_centrality(G, alpha=None, beta=1.0, max_iter=2000, tol=1e-6):
    if alpha is None:
        dmax = max(dict(G.degree()).values()) if G.number_of_nodes() else 1
        alpha = 0.9 / max(1, dmax)
    
    return nx.katz_centrality(G, alpha=alpha, beta=beta, max_iter=max_iter, tol=tol)

def centrality_pair_features(hash_data, central_dict, eps=1e-9):
    feats = []
    for _, u, v in hash_data:
        cu = central_dict.get(u, 0)
        cv = central_dict.get(v, 0)
        
        cmin = min(cu, cv)
        cmax = max(cu, cv)
        cmean = 0.5 * (cu + cv)
        cdiff = cmax - cmin
        cratio = cmax / max(cmin, eps)

        feats.append([cmin, cmax, cmean, cdiff, cratio])
    feats = np.array(feats, dtype=np.float32)
    return feats

if __name__ == '__main__':
    base_path = 'artifacts/'
    train_hash = np.load(base_path+'train_hash.npy')
    test_hash = np.load(base_path+'test_hash.npy')
    
    G = build_graph([train_hash[:, :-1], test_hash])
    H = largest_cc_subgraph(G)
    katz = compute_katz_centrality(H, max_iter=200)
    
    train_katz = centrality_pair_features(train_hash[:, :-1], katz)
    test_katz = centrality_pair_features(test_hash, katz)
    
    id = train_hash[:, [0]].astype(np.float32)
    test_id = test_hash[:, [0]].astype(np.float32)
    train_feats = np.concatenate([id, train_katz], axis=1)
    test_feats = np.concatenate([test_id, test_katz], axis=1)
    np.save('artifacts/training/katz.npy', train_feats)
    np.save('artifacts/prediction/katz.npy', test_feats)
    
#%%
