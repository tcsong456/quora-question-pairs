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

def neighbor_deg_graph(hash_data, deg_dict):
    nei_stats = []
    for _, h1, h2 in hash_data:
        h1_nei_deg = deg_dict.get(h1, 0)
        h2_nei_deg = deg_dict.get(h2, 0)
        nei_max = max(h1_nei_deg, h2_nei_deg)
        nei_min = min(h1_nei_deg, h2_nei_deg)
        nei_diff = nei_max - nei_min
        nei_mean = (nei_min + nei_max) / 2
        nei_stats.append([nei_min, nei_max, nei_mean, nei_diff])
    nei_stats = np.vstack(nei_stats)
    id = hash_data[:, [0]].astype(np.float32)
    nei_stats = np.concatenate([id, nei_stats], axis=1).astype(np.float32)
    return nei_stats

if __name__ == '__main__':
    base_path = 'artifacts/'
    train_hash = np.load(base_path+'train_hash.npy')
    test_hash = np.load(base_path+'test_hash.npy')
    
    G = build_graph([train_hash[:, :-1], test_hash])
    avg_degrees = nx.average_neighbor_degree(G)
    train_nei_stats = neighbor_deg_graph(train_hash[:, :-1], avg_degrees)
    trest_nei_stats = neighbor_deg_graph(test_hash, avg_degrees)
    np.save('artifacts/training/neighbor_avg_degree.npy', train_nei_stats)
    np.save('artifacts/prediction/neighbor_avg_degree.npy', trest_nei_stats)

