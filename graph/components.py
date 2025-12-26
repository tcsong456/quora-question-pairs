import numpy as np
import networkx as nx

def build_graph(train_hash):
    G = nx.Graph()
    if type(train_hash) == list:
        for dhash in train_hash:
            for _, h1, h2 in dhash:
                G.add_edge(h1, h2)
    else:
        for _, h1, h2 in train_hash:
            G.add_edge(h1, h2)
    return G

def component_features(G):
    deg = dict(G.degree())
    clu = nx.clustering(G)
    
    comp_size = {}
    comp_density = {}
    comp_mean_d = {}
    comp_max_d = {}
    comp_avg_clu = {}
    
    for nodes in nx.connected_components(G):
        nodes = list(nodes)
        n = len(nodes)
        
        m = G.subgraph(nodes).number_of_edges()
        denom = n * (n - 1) / 2
        dens = 2 * m / denom if denom > 0 else 0
        mean_d = 2 * m / n if n > 0 else 0
        
        max_d = 0
        clu_sum = 0
        for node in nodes:
            dg = deg.get(node, 0)
            if dg > max_d:
                max_d = dg
            clu_sum += clu.get(node, 0)
        clu_avg = clu_sum / n if n > 0 else 0
    
        for u in nodes:
            comp_size[u] = n
            comp_density[u] = dens
            comp_mean_d[u] = mean_d
            comp_max_d[u] = max_d
            comp_avg_clu[u] = clu_avg
    
    return comp_size, comp_density, comp_mean_d, comp_max_d, comp_avg_clu

def pair_stats_from_node_metric(train_hash, metric_dict, default=0.0):
    feats = []
    eps = 1e-12

    for _, u, v in train_hash:
        a = float(metric_dict.get(u, default))
        b = float(metric_dict.get(v, default))

        mn = min(a, b)
        mx = max(a, b)
        mean = 0.5 * (a + b)
        diff = mx - mn
        ratio = mx / max(mn, eps)

        feats.append([mn, mx, mean, diff, ratio])

    return np.asarray(feats, dtype=np.float32)

if __name__ == '__main__':
    base_path = 'artifacts/'
    train_hash = np.load(base_path+'train_hash.npy')
    test_hash = np.load(base_path+'test_hash.npy')
    
    G = build_graph([train_hash[:, :-1], test_hash])
    comp_size, comp_density, comp_mean_deg, comp_max_deg, comp_avg_clust = component_features(G)

    X_size   = pair_stats_from_node_metric(train_hash[:, :-1], comp_size, default=1.0)
    X_dens   = pair_stats_from_node_metric(train_hash[:, :-1], comp_density, default=0.0)
    X_mdeg   = pair_stats_from_node_metric(train_hash[:, :-1], comp_mean_deg, default=0.0)
    X_xdeg   = pair_stats_from_node_metric(train_hash[:, :-1], comp_max_deg, default=0.0)
    X_cclust = pair_stats_from_node_metric(train_hash[:, :-1], comp_avg_clust, default=0.0)
    id = train_hash[:, [0]].astype(np.float32)
    X_component_train = np.concatenate([id, X_size, X_dens, X_mdeg, X_xdeg, X_cclust], axis=1)
    
    X_size   = pair_stats_from_node_metric(test_hash, comp_size, default=1.0)
    X_dens   = pair_stats_from_node_metric(test_hash, comp_density, default=0.0)
    X_mdeg   = pair_stats_from_node_metric(test_hash, comp_mean_deg, default=0.0)
    X_xdeg   = pair_stats_from_node_metric(test_hash, comp_max_deg, default=0.0)
    X_cclust = pair_stats_from_node_metric(test_hash, comp_avg_clust, default=0.0)
    id = test_hash[:, [0]].astype(np.float32)
    X_component_test = np.concatenate([id, X_size, X_dens, X_mdeg, X_xdeg, X_cclust], axis=1)
    np.save('artifacts/training/components.npy', X_component_train)
    np.save('artifacts/prediction/components.npy', X_component_test)


