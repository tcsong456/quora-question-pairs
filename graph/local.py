import math
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

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

def precompute_graph_cahe(G):
    deg = dict(G.degree())
    comp_id = {}
    comp_size = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for node in comp:
            comp_id[node] = i
            comp_size[node] = len(comp)
    
    pagerank = nx.pagerank(G, alpha=0.85)
    outputs = {
            'deg': deg,
            'comp_id': comp_id,
            'comp_size': comp_size,
            'pagerank': pagerank
        }
    return outputs

def safe_divide(a, b, eps=1e-9):
    return a / (b + eps)

def bounded_spl(G, u, v, cutoff=1):
    if u == v:
        return 0
    if (u not in G) or (v not in G):
        return -1
    try:
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=cutoff)
        return lengths.get(v, cutoff+1)
    except nx.NetworkXNoPath:
        return -1
    
def add_graph_features(df, G, cache, sp_cutoff=1):
    deg = cache['deg']
    # comp_id = cache['comp_id']
    comp_size = cache['comp_size']
    pr = cache['pagerank']
    
    neighbours = {n:set(G.neighbors(n)) for n in G.nodes()}
    rows = []
    row_iter = tqdm(df, total=df.shape[0], desc='building graph local features')
    for _, u, v in row_iter:
        du = deg.get(u, 0)
        dv = deg.get(v, 0)
        deg_min = min(du, dv)
        deg_max = max(du, dv)
        deg_diff = abs(du - dv)
        deg_ratio = safe_divide(du, dv)
        
        nu = neighbours.get(u, set())
        nv = neighbours.get(v, set())
        cn = len(nu & nv)
        union = len(nu | nv)
        jacc = safe_divide(cn, union)
        
        aa, ra = 0.0, 0.0
        for z in (nu & nv):
            dz = deg.get(z, 0)
            if dz > 1:
                aa += 1.0 / math.log(dz)
            if dz > 0:
                ra += 1.0 / dz
        pa = float(du) * float(dv)
        
        csize_u = comp_size.get(u, 0)
        csize_v = comp_size.get(v, 0)
        csize_min = min(csize_u, csize_v)
        csize_max = max(csize_u, csize_v)
        spl = bounded_spl(G, u, v, cutoff=sp_cutoff)
        
        pr_u = pr.get(u, 0)
        pr_v = pr.get(v, 0)
        pr_min = min(pr_u, pr_v)
        pr_max = max(pr_u, pr_v)
        pr_diff = abs(pr_u - pr_v)
        pr_ratio = safe_divide(pr_u, pr_v)
        
        rows.append({
            "deg_u": du,
            "deg_v": dv,
            "deg_min": deg_min,
            "deg_max": deg_max,
            "deg_diff": deg_diff,
            "deg_ratio": deg_ratio,

            "cn": cn,
            "jaccard": jacc,
            "adamic_adar": aa,
            "resource_allocation": ra,
            "preferential_attachment": pa,

            "comp_size_u": csize_u,
            "comp_size_v": csize_v,
            "comp_size_min": csize_min,
            "comp_size_max": csize_max,

            f"shortest_path_le_{sp_cutoff}": spl,
            "pagerank_min": pr_min,
            "pagerank_max": pr_max,
            "pagerank_diff": pr_diff,
            'pagerank_ratio': pr_ratio
        })
        
    feats = pd.DataFrame(rows).values
    id = df[:, [0]].astype(np.float32)
    feats = np.concatenate([id, feats], axis=1)
    return feats
        
if __name__ == '__main__':
    base_path = 'artifacts/'
    train_hash = np.load(base_path+'train_hash.npy')
    test_hash = np.load(base_path+'test_hash.npy')
    
    G = build_graph([train_hash[:, :-1], test_hash])
    cache = precompute_graph_cahe(G)
    train_feats = add_graph_features(train_hash[:, :-1], G, cache, sp_cutoff=4)
    test_feats = add_graph_features(test_hash, G, cache, sp_cutoff=4)
    np.save('artifacts/training/graph_local.npy', train_feats)
    np.save('artifacts/prediction/graph_local.npy', test_feats)
