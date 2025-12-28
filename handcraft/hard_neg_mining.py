import re
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from models.utils.build_vocab import BuildVocab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.neighbors import NearestNeighbors

_ws = re.compile(r"\s+")
def normalize_text(s):
    s = str(s).strip().lower()
    s = _ws.sub(" ", s)
    return s

class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def build_component_from_qids(
        train_qids1,
        train_qids2,
        train_target
        ):
    max_id = max(train_qids1.max(), train_qids2.max()) + 1
    uf = UnionFind(max_id)
    pos_mask = train_target == 1
    train_qid1_pos = train_qids1[pos_mask]
    train_qid2_pos = train_qids2[pos_mask]
    for q1, q2 in zip(train_qid1_pos, train_qid2_pos):
        uf.union(q1, q2)
    
    comp = {}
    all_qids = np.unique(train_qids1.tolist() + train_qids2.tolist())
    for qid in all_qids:
        comp[qid] = uf.find(qid)
    return comp

class SBERTEncoder:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts, batch_size=512, max_length=40):
        texts = [str(t) for t in texts]
        out_chunks = []
        total = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(texts), batch_size), total=total, desc="SBERT encode", leave=False):
            chunk = texts[i:i+batch_size]
            batch = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                hs = self.model(**batch, return_dict=True).last_hidden_state

            mask = batch["attention_mask"].unsqueeze(-1).type_as(hs)
            sent = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
            sent = F.normalize(sent, p=2, dim=-1)
            out_chunks.append(sent.detach().cpu().numpy())

        return np.concatenate(out_chunks, axis=0)

def build_tfidf_nn(cand_texts, n_neighbors=80):
    cand_norm = [normalize_text(str(t)) for t in cand_texts]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, dtype=np.float32)
    X_cand = vec.fit_transform(cand_norm)
    X_cand = sk_normalize(X_cand, norm="l2", axis=1, copy=False)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=1)
    nn.fit(X_cand)
    return vec, nn

def collect_shortlist_ids_for_anchors(
    anchors, vec, nn, n_neighbors=80, batch_size=4096
):
    all_neigh = []
    union = set()

    total = (len(anchors) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(anchors), batch_size), total=total, desc="TF-IDF knn (batched)"):
        chunk = [normalize_text(str(t)) for t in anchors[i:i+batch_size]]
        Q = vec.transform(chunk)
        Q = sk_normalize(Q, norm="l2", axis=1, copy=False)
        _, idx = nn.kneighbors(Q, n_neighbors=n_neighbors)

        all_neigh.append(idx)
        for row in idx:
            union.update(row.tolist())

    neigh_ids = np.vstack(all_neigh)
    union_ids = np.fromiter(union, dtype=np.int64)
    return neigh_ids, union_ids

def build_cand_meta(cand_texts, qid_dict, comp_dict):
    Nc = len(cand_texts)
    cand_comp = np.empty(Nc, dtype=np.int64)
    cand_qid = np.empty(Nc, dtype=np.int64)

    for i, t in enumerate(cand_texts):
        tt = str(t).strip()
        qid = qid_dict.get(tt, -1)
        cand_qid[i] = qid
        cand_comp[i] = comp_dict.get(qid, qid) if qid != -1 else -1

    return cand_qid, cand_comp

def mine_hard_negatives_ultrafast(
    pos_pairs,
    cand_texts,
    qid_dict,
    comp_dict,
    sbert_encoder,
    tfidf_neighbors=120, 
    shortlist=60, 
    final_k=3,
    tfidf_batch=4096,
    sbert_bs=1024,
    max_length=40,
    store_float16=True
):
    vec, nn = build_tfidf_nn(
        cand_texts, n_neighbors=tfidf_neighbors
    )

    anchors = []
    seen = set()
    for a, b in pos_pairs:
        aa = str(a).strip()
        bb = str(b).strip()
        if aa not in seen:
            seen.add(aa); anchors.append(aa)
        if bb not in seen:
            seen.add(bb); anchors.append(bb)

    neigh_ids, union_ids = collect_shortlist_ids_for_anchors(
        anchors, vec, nn, n_neighbors=tfidf_neighbors, batch_size=tfidf_batch
    )
    print(f"Unique anchors: {len(anchors)}")
    print(f"Union candidate ids to SBERT-encode: {len(union_ids)} out of {len(cand_texts)}")

    E_anchor = sbert_encoder.encode(anchors, batch_size=sbert_bs, max_length=max_length)
    E_cand_union = sbert_encoder.encode([cand_texts[i] for i in union_ids],
                                        batch_size=sbert_bs, max_length=max_length)

    E_anchor = E_anchor.astype(np.float32)  # queries small; keep fp32
    if store_float16:
        E_cand_union = E_cand_union.astype(np.float16)

    cand_to_row = -np.ones(len(cand_texts), dtype=np.int64)
    cand_to_row[union_ids] = np.arange(len(union_ids), dtype=np.int64)

    cand_qids, cand_comps = build_cand_meta(cand_texts, qid_dict, comp_dict)

    anchor_index = {t: i for i, t in enumerate(anchors)}

    def rerank_for_anchor(anchor_text, other_text, neigh_list):
        anchor_text = str(anchor_text).strip()
        other_text = str(other_text).strip()

        ai = anchor_index.get(anchor_text)
        if ai is None:
            return []

        u = E_anchor[ai]

        forbid = set()
        qid_a = qid_dict.get(anchor_text, None)
        qid_o = qid_dict.get(other_text, None)
        if qid_a is not None:
            forbid.add(comp_dict.get(qid_a, qid_a))
        if qid_o is not None:
            forbid.add(comp_dict.get(qid_o, qid_o))

        kept_ids = []
        kept_txt = []
        for j in neigh_list:
            ct = str(cand_texts[j]).strip()
            if not ct:
                continue
            if ct == anchor_text or ct == other_text:
                continue
            compj = cand_comps[j]
            if compj != -1 and compj in forbid:
                continue
            kept_ids.append(j)
            kept_txt.append(ct)
            if len(kept_ids) >= shortlist:
                break

        if not kept_ids:
            return []

        rows = cand_to_row[np.array(kept_ids)]
        m = rows != -1
        if not np.any(m):
            return []
        rows = rows[m]
        kept_txt = [kept_txt[i] for i in np.where(m)[0]]

        V = E_cand_union[rows]
        sims = V.astype(np.float32) @ u.astype(np.float32)
        order = np.argsort(-sims)[:final_k]
        return [kept_txt[i] for i in order]

    q1_negs, q2_negs = [], []
    for q1, q2 in tqdm(pos_pairs, total=len(pos_pairs), desc="mining (SBERT rerank only)"):
        n1 = neigh_ids[anchor_index[str(q1).strip()]]
        n2 = neigh_ids[anchor_index[str(q2).strip()]]

        q1_negs.append(rerank_for_anchor(q1, q2, n1))
        q2_negs.append(rerank_for_anchor(q2, q1, n2))

    return q1_negs, q2_negs

if __name__ == '__main__':
    bv = BuildVocab(
            'data/train.csv',
            'data/test.csv'
        )
    train = bv.train_data
    
    qid = 0
    question_ids = {}
    all_questions = np.unique(train['question1'].tolist() + train['question2'].tolist())
    for i, q in enumerate(all_questions):
        q = q.strip()
        if q not in question_ids:
            question_ids[q] = qid
            qid += 1
        all_questions[i] = q
    
    train_qid1 = train['question1'].map(lambda r: question_ids[r.strip()]).values
    train_qid2 = train['question2'].map(lambda r: question_ids[r.strip()]).values
    train_y = train['is_duplicate'].values
    
    comp_map = build_component_from_qids(train_qid1, train_qid2, train_y)
    pos_pairs = train[train['is_duplicate']==1][['question1', 'question2']].values
    
    sbert = SBERTEncoder("sentence-transformers/all-mpnet-base-v2")
    
    q1_neg, q2_neg = mine_hard_negatives_ultrafast(
            pos_pairs,
            all_questions,
            question_ids,
            comp_map,
            sbert,
            tfidf_batch=128,
            final_k=8
        )

    with open('artifacts/q1_neg_sample.pkl', 'wb') as f:
        pickle.dump(q1_neg, f)

    with open('artifacts/q2_neg_sample.pkl', 'wb') as f:
        pickle.dump(q2_neg, f)

