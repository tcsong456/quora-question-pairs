import re
import os
import math
import numpy as np
from tqdm import tqdm
from collections import Counter
from models.utils.build_vocab import BuildVocab
from gensim.models.fasttext import load_facebook_model
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def build_stopwords_for_btm():
    sw = set(ENGLISH_STOP_WORDS)
    neg = {"no", "not", "nor", "never", "n't"}
    qwords = {"how", "what", "why", "when", "where", "which", "who"}
    sw -= set(neg)
    sw |= set(qwords)
    sw |= {"would", "could", "should", "may", "might", "can", "also"}
    return sw

def normalize_text(text):
    PUNCT_REGEX = re.compile(r"[^\w\s.'-]+")
    text = text.lower().strip()
    text = PUNCT_REGEX.sub(" ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)  # remove sentence periods only
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_idf(
        text,
        min_df,
        max_df_ratio
    ):
    df = Counter()
    N = 0
    for doc in text:
        df.update(set(doc))
        N += 1
    
    out = {}
    max_df = int(max_df_ratio * N)
    for t, c in df.items():
         if c >= min_df and c <= max_df:
             out[t] = math.log((1.0 + N) / (1.0 + c))
    return out

def build_alignment(
        q1s, q2s,
        idf_dict, glove, fasttext,
        topk, cover_thrs, stop_words
    ):
    eps = 1e-9
    features = []
    mode = 'testing' if len(q1s) > 500000 else 'training'
    iter = tqdm(q1s, total=len(q1s), desc=f'building alignment features for {mode}')
    for q1, q2 in zip(iter, q2s):
        feats = []
        q1_vecs, q2_vecs = [], []
        q1_idf, q2_idf = [], []
        q1 = set([t for t in q1 if t not in stop_words])
        q2 = set([t for t in q2 if t not in stop_words])
        
        for word in q1:
            if word in glove:
                emb = glove[word]
            else:
                emb = fasttext.wv.get_vector(word)
            U = emb / (np.linalg.norm(emb) + 1e-9)
            q1idf = idf_dict.get(word, 0)
            q1_vecs.append(U)
            q1_idf.append(q1idf)
            
        for word in q2:
            if word in glove:
                emb = glove[word]
            else:
                emb = fasttext.wv.get_vector(word)
            V = emb / (np.linalg.norm(emb) + 1e-9)
            q2idf = idf_dict.get(word, 0)
            q2_vecs.append(V)
            q2_idf.append(q2idf)
        
        if len(q1_vecs) == 0 or len(q2_vecs) == 0:
            feats = [0] * 10
            features.append(feats)
            continue
        
        q1_vecs = np.stack(q1_vecs, axis=0)
        q2_vecs = np.stack(q2_vecs, axis=0)
        S = q1_vecs @ q2_vecs.T
        
        row_max = S.max(axis=1)
        col_max = S.max(axis=0)
        
        row_mean_idf = np.sum(row_max * q1_idf) / (np.sum(q1_idf) + eps)
        col_mean_idf = np.sum(col_max * q2_idf) / (np.sum(q2_idf) + eps)
        
        row_topk = min(topk, len(row_max))
        idx = np.argpartition(row_max, -row_topk)[-row_topk:]
        topk_row = row_max[idx]
        topk_idf_row = np.array(q1_idf)[idx]
        topk_row_mean = np.sum(topk_row * topk_idf_row) / (np.sum(topk_idf_row) + eps)
        
        col_topk = min(topk, len(col_max))
        idx = np.argpartition(col_max, -col_topk)[-col_topk:]
        topk_col = col_max[idx]
        topk_idf_col = np.array(col_max)[idx]
        topk_col_mean = np.sum(topk_col * topk_idf_col) / (np.sum(topk_idf_col) + eps)
        feats += [row_mean_idf, col_mean_idf, topk_row_mean, topk_col_mean, row_max.max(), col_max.max()]
        
        for cover in cover_thrs:
            row_cover_rate = row_max > cover
            row_cover_rate = (q1_idf * row_cover_rate).sum() / (np.sum(q1_idf) + eps)
            col_cover_rate = col_max > cover
            col_cover_rate = (q2_idf * col_cover_rate).sum() / (np.sum(q2_idf) + eps)
            
            feats += [row_cover_rate, col_cover_rate]
        features.append(feats)
    features = np.stack(features, axis=0).shape
    return features

if __name__ == '__main__':
    bv = BuildVocab(
            'data/train.csv',
            'data/test.csv'
        )
    train = bv.train_data
    test = bv.test_data
    
    stopwords = build_stopwords_for_btm()
    all_q = train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()
    all_q = [normalize_text(q).split() for q in all_q]
    idf_dict = build_idf(all_q, min_df=2, max_df_ratio=0.9)
    
    glove = {}
    path = 'artifacts/glove.840B.300d.txt'
    file_size = os.path.getsize(path)
    with open(path, 'r', encoding='utf8') as f, \
        tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading GloVe") as pbar:
            for line in f:
                parts = line.rstrip().split(' ')
                word = parts[0]
                vec = np.array(parts[1:], dtype=np.float32)
                glove[word] = vec
                pbar.update(len(line.encode('utf8')))
    fasttext = load_facebook_model('artifacts/cc.en.300.bin')
    
    tr_q1 = train['question1'].tolist()
    tr_q1 = [normalize_text(q).split() for q in tr_q1]
    tr_q2 = train['question2'].tolist()
    tr_q2 = [normalize_text(q).split() for q in tr_q2]
    train_features = build_alignment(tr_q1, tr_q2,
        idf_dict, glove, fasttext,
        topk=3, cover_thrs=[0.65, 0.8], stop_words=stopwords)
    train_features = np.concatenate([train['id'].values[:, None], train_features], axis=1)
    
    te_q1 = test['question1'].tolist()
    te_q1 = [normalize_text(q).split() for q in te_q1]
    te_q2 = test['question2'].tolist()
    te_q2 = [normalize_text(q).split() for q in te_q2]
    test_features = build_alignment(te_q1, te_q2,
        idf_dict, glove, fasttext,
        topk=3, cover_thrs=[0.65, 0.8], stop_words=stopwords)
    test_features = np.concatenate([test['test_id'].values[:, None], test_features], axis=1)
    
    np.save('artifact/training/alignment_features.npy', train_features.astype(np.float32))
    np.save('artifact/prediction/alignment_features.npy', test_features.astype(np.float32))
