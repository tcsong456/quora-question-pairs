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
    sw -= set(qwords)
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

def get_sentence_embedding(text, idf_dict, stopwords,
                           glove, fasttext):
    sent_embs = []
    mode = 'testing' if len(text) > 450000 else 'training'
    for t in tqdm(text, total=len(text), desc=f'building sentence embedding for {mode}'):
        total_idf = 0
        final_emb = np.zeros([300], dtype=np.float32)
        q = [tok for tok in t if tok not in stopwords]
        for word in q:
            if word in glove:
                emb = glove[word]
            else:
                emb = fasttext.wv.get_vector(word)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            idf = idf_dict.get(word, 0)
            emb = emb * idf
            total_idf += idf
            final_emb += emb
        sent_emb = final_emb / (total_idf + 1e-9)
        sent_embs.append(sent_emb)
    sent_embs = np.stack(sent_embs, axis=0)
    return sent_embs

def pairwise_feature(q1_emb, q2_emb):
    cos = (q1_emb * q2_emb).sum(axis=1)
    diff = q1_emb - q2_emb
    l1 = np.sum(diff, axis=1)
    l2 = np.sqrt(np.sum(diff**2, axis=1))
    abs_diff = abs(diff)
    abs_diff_mean = abs_diff.mean(axis=1)
    abs_diff_max = abs_diff.max(axis=1)
    abs_diff_std = abs_diff.std(axis=1)
    prod = q1_emb * q2_emb
    prod_mean = prod.mean(axis=1)
    prod_max = prod.max(axis=1)
    prod_std = prod.std(axis=1)
    features = [
            cos,
            l1,
            l2,
            abs_diff_mean,
            abs_diff_max,
            abs_diff_std,
            prod_mean,
            prod_max,
            prod_std
        ]
    features = np.stack(features, axis=1)
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
    q1_sent = get_sentence_embedding(tr_q1, idf_dict, stopwords,
                                      glove, fasttext)
    q2_sent = get_sentence_embedding(tr_q2, idf_dict, stopwords,
                                      glove, fasttext)
    features_train = pairwise_feature(q1_sent, q2_sent)
    train_features = np.concatenate([train['id'].values[:, None], features_train], axis=1)
    
    te_q1 = test['question1'].tolist()
    te_q1 = [normalize_text(q).split() for q in te_q1]
    te_q2 = test['question2'].tolist()
    te_q2 = [normalize_text(q).split() for q in te_q2]
    q1_sent = get_sentence_embedding(te_q1, idf_dict, stopwords,
                                     glove, fasttext)
    q2_sent = get_sentence_embedding(te_q2, idf_dict, stopwords,
                                     glove, fasttext)
    features_test = pairwise_feature(q1_sent, q2_sent)
    test_features = np.concatenate([test['test_id'].values[:, None], features_test], axis=1)
    
    np.save('artifacts/training/emb_pairwise_features.npy', train_features.astype(np.float32))
    np.save('artifacts/prediction/emb_pairwise_features.npy', test_features.astype(np.float32))
