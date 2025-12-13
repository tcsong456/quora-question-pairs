import re
import numpy as np
from tqdm import tqdm
from rapidfuzz import fuzz
from models.utils.build_vocab import BuildVocab

def jaccard(q1, q2):
    return 1 - len(set(q1) & set(q2)) / (len(set(q1) | set(q2)) + 1e-9)

def get_dice(q1, q2):
    q1 = set(q1)
    q2 = set(q2)
    intersect = len(q1.intersection(q2))
    union = len(q1) + len(q2)
    return 2 * intersect / (union + 1e-9)

def tokenize(text):
    _TOKEN_RE = re.compile(r"[a-z0-9]+")
    return _TOKEN_RE.findall(str(text).lower())

def fuzz_features(df):
    q1 = df["question1"].fillna("").astype(str).to_numpy()
    q2 = df["question2"].fillna("").astype(str).to_numpy()
    df["fuzz_WRatio"] = np.fromiter((fuzz.WRatio(a,b) for a,b in zip(q1,q2)), dtype=np.float32)
    df["fuzz_token_set_ratio"] = np.fromiter((fuzz.token_set_ratio(a,b) for a,b in zip(q1,q2)), dtype=np.float32)
    df["fuzz_partial_ratio"] = np.fromiter((fuzz.partial_ratio(a,b) for a,b in zip(q1,q2)), dtype=np.float32)
    fuzz_feats = df[fuzz_cols].values
    return fuzz_feats

if __name__ == '__main__':
    bv = BuildVocab(
            'data/train.csv',
            'data/test.csv'
        )
    train = bv.train_data
    test = bv.test_data
    
    fuzz_cols = ['fuzz_WRatio', 'fuzz_token_set_ratio', 'fuzz_partial_ratio']
    train_feats, test_feats = [], []
    row_train_iterator = tqdm(train.iterrows(), total=train.shape[0], desc='building training similarity features between two questions')
    for _ , row in row_train_iterator:
        q1, q2 = row['question1'], row['question2']
        z1, z2 = tokenize(q1), tokenize(q2)
        jad = jaccard(z1, z2)
        dice = get_dice(z1, z2)
        train_feats.append([jad, dice])
    fuzz_feats_tr = fuzz_features(train)
    train_feats = np.vstack(train_feats)
    id = train['id'].values[:, None]
    train_feats = np.concatenate([id, train_feats, fuzz_feats_tr], axis=1).astype(np.float32)
    
    row_test_iterator = tqdm(test.iterrows(), total=test.shape[0], desc='building testing similarity features between two questions')
    for _ , row in row_test_iterator:
        q1, q2 = row['question1'], row['question2']
        z1, z2 = tokenize(q1), tokenize(q2)
        jad = jaccard(z1, z2)
        dice = get_dice(z1, z2)
        test_feats.append([jad, dice])
    fuzz_feats_te = fuzz_features(test)
    test_feats = np.vstack(test_feats)
    id = test['test_id'].values[:, None]
    test_feats = np.concatenate([id, test_feats, fuzz_feats_te], axis=1).astype(np.float32)
    np.save('artifacts/training/basic_feats.npy', train_feats)
    np.save('artifacts/prediction/basic_feats.npy', test_feats)

#%%


