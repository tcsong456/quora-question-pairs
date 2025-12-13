import re
import warnings
warnings.filterwarnings(action='ignore')
import operator
import numpy as np
from functools import reduce
from models.utils.build_vocab import BuildVocab
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

def tokenize(text):
    _TOKEN_RE = re.compile(r"[a-z0-9]+")
    return _TOKEN_RE.findall(str(text).lower()) 

def common_words(row):
    q1, q2 = row['question1'], row['question2']
    s1 = set(tokenize(q1))
    s2 = set(tokenize(q2))
    return " ".join(s1 & s2)

def diff_words(row):
    q1, q2 = row['question1'], row['question2']
    s1 = set(tokenize(q1))
    s2 = set(tokenize(q2))
    diff = ' '.join((set(s1 - s2)) | (set(s2 - s1)))
    return diff

if __name__ == '__main__':
    bv = BuildVocab(
            'data/train.csv',
            'data/test.csv'
        )
    train = bv.train_data
    test = bv.test_data
    
    tfidf_common = TfidfVectorizer(
    tokenizer=tokenize, token_pattern=None, lowercase=False,
    ngram_range=(1,1),
    min_df=3, max_df=0.95,
    sublinear_tf=True,
    max_features=120000
    )
    svd_common = TruncatedSVD(n_components=64, random_state=951)
    pipe_common = make_pipeline(
        tfidf_common,
        svd_common
        )
    
    tfidf_diff = TfidfVectorizer(
    tokenizer=tokenize, token_pattern=None, lowercase=False,
    ngram_range=(1,1),
    min_df=3, max_df=0.98,
    sublinear_tf=True,
    max_features=80000
)
    svd_diff = TruncatedSVD(n_components=32, random_state=951)
    pipe_diff = make_pipeline(
        tfidf_diff,
        svd_diff
    )
    
    test['common_words'] = test.apply(common_words, axis=1)
    test['diff_words'] = test.apply(diff_words, axis=1)
    common_features, diff_features, ids = [], [], []
    common_test_features, diff_test_features = [], []
    y = train['is_duplicate']
    x = train.drop('is_duplicate', axis=1)
    skf = StratifiedKFold(n_splits=5, random_state=7610, shuffle=True)
    for train_idx, val_idx in skf.split(x, y):
        x_tr = x.iloc[train_idx]
        x_val = x.iloc[val_idx]
        
        x_tr['common_words'] = x_tr.apply(common_words, axis=1)
        x_tr['diff_words'] = x_tr.apply(diff_words, axis=1)
        pipe_common.fit(x_tr['common_words'].values.tolist())
        pipe_diff.fit(x_tr['diff_words'].values.tolist())
        
        x_val['common_words'] = x_val.apply(common_words, axis=1)
        x_val['diff_words'] = x_val.apply(diff_words, axis=1)
        x_common = pipe_common.transform(x_val['common_words'].values.tolist())
        x_diff = pipe_diff.transform(x_val['diff_words'].values.tolist())
        common_features.append(x_common)
        diff_features.append(np.sqrt((x_diff * x_diff).sum(axis=1)))
        ids.append(x_val['id'])
        
        x_common_te = pipe_common.transform(test['common_words'].values.tolist())
        x_diff_te = pipe_diff.transform(test['diff_words'].values.tolist())
        common_test_features.append(x_common_te)
        diff_test_features.append(np.sqrt((x_diff_te * x_diff_te).sum(axis=1)))
    
    common_features = np.concatenate(common_features)
    diff_features = np.concatenate(diff_features)[:, None]
    ids = np.concatenate(ids)[:, None]
    train_features = np.concatenate([ids, common_features, diff_features], axis=1).astype(np.float32)
    train_features = train_features[np.argsort(train_features[:, 0])].astype(np.float32)
    
    common_test_features = reduce(operator.add, common_test_features) / len(common_test_features)
    diff_test_features = reduce(operator.add, diff_test_features) / len(diff_test_features)
    test_id = test['test_id'].values[:, None]
    test_features = np.concatenate([test_id, common_test_features, diff_test_features[:, None]], axis=1).astype(np.float32)
    np.save('artifacts/training/single_pair_tfidf.npy', train_features)
    np.save('artifacts/prediction/single_pair_tfidf.npy', test_features)

