import re
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from models.utils.build_vocab import BuildVocab
from sklearn.feature_extraction.text import TfidfVectorizer

space_re = re.compile('\s+')

def tokenize(text):
    _TOKEN_RE = re.compile(r"[a-z0-9]+")
    return _TOKEN_RE.findall(text)

def l2_norm(x):
    return np.sqrt(x.multiply(x).sum(axis=1)).A1

if __name__ == '__main__':
    bv = BuildVocab(
            'data/train.csv',
            'data/test.csv'
        )
    train = bv.train_data
    test = bv.test_data
    
    common_words, diff_words = [], []
    for q1, q2 in train[['question1', 'question2']].values:
        q1_tok = tokenize(space_re.sub(' ', q1.lower().strip()))
        q2_tok = tokenize(space_re.sub(' ', q2.lower().strip()))
        cw = ' '.join(set(q1_tok) & set(q2_tok))
        dw = ' '.join((set(q1_tok) - set(q2_tok)) | (set(q2_tok) - set(q1_tok)))
        common_words.append(cw); diff_words.append(dw)

    for q1, q2 in test[['question1', 'question2']].values:
        q1_tok = tokenize(space_re.sub(' ', q1.lower().strip()))
        q2_tok = tokenize(space_re.sub(' ', q2.lower().strip()))
        cw = ' '.join(set(q1_tok) & set(q2_tok))
        dw = ' '.join((set(q1_tok) - set(q2_tok)) | (set(q2_tok) - set(q1_tok)))
        common_words.append(cw); diff_words.append(dw)

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
    max_features=80000,
    norm=None
        )
    
    x_common = pipe_common.fit_transform(common_words)
    x_diff = tfidf_diff.fit_transform(diff_words)
    
    N = train.shape[0]
    x_common_tr, x_common_te = x_common[: N], x_common[N:]
    x_diff_tr, x_diff_te = x_diff[: N], x_diff[N:]
    l2_tr = l2_norm(x_diff_tr)
    l2_te = l2_norm(x_diff_te)
    l1_tr = x_diff_tr.sum(axis=1).A1
    l1_te = x_diff_te.sum(axis=1).A1
    
    x_tr = np.concatenate([train['id'].values[:, None], x_common_tr, l1_tr[:,None], l2_tr[:, None]], axis=1)
    x_te = np.concatenate([test['test_id'].values[:, None], x_common_te, l1_te[:,None], l2_te[:, None]], axis=1)
    np.save('artifacts/training/lsa_features.npy', x_tr.astype(np.float32))
    np.save('artifacts/prediction/lsa_features.npy', x_te.astype(np.float32))







