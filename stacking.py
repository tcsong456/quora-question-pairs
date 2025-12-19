import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

def build_data(mode='train'):
    assert mode in ['train', 'test']
    location = 'training' if mode == 'train' else 'prediction'
    dfs = []
    ft_name = ['bimpm_features_multi_head', 'diin_features', 'esim_features', 'sbert_features', 'deberta_features',
               'lda_features', 'single_pair_tfidf', 'double_pair_tfidf', 'graph_local', 'basic_feats', 'neighbor_avg_degree',
               'kcore', 'katz', 'triangle_clustring', 'components', '2_hop_neigh', 'n2v']
    for f in ft_name:
        model = f.split('_')[0]
        f = f'artifacts/{location}/{f}.npy'
        d = np.load(f)
        columns = ['id', f'{model}_prob'] + [f'{model}_feature_{i}' for i in range(d.shape[1]-2)]
        d = pd.DataFrame(d, columns=columns).set_index('id')
        dfs.append(d)
        
    if mode == 'train':
        train = pd.read_csv('data/train.csv')
        x_meta = pd.concat(dfs, axis=1)
        x_meta = x_meta.merge(train[['id', 'is_duplicate']], how='left', on=['id']).drop('id', axis=1)
        x_meta, y = x_meta.iloc[:, :-1].values, x_meta.iloc[:, -1].values
        return x_meta, y
    else:
        x_meta = pd.concat(dfs, axis=1).reset_index()
        test_id = x_meta['id']
        x_meta = x_meta.drop('id', axis=1)
        return test_id, x_meta

params = {
    'num_leaves': 120,
    "objective": "binary",
    "metric": "binary_logloss",
    'min_data_in_leaf': 150,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'num_threads': 16
}

X_meta_tr, y = build_data('train')
test_id, X_meta_te = build_data('test')
val_losses = []
predictions = []
skf = StratifiedKFold(n_splits=5, random_state=7610, shuffle=True)
for train_idx, val_idx in skf.split(X_meta_tr, y):
    X_tr, y_tr = X_meta_tr[train_idx], y[train_idx]
    X_val, y_val = X_meta_tr[val_idx], y[val_idx]
    
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model_meta = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ],
    )
    
    p_val = model_meta.predict(X_val, num_iteration=model_meta.best_iteration)
    val_loss = log_loss(y_val, p_val)
    val_losses.append(val_loss)
    p_test = model_meta.predict(X_meta_te, num_iteration=model_meta.best_iteration)
    predictions.append(p_test)
val_loss = np.mean(val_losses)
print(f'average validation loss: {val_loss: .5f}')

test_id = pd.DataFrame(test_id).rename(columns={'id': 'test_id'}).astype(np.int32)
score = np.zeros([X_meta_te.shape[0]], dtype=np.float32)
for p in predictions:
    score += p
score /= len(predictions)
score = pd.DataFrame(score, columns=['is_duplicate'])
submission = pd.concat([test_id, score], axis=1)

sample = pd.read_csv('data/sample_submission.csv')
submission = sample[['test_id']].merge(submission, how='left', on=['test_id']).fillna(0)
submission.to_csv('artifacts/submission.csv', index=False)