import pandas as pd
from models.bimpm import BiMPM
from utils.build_vocab import BuildVocab
from utils.dataset import QQPDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from gensim.models.fasttext import load_facebook_model

class Trainer:
    def __init__(self,
                 vocab,
                 vec_model,
                 train_batch_size=128,
                 test_batch_size=512):
        train = vocab.train_data
        test = vocab.test_data
        self.train = train
        self.test = test
        self.words_index_dict = vocab.load()
    
    def train(self, kfold=3):
        y = self.train['is_duplicate']
        x = self.train.drop('is_duplicate', axis=1)
        skf = StratifiedKFold(n_splits=kfold, random_state=7610, shuffle=True)
        for i, (train_idx, val_idx) in enumerate(skf.split(x, y)):
            x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            x_train.reset_index.drop('index', axis=1, inplace=True)
            x_val.reset_index.drop('index', axis=1, inplace=True)
            y_train.reset_index.drop('index', axis=1, inplace=True)
            y_val.reset_index.drop('index', axis=1, inplace=True)
            
            train_dataset = QQPDataset(x_train)

#%%
# target = train['is_duplicate']
# train.drop('is_duplicate', axis=1, inplace=True)
skf = StratifiedKFold(n_splits=5, random_state=7610, shuffle=True)
for i, (train_idx, val_idx) in enumerate(skf.split(train, target)):
    x_train, x_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
    break
    
  #%%
y_val.reset_index().drop('index', axis=1)