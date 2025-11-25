import torch
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch import optim
from models.bimpm import BiMPM
from utils.build_vocab import BuildVocab
from utils.dataset import QQPDataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from gensim.models.fasttext import load_facebook_model

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.cnt = 0
        self.sum = 0
        self.average = 0
    
    def update(self, value, n):
        self.cnt += n
        self.sum += value * n
        self.average = self.sum / max(self.cnt, 1)

class CrossEntropyLoss(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
    
    def forward(self, yt, yp):
        if yt.ndim == 1:
            yt = yt[:, None]
        if yp.ndim == 1:
            yp = yp[:, None]
        
        ce_loss = -(yt * torch.log(yp + self.eps) + (1 - yt) * torch.log(1 - yp + self.eps)).mean()
        return ce_loss

class Trainer:
    def __init__(self,
                 vocab,
                 vec_model,
                 epochs=50,
                 amp=True):
        train = vocab.train_data
        test = vocab.test_data
        self.train = train
        self.test = test
        words_index_dict = vocab.load()
        self.words_index_dict = words_index_dict
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.amp = amp
        self.epochs = epochs
        
        self.data_train, self.data_val = [], []
        y = train['is_duplicate']
        x = train.drop('is_duplicate', axis=1)
        skf = StratifiedKFold(n_splits=5, random_state=7610, shuffle=True)
        for train_idx, val_idx in skf.split(x, y):
            x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            x_train = x_train.reset_index().drop('index', axis=1)
            x_val = x_val.reset_index().drop('index', axis=1)
            y_train = y_train.reset_index().drop('index', axis=1)
            y_val = y_val.reset_index().drop('index', axis=1)
            
            self.data_train.append((x_train, y_train))
            self.data_val.append((x_val, y_val))
        
        self.model = BiMPM(
            emb_dim=300,
            hidden_size=150,
            max_len=40,
            words_index_dict=words_index_dict,
            mp_dim=20,
            vec_model=vec_model,
            device=device,
            multi_attn_head=False
          ).to(device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=0.01)
    
    def train_one_epoch(self, epoch, scaler):
        self.model.train()
        for i, (x_tr, y_tr) in enumerate(self.data_train):
            loss_meter = AverageMeter()
            current_lr = self.optimizer.param_groups[0]['lr']
            train_dataset = QQPDataset(data=x_tr,
                                       words_index=self.words_index_dict,
                                       max_len=40,
                                       target=y_tr,
                                       mode='train')
            train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=128
              )
            train_dl = tqdm(train_dataloader,
                            total=len(train_dataloader),
                            desc=f'running fold {i} in training')
            
            for batch in train_dl:
                for i, v in enumerate(batch):
                    if isinstance(v, torch.Tensor):
                        batch[i] = v.to(self.device)
                
                self.optimizer.zero_grad()
                y_true = batch[-1]
                with autocast(enabled=self.amp):
                    y_pred = self.model(batch)
                    loss = self.loss_fn(y_pred, y_true.float())
                
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                loss_meter.update(loss.item(), 1)
                loss = loss_meter.average
                train_dl.set_postfix({
                    f'epoch {epoch} loss': f'{loss:.5f}',
                   'lr': f'{current_lr: .4f}'
                   } 
                  )
      
    def fit(self):
        scaler = GradScaler(enabled=self.amp)
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch, scaler)

if __name__ == '__main__':
    # bv = BuildVocab('data/train.csv',
    #                 'data/test.csv')
    # words_index = bv.load()
    # vec_model = load_facebook_model('artifacts/cc.en.300.bin')
    trainer = Trainer(
        vocab=bv,
        vec_model=vec_model,
        amp=True
      )
    trainer.fit()

#%%
train = pd.read_csv('data/train.csv')
target = train['is_duplicate']
train.drop('is_duplicate', axis=1, inplace=True)
skf = StratifiedKFold(n_splits=5, random_state=7610, shuffle=True)
for i, (train_idx, val_idx) in enumerate(skf.split(train, target)):
    x_train, x_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
    
    # x_train.reset_index().drop('index', axis=1, inplace=True)
    # x_val.reset_index().drop('index', axis=1, inplace=True)
    # y_train.reset_index().drop('index', axis=1, inplace=True)
    # y_val.reset_index().drop('index', axis=1, inplace=True)
    
    break
    
  #%%
# x_train = x_train.reset_index().drop('index', axis=1)
x_train

