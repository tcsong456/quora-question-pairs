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
    
    def forward(self, yp, yt):
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
        words_index_dict = vocab.load_dict()
        self.words_index_dict = words_index_dict
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.amp = amp
        self.epochs = epochs
        self.vocab = vocab
        
        self.data_train, self.data_val = [], []
        y = train['is_duplicate'].values
        x = train.drop('is_duplicate', axis=1)
        skf = StratifiedKFold(n_splits=5, random_state=7610, shuffle=True)
        for train_idx, val_idx in skf.split(x, y):            
            self.data_train.append(train_idx)
            self.data_val.append(val_idx)
        
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
                                    lr=0.002)
    
    def train_one_epoch(self, epoch, scaler):
        self.model.train()
        for i, train_idx in enumerate(self.data_train):
            loss_meter = AverageMeter()
            current_lr = self.optimizer.param_groups[0]['lr']
            train_dataset = QQPDataset(bv=self.vocab,
                                       q_idx=train_idx,
                                       mode='train')
            train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=256
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
                    loss = self.loss_fn(y_pred.view(-1), y_true.float())
                
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
    bv = BuildVocab('data/train.csv',
                    'data/test.csv')
    vec_model = load_facebook_model('artifacts/cc.en.300.bin')
    trainer = Trainer(
        vocab=bv,
        vec_model=vec_model,
        amp=True
      )
    trainer.fit()
    
  #%%


