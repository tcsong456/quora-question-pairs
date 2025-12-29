import warnings
warnings.filterwarnings('ignore')
import pickle
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from models.utils.build_vocab import BuildVocab
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold

# class InfoNceDataset(Dataset):
#     def __init__(self,
#                  data):
#         self.data = data
#         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
#         self.max_len = 40
        
#         with open('artifacts/q1_neg_sample.pkl', 'rb') as f:
#             self.q1_neg_dict = pickle.load(f)
        
#         with open('artifacts/q2_neg_sample.pkl', 'rb') as f:
#             self.q2_neg_dict = pickle.load(f)
    
#     def __len__(self):
#         return self.data.shape[0]
    
#     def __getitem__(self, index):
#         row = self.data.iloc[index]
#         q1 = row['question1'].strip()
#         q2 = row['question2'].strip()
#         q1_hard_neg = self.q1_neg_dict[q1]
#         q2_hard_neg = self.q2_neg_dict[q2]
        
#         enc1 = self.tokenizer(
#                 q1,
#                 padding="max_length",
#                 truncation=True,
#                 max_length=self.max_len,
#                 return_tensors='pt'
#             )
#         enc2 = self.tokenizer(
#                 q2,
#                 padding="max_length",
#                 truncation=True,
#                 max_length=self.max_len,
#                 return_tensors='pt'
#             )
#         enc_neg1 = self.tokenizer(
#                 q1_hard_neg,
#                 padding="max_length",
#                 truncation=True,
#                 max_length=self.max_len,
#                 return_tensors='pt'
#             )
#         enc_neg2 = self.tokenizer(
#                 q2_hard_neg,
#                 padding="max_length",
#                 truncation=True,
#                 max_length=self.max_len,
#                 return_tensors='pt'
#             )

#         return {
#             "q1_ids": enc1["input_ids"].squeeze(0),          
#             "q1_mask": enc1["attention_mask"].squeeze(0),    
#             "q2_ids": enc2["input_ids"].squeeze(0),          
#             "q2_mask": enc2["attention_mask"].squeeze(0),    
#             "q1n_ids": enc_neg1["input_ids"],                
#             "q1n_mask": enc_neg1["attention_mask"],          
#             "q2n_ids": enc_neg2["input_ids"],                
#             "q2n_mask": enc_neg2["attention_mask"],          
#         }

class InfoNceDataset(Dataset):
    def __init__(self, df):
        self.q1 = df["question1"].astype(str).tolist()
        self.q2 = df["question2"].astype(str).tolist()
        with open('artifacts/q1_neg_sample.pkl', 'rb') as f:
            self.q1_neg_dict = pickle.load(f)
        
        with open('artifacts/q2_neg_sample.pkl', 'rb') as f:
            self.q2_neg_dict = pickle.load(f)
    
    def __len__(self):
        return len(self.q1)
    
    def __getitem__(self, i):
        q1 = self.q1[i].strip()
        q2 = self.q2[i].strip()

        q1n = self.q1_neg_dict[q1]
        q2n = self.q2_neg_dict[q2]
        return q1, q2, q1n, q2n

class InfoNceCollator:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.max_len = 40
        self.K = 3

    def __call__(self, batch):
        q1, q2, q1n, q2n = zip(*batch)  # tuples of strings
        q1 = [t.strip() for t in q1]
        q2 = [t.strip() for t in q2]
        B, K = len(q1), self.K
        
        def fix(negs):
            return [str(t).strip() for t in negs]
        
        q1n_list = [fix(neg) for neg in q1n]
        q2n_list = [fix(neg) for neg in q2n]
        q1n_flat = [t for negs in q1n_list for t in negs]
        q2n_flat = [t for negs in q2n_list for t in negs]
        
        enc1  = self.tok(q1,       padding="max_length", truncation=True,
                         max_length=self.max_len, return_tensors="pt")
        enc2  = self.tok(q2,       padding="max_length", truncation=True,
                         max_length=self.max_len, return_tensors="pt")
        encn1 = self.tok(q1n_flat, padding="max_length", truncation=True,
                         max_length=self.max_len, return_tensors="pt")
        encn2 = self.tok(q2n_flat, padding="max_length", truncation=True,
                         max_length=self.max_len, return_tensors="pt")

        q1n_ids  = encn1["input_ids"].view(B, K, -1)
        q1n_mask = encn1["attention_mask"].view(B, K, -1)
        q2n_ids  = encn2["input_ids"].view(B, K, -1)
        q2n_mask = encn2["attention_mask"].view(B, K, -1)

        return {
            "q1_ids": enc1["input_ids"], "q1_mask": enc1["attention_mask"],
            "q2_ids": enc2["input_ids"], "q2_mask": enc2["attention_mask"],
            "q1n_ids": q1n_ids, "q1n_mask": q1n_mask,
            "q2n_ids": q2n_ids, "q2n_mask": q2n_mask,
        }

def infoNCE_with_hard_negatives(
            q1, q2,
            q1_neg, q2_neg,
            temperature=0.05
        ):
    B, H = q1.shape
    K = q2_neg.shape[1]
    labels = torch.arange(B, device=q1.device)
    
    u = torch.cat([q1, q1_neg.reshape(B*K, H)], dim=0)
    logit_v2u = q2 @ u.T / temperature
    loss_v2u = F.cross_entropy(logit_v2u, labels)
    
    v = torch.cat([q2, q2_neg.reshape(B*K, H)], dim=0)
    logit_u2v = q1 @ v.T / temperature
    loss_u2v = F.cross_entropy(logit_u2v, labels)
    
    return 0.5 * (loss_v2u + loss_u2v)

class SBERTEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
    
    def mean_pool(self, last_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_state)
        summed = (last_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom
    
    def forward(self, input_ids, att_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=att_mask)
        emb = self.mean_pool(out.last_hidden_state, att_mask)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb

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
        
class Trainer:
    def __init__(self,
                 bv):
        train = bv.train_data
        test = bv.test_data
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data_train, self.data_pos_val, self.data_val = [], [], []
        self.models, self.optimizers = [], []
        y = train['is_duplicate']
        x = train.drop('is_duplicate', axis=1)
        skf = StratifiedKFold(n_splits=5, random_state=7610, shuffle=True)
        for train_idx, val_idx in skf.split(x, y):
            x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
            x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]
            
            y_pos_tr = y_train == 1
            x_train = x_train[y_pos_tr]
            y_pos_val = y_val == 1
            x_val_pos = x_val[y_pos_val]
            
            self.data_train.append(x_train)
            self.data_pos_val.append(x_val_pos)
            self.data_val.append(x_val)
            
            model = SBERTEncoder("sentence-transformers/all-mpnet-base-v2").to(device)
            self.models.append(model)
            
            encoder_lr = 2e-5
            no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layer_norm.weight", "layer_norm.bias"]
            encoder_decay, encoder_nodecay = [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                use_nodecay = any(nd in name for nd in no_decay)
                if use_nodecay:
                    encoder_nodecay.append(param)
                else:
                    encoder_decay.append(param)

            weight_decay = 0.01
            optimizer = optim.AdamW(
                  [
                    {"params": encoder_decay,   "lr": encoder_lr, "weight_decay": weight_decay},
                    {"params": encoder_nodecay, "lr": encoder_lr, "weight_decay": 0.0},
                ]
                )
            self.optimizers.append(optimizer)
            self.models.append(model)
            self.dataset = InfoNceDataset
        self.device = device
    
    def train(self):
        for fold in range(5):
            model = self.models[fold]
            optimizer = self.optimizers[fold]
            ds_train = self.dataset(self.data_train[fold])
            ds_val = self.dataset(self.data_pos_val[fold])
            collate = InfoNceCollator()
            dl_train = DataLoader(
                    ds_train,
                    batch_size=64,
                    shuffle=True,
                    collate_fn=collate
                )
            dl_val = DataLoader(
                    ds_val,
                    batch_size=128,
                    shuffle=False,
                    collate_fn=collate
                )
            loss_meter_tr = AverageMeter()
            loss_meter_val = AverageMeter()
            scaler = GradScaler(enabled=True)
            
            for epoch in range(20):
                model.train()
                train_dl = tqdm(dl_train, total=len(dl_train),
                                desc='training sbert on infonce loss')
                for batch in train_dl:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)
                
                    optimizer.zero_grad()
                    with autocast(enabled=True):
                        B, K, L = batch["q2n_ids"].shape
                        q1 = model(batch['q1_ids'], batch['q1_mask'])
                        q2 = model(batch['q2_ids'], batch['q2_mask'])
                        q1n = model(
                            batch["q1n_ids"].view(B*K, L),
                            batch["q1n_mask"].view(B*K, L),
                        ).view(B, K, -1)                
                        q2n = model(
                            batch["q2n_ids"].view(B*K, L),
                            batch["q2n_mask"].view(B*K, L),
                        ).view(B, K, -1)
                        loss = infoNCE_with_hard_negatives(q1, q2, q1n, q2n, temperature=0.08)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    loss_meter_tr.update(loss.item(), 64)
                    loss = loss_meter_tr.average
                    train_dl.set_postfix({
                        f'epoch {epoch} loss': f'{loss:.5f}'
                        } 
                      )
                
                with torch.no_grad():
                    model.eval()
                    val_dl = tqdm(dl_val, total=len(dl_val),
                                  desc='evaluating sbert on infonce loss')
                    for batch in val_dl:
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                batch[k] = v.to(self.device)
                    
                        with autocast(enabled=True):
                            B, K, L = batch["q2n_ids"].shape
                            q1 = model(batch['q1_ids'], batch['q1_mask'])
                            q2 = model(batch['q2_ids'], batch['q2_mask'])
                            q1n = model(
                                batch["q1n_ids"].view(B*K, L),
                                batch["q1n_mask"].view(B*K, L),
                            ).view(B, K, -1)                
                            q2n = model(
                                batch["q2n_ids"].view(B*K, L),
                                batch["q2n_mask"].view(B*K, L),
                            ).view(B, K, -1)
                            val_loss = infoNCE_with_hard_negatives(q1, q2, q1n, q2n, temperature=0.08)
                        
                        loss_meter_val.update(val_loss.item(), 128)
                        val_loss = loss_meter_val.average
                        val_dl.set_postfix({
                            f'epoch {epoch} loss': f'{val_loss: .5f}'
                          })

if __name__ == '__main__':
    bv = BuildVocab(
              'data/train.csv',
              'data/test.csv'
          )
    trainer = Trainer(bv)
    trainer.train()

#%%
# import numpy as np
# from torch.utils.data import DataLoader


# train = bv.train_data
# pos_data = train[train['is_duplicate']==1]
# collate = InfoNceCollator()
# ds = InfoNceDataset(
#         pos_data
#     )

# dl = DataLoader(ds, batch_size=128, shuffle=True)
# model = SBERTEncoder("sentence-transformers/all-mpnet-base-v2").to('cuda:0')
# for batch in dl:
#     for k, v in batch.items():
#         if isinstance(v, torch.Tensor):
#             batch[k] = v.to('cuda:0')
#     B, K, L = batch["q2n_ids"].shape
#     q1 = model(batch['q1_ids'], batch['q1_mask'])
#     q2 = model(batch['q2_ids'], batch['q2_mask'])
#     q1n = model(
#         batch["q1n_ids"].view(B*K, L),
#         batch["q1n_mask"].view(B*K, L),
#     ).view(B, K, -1)  # (B,K,H)
    
#     q2n = model(
#         batch["q2n_ids"].view(B*K, L),
#         batch["q2n_mask"].view(B*K, L),
#     ).view(B, K, -1)
    
#     loss = infoNCE_with_hard_negatives(q1, q2, q1n, q2n, temperature=0.05)
#     break

#%%
# trainer = Trainer(bv=bv)
# trainer.data_val