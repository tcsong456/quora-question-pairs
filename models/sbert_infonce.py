import warnings
warnings.filterwarnings('ignore')
import os
import pickle
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from models.utils.build_vocab import BuildVocab
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold

class InfoNceDataset(Dataset):
    def __init__(self,
                 df):
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
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.max_len = 40
        self.K = 3

    def __call__(self, batch):
        q1, q2, q1n, q2n = zip(*batch)
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

class InfoNceDatasetV1(Dataset):
    def __init__(self,
                 data,
                 mode='train'):
        self.id = data['id'].values if mode != 'test' else data['test_id'].values
        self.q1 = data['question1'].astype(str).tolist()
        self.q2 = data['question2'].astype(str).tolist()
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, i):
        q1 = self.q1[i]
        q2 = self.q2[i]
        id = self.id[i]
        return id, q1, q2

class InfoNceCollatorV1:
    def __init__(self, tokenizer):
        self.max_len = 40
        self.tok = tokenizer
    
    def __call__(self, batch):
        id, q1, q2 = zip(*batch)
        q1 = [t.strip() for t in q1]
        q2 = [t.strip() for t in q2]
        
        enc1  = self.tok(q1,       padding="max_length", truncation=True,
                         max_length=self.max_len, return_tensors="pt")
        enc2  = self.tok(q2,       padding="max_length", truncation=True,
                         max_length=self.max_len, return_tensors="pt")
        id = torch.as_tensor(id)
        
        return {
            "id": id,
            "q1_ids": enc1["input_ids"], "q1_mask": enc1["attention_mask"],
            "q2_ids": enc2["input_ids"], "q2_mask": enc2["attention_mask"],
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

def pair_features(u, v, eps=1e-9):
    cos = np.sum(u * v, axis=1, keepdims=True)
    diff = u - v
    absdiff = abs(diff)
    prod = u * v
    
    l2 = np.sqrt(np.sum(diff**2, axis=1, keepdims=True) + eps)
    l1 = np.sum(absdiff, axis=1, keepdims=True)
    
    feats = np.concatenate([
        cos, l2, l1,
        absdiff.mean(1, keepdims=True),
        absdiff.max(1, keepdims=True),
        absdiff.std(1, keepdims=True),
        prod.mean(1, keepdims=True),
        prod.max(1, keepdims=True),
        prod.std(1, keepdims=True),
    ], axis=1)
    return feats

def info_retrieval_metrics(u, v, n, temperature=0.05):
    B, D = u.shape
    candidates = torch.cat([v.unsqueeze(1), n], dim=1)
    logits = torch.einsum('bd, bkd->bk', u, candidates) / temperature
    pos = logits[:, 0]
    neg_max = logits[:, 1:].max(dim=1).values
    margin = (pos - neg_max).mean()
    win_rate = (pos > neg_max).float().mean()
    pos_rank = (logits > pos[:, None]).sum(dim=1)
    
    metrics = {
        "margin": margin,
        "win_rate": win_rate
    }
    metrics["top1"] = (pos_rank < 1).float().mean()
    return metrics

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
        self.test = test
        
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
        self.device = device
    
    def train(self):
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.tokenizer = tokenizer
        for fold in range(1, 5):
            best_top1_hit_rate = 0
            best_loss = np.inf
            model = self.models[fold]
            optimizer = self.optimizers[fold]
            ds_train = InfoNceDataset(self.data_train[fold])
            ds_pos_val = InfoNceDataset(self.data_pos_val[fold])
            ds_val = InfoNceDatasetV1(self.data_val[fold])
            collate = InfoNceCollator(tokenizer=tokenizer)
            collatev1 = InfoNceCollatorV1(tokenizer=tokenizer)
            dl_train = DataLoader(
                    ds_train,
                    batch_size=32,
                    shuffle=True,
                    collate_fn=collate
                )
            dl_pos_val = DataLoader(
                    ds_pos_val,
                    batch_size=256,
                    shuffle=False,
                    collate_fn=collate
                )
            dl_val = DataLoader(
                    ds_val,
                    batch_size=256,
                    shuffle=False,
                    collate_fn=collatev1
                )
            scaler = GradScaler(enabled=True)
            
            for epoch in range(10):
                loss_meter_tr = AverageMeter()
                loss_meter_val = AverageMeter()
                top1_meter_val = AverageMeter()
                
                model.train()
                train_dl = tqdm(dl_train, total=len(dl_train),
                                desc=f'training sbert on infonce loss on fold{fold} epoch{epoch}')
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
                    
                    loss_meter_tr.update(loss.item(), 32)
                    loss = loss_meter_tr.average
                    train_dl.set_postfix({
                        f'epoch {epoch} loss': f'{loss:.5f}'
                        } 
                      )
                
                with torch.no_grad():
                    model.eval()
                    metrics = {}
                    val_pos_dl = tqdm(dl_pos_val, total=len(dl_pos_val),
                                  desc=f'evaluating sbert on infonce loss on fold{fold} epoch{epoch}')
                    for batch in val_pos_dl:
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
                            metrics_u2v = info_retrieval_metrics(q1, q2, q2n, temperature=0.08)
                            metrics_v2u = info_retrieval_metrics(q2, q1, q1n, temperature=0.08)
                            for (k1, v1), (k2, v2) in zip(metrics_u2v.items(), metrics_v2u.items()):
                                if k1 != k2:
                                    raise ValueError(f'metric1 has key: {k1} whicle metric2 has key: {k2}')
                                v = (v1 + v2) / 2
                                metrics[k1] = v
                                
                            top1_hit = metrics['top1']
                        
                        loss_meter_val.update(val_loss, B)
                        top1_meter_val.update(top1_hit, B)
                        val_loss = loss_meter_val.average
                        top1_hit = top1_meter_val.average
                        val_pos_dl.set_postfix({
                            f'epoch {epoch} loss': f'{val_loss: .5f}',
                            f'epoch {epoch} top1-hit': f'{top1_hit: .4f}',
                          })
                    
                feats, ids = [], []
                val_dl = tqdm(dl_val, total=len(dl_val),
                              desc=f'generating features for val fold {fold}')
                for batch in val_dl:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)
                    
                    with autocast(enabled=True):
                        id = batch['id'].cpu().numpy()
                        q1 = model(batch['q1_ids'], batch['q1_mask'])
                        q2 = model(batch['q2_ids'], batch['q2_mask'])
                        q1 = q1.detach().cpu().numpy()
                        q2 = q2.detach().cpu().numpy()
                        features = pair_features(q1, q2)
                        ids.append(id), feats.append(features)
                    
                if top1_hit > best_top1_hit_rate and val_loss < best_loss:
                    best_top1_hit_rate = top1_hit
                    best_loss = val_loss
                    bad_epoch = 0
                    torch.save({
                        'model': model.state_dict(),
                        'infonce_loss': val_loss,
                        'top1-hit': top1_hit
                      }, f'checkpoints/sbert_infonce_{fold}.pth')
                    ids = np.concatenate(ids)
                    pair_feats = np.concatenate(feats, axis=0)
                    infonce_pair_features = np.concatenate([ids[:, None], pair_feats], axis=1)
                    np.save(f'artifacts/training/infonce_pair_features_{fold}.npy', infonce_pair_features.astype(np.float32))
                else:
                    bad_epoch += 1
                
                if bad_epoch == 1:
                    print(f'early stopping reaches at epoch: {epoch}')
                    break
    
    @torch.no_grad()
    def predict(self):
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.tokenizer = tokenizer
        for fold in range(5):
            model = self.models[fold]
            model.eval()
            checkpoint = f'checkpoints/sbert_infonce_{fold}.pth'
            ckpt = torch.load(checkpoint)
            model.load_state_dict(ckpt['model'])
            collate_fn = InfoNceCollatorV1(self.tokenizer)
            
            ds_test = InfoNceDatasetV1(
                    self.test,
                    mode='test'
                )
            dl_test = DataLoader(
                    ds_test,
                    batch_size=256,
                    shuffle=False,
                    collate_fn=collate_fn
                )
            test_dl = tqdm(dl_test, total=len(dl_test),
                           desc=f'generating pair features for test set on fold {fold}')
            
            ids, feats = [], []
            for batch in test_dl:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                id = batch['id'].cpu().numpy()
                q1 = model(batch['q1_ids'], batch['q1_mask'])
                q2 = model(batch['q2_ids'], batch['q2_mask'])
                q1 = q1.detach().cpu().numpy()
                q2 = q2.detach().cpu().numpy()
                features = pair_features(q1, q2)
                ids.append(id), feats.append(features)
            
            ids = np.concatenate(ids)
            pair_feats = np.concatenate(feats, axis=0)
            infonce_pair_features = np.concatenate([ids[:, None], pair_feats], axis=1)
            np.save(f'artifacts/prediction/infonce_pair_features_{fold}.npy', infonce_pair_features.astype(np.float32))
    
    def merge(self):
        print('merging scattered features')
        train_feats = []
        for fold in range(5):
            prediction_path = f'artifacts/prediction/infonce_pair_features_{fold}.npy'
            train_path = f'artifacts/training/infonce_pair_features_{fold}.npy'
            test_features = np.load(prediction_path)
            train_features = np.load(train_path)
            if fold == 0:
                total_features = np.zeros([*test_features.shape], dtype=np.float32)
            total_features += test_features
            train_feats.append(train_features.astype(np.float32))
            os.remove(prediction_path)
            os.remove(train_path)
            
        total_features /= 5
        train_features = np.concatenate(train_feats, axis=0)
        sorted_index = train_features[:, 0].argsort()
        train_features = train_features[sorted_index]
        sorted_index = total_features[:, 0].argsort()
        test_features = total_features[sorted_index]
        
        np.save('artifacts/training/infonce_pair_features.npy', train_features)
        np.save('artifacts/prediction/infonce_pair_features.npy', total_features)

if __name__ == '__main__':
    bv = BuildVocab(
              'data/train.csv',
              'data/test.csv'
          )
    trainer = Trainer(bv)
    trainer.train()
    trainer.predict()
    trainer.merge()