import pickle
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class InfoNceDataset(Dataset):
    def __init__(self,
                 data):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.max_len = 40
        
        with open('artifacts/q1_neg_sample.pkl', 'rb') as f:
            self.q1_neg_dict = pickle.load(f)
        
        with open('artifacts/q2_neg_sample.pkl', 'rb') as f:
            self.q2_neg_dict = pickle.load(f)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        q1 = row['question1'].strip()
        q2 = row['question2'].strip()
        q1_hard_neg = self.q1_neg_dict[q1]
        q2_hard_neg = self.q2_neg_dict[q2]
        
        enc1 = self.tokenizer(
                q1,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )
        enc2 = self.tokenizer(
                q2,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )
        enc_neg1 = self.tokenizer(
                q1_hard_neg,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )
        enc_neg2 = self.tokenizer(
                q2_hard_neg,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )
        # q1_neg_ids = enc_neg1['input_ids']
        # q1_neg_mask = enc_neg1['attention_mask']
        # q2_neg_ids = enc_neg2['input_ids']
        # q2_neg_mask = enc_neg2['attention_mask']
        # q1_enc = enc1['input_ids']
        # q2_enc = enc2['input_ids']
        # q1_mask = enc1['attention_mask']
        # q2_mask = enc2['attention_mask']

        return {
            "q1_ids": enc1["input_ids"].squeeze(0),          # (L,)
            "q1_mask": enc1["attention_mask"].squeeze(0),    # (L,)
            "q2_ids": enc2["input_ids"].squeeze(0),          # (L,)
            "q2_mask": enc2["attention_mask"].squeeze(0),    # (L,)
            "q1n_ids": enc_neg1["input_ids"],                # (K, L)
            "q1n_mask": enc_neg1["attention_mask"],          # (K, L)
            "q2n_ids": enc_neg2["input_ids"],                # (K, L)
            "q2n_mask": enc_neg2["attention_mask"],          # (K, L)
        }

def infoNCE_with_hard_negatives(
            q1, q2,
            q1_neg, q2_neg,
            temperature=0.05
        ):
    B, H = q1.shape
    K = q2_neg.shape[1]
    labels = torch.arange(B, device=q1.device)
    
    u = torch.cat([q1, q1.reshape(B*K, H)], dim=0)
    logit_v2u = q2 @ u.T / temperature
    loss_v2u = F.cross_entropy(logit_v2u, labels)
    
    v = torch.cat([q2, q2_neg.reshape(B*K, H)], dim=0)
    logit_u2v = q1 @ v.T / temperature
    loss_u2v = F.cross_entropy(logit_u2v, labels)
    
    return 0.5 * (loss_v2u + loss_u2v)

#%%
import numpy as np
from torch.utils.data import DataLoader
from models.utils.build_vocab import BuildVocab
# bv = BuildVocab(
#           'data/train.csv',
#           'data/test.csv'
#       )
# train = bv.train_data
pos_data = train[train['is_duplicate']==1]

ds = InfoNceDataset(
        pos_data
    )
# collate_fn = make_collate_fn(mode='train')
dl = DataLoader(ds, batch_size=128, shuffle=True)
for batch in dl:
    break

#%%