from torch.utils.data import Dataset
from transformers import AutoTokenizer

class QQPDataset(Dataset):
    def __init__(self,
                 bv,
                 q_idx,
                 mode='train'):
        assert mode in ['train', 'val', 'test']
        sent_idx = bv.load_arrays()
        self.sent_idx = sent_idx
        data = bv.train_data.values if mode != 'test' else bv.test_data.values
        self.q_idx = q_idx
        self.data = data
        self.mode = mode
    
    def __len__(self):
      return len(self.q_idx)
    
    def __getitem__(self, index):
        idx = self.q_idx[index]
        base_data = 'train' if self.mode != 'test' else 'test'
        q1_key = f'{base_data}_question1'
        q2_key = f'{base_data}_question2'
        q1_len_key = f'{q1_key}_len'
        q2_len_key = f'{q2_key}_len'
        q1_char_key = f'{q1_key}_char'
        q2_char_key = f'{q2_key}_char'
        q1_data = self.sent_idx[q1_key]
        q2_data = self.sent_idx[q2_key]
        q1_lens = self.sent_idx[q1_len_key]
        q2_lens = self.sent_idx[q2_len_key]
        q1_chars = self.sent_idx[q1_char_key]
        q2_chars = self.sent_idx[q2_char_key]
        
        q1 = q1_data[idx]
        q2 = q2_data[idx]
        q1_len = q1_lens[idx]
        q2_len = q2_lens[idx]
        q1_char = q1_chars[idx]
        q2_char = q2_chars[idx]
        row = self.data[idx]
        id = row[0]
        if self.mode != 'test':
            y = row[-1]
            return id, q1, q2, q1_len, q2_len, q1_char, q2_char, y
        else:
            return id, q1, q2, q1_len, q2_len, q1_char, q2_char
          
class SBERTDataset(Dataset):
    def __init__(self,
                 bv,
                 q_idx,
                 mode='train'):
        assert mode in ['train', 'val', 'test']
        data = bv.train_data if mode != 'test' else bv.test_data
        self.q_idx = q_idx
        self.data = data
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.max_len = 40
    
    def __len__(self):
      return len(self.q_idx)
    
    def __getitem__(self, index):
        idx = self.q_idx[index]
        row = self.data.iloc[idx]
        q1 = str(row["question1"])
        q2 = str(row["question2"])
        
        enc1 = self.tokenizer(
            q1,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        enc2 = self.tokenizer(
            q2,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        q1_input_ids = enc1["input_ids"].squeeze(0),
        q1_attention_mask = enc1["attention_mask"].squeeze(0),
        q2_input_ids = enc2["input_ids"].squeeze(0),
        q2_attention_mask = enc2["attention_mask"].squeeze(0),
            
        if self.mode != 'test':
            label = float(row["is_duplicate"])
            return row['id'], q1_input_ids[0], q2_input_ids[0], q1_attention_mask[0], q2_attention_mask[0], label
        else:
            return row['test_id'], q1_input_ids[0], q2_input_ids[0], q1_attention_mask[0], q2_attention_mask[0]

class DeBERTaV3Dataset(Dataset):
    def __init__(self,
                 bv,
                 q_idx,
                 mode='tain'):
        assert mode in ['train', 'val', 'test']
        data = bv.train_data if mode != 'test' else bv.test_data
        self.q_idx = q_idx
        self.data = data
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base",
                                                       use_fast=False)
        self.max_len = 40
    
    def __len__(self):
      return len(self.q_idx)
    
    def __getitem__(self, index):
        idx = self.q_idx[index]
        row = self.data.iloc[idx]
        q1 = row['question1']
        q2 = row['question2']
        enc = self.tokenizer(
            q1,
            q2,
            padding='max_length',
            truncation=True,
            max_length=80,
            return_tensors='pt'
          )
        
        input_ids = enc['input_ids'].squeeze(0)
        att_mask = enc['attention_mask'].squeeze(0)
        
        if self.mode != 'test':
            label = float(float(row['is_duplicate']))
            return row['id'], input_ids, att_mask, label
        else:
            return row['test_id'], input_ids, att_mask
#%%
# import numpy as np
# from models.utils.build_vocab import BuildVocab
# from torch.utils.data import DataLoader
# bv = BuildVocab(
#         'data/train.csv',
#         'data/test.csv'
#     )
# train = bv.train_data
# dataset = QQPDataset(bv, np.arange(train.shape[0]), mode='train')
# dl = DataLoader(dataset,
#                 batch_size=64,
#                 shuffle=False)
# for batch in dl:
#     break

        
