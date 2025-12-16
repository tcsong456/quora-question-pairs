import re
import torch
import spacy
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from models.utils.soft_alignment import build_alignments_for_batch

def simple_word_tokenizer(text):
    _WORD_RE = re.compile(r"[a-zA-Z]+|\d+(?:\.\d+)?")
    return _WORD_RE.findall(text)
nlp_tok = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)

def spacy_tokens_drop_punct(doc):
    return [t.text for t in doc if not t.is_space and not t.is_punct]

def sa_collate_fn(
    tokenizer_name,
    neg_bias,
    max_len,
    batch_size=32,
    mode='train'
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    def collate(batch):
        q1_raw = [ex['question1'] for ex in batch]
        q2_raw = [ex['question2'] for ex in batch]
        docs1 = list(nlp_tok.pipe(q1_raw, batch_size=batch_size))
        docs2 = list(nlp_tok.pipe(q2_raw, batch_size=batch_size))
        
        q1_words = [spacy_tokens_drop_punct(d) for d in docs1]
        q2_words = [spacy_tokens_drop_punct(d) for d in docs2]
        aligned_pairs = build_alignments_for_batch(q1_raw, q2_raw, device=device, topk=3, batch_size=batch_size)

        
        enc = tokenizer(
                q1_words,
                q2_words,
                padding=True,
                truncation=True,
                max_length=max_len,
                is_split_into_words=True,
                return_tensors='pt'
            )
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask'].bool()
        if mode == 'train':
            labels = torch.tensor([ex['label'] for ex in batch], dtype=torch.int8)
        else:
            labels = None
        
        B, L = input_ids.shape
        bias = torch.zeros((B, L, L), dtype=torch.float32)
        
        for b in range(B):
            word_ids = enc.word_ids(batch_index=b)
            seq_ids = enc.sequence_ids(batch_index=b)
            
            q1_pos, q2_pos = [], []
            q1_wid, q2_wid = {}, {}
            for t in range(L):
                if not attention_mask[b, t]:
                    continue
                sid = seq_ids[t]
                wid = word_ids[t]
                if sid == 0:
                    q1_pos.append(t)
                    q1_wid.setdefault(wid, []).append(t)
                elif sid == 1:
                    q2_pos.append(t)
                    q2_wid.setdefault(wid, []).append(t)
                
            if not q1_pos or not q2_pos:
                continue
            
            aligned_pair = aligned_pairs[b]
            q1_pos_t = torch.tensor(q1_pos, dtype=torch.long)
            q2_pos_t = torch.tensor(q2_pos, dtype=torch.long)
            
            if aligned_pair:
                bias[b][q1_pos_t[:, None], q2_pos_t[None, :]] = -neg_bias
                bias[b][q2_pos_t[:, None], q1_pos_t[None, :]] = -neg_bias
            
                for wi, wj, conf in aligned_pair:
                    ti_list = q1_wid.get(wi)
                    tj_list = q2_wid.get(wj)
                    if not ti_list or not tj_list:
                        continue
                    ti = torch.tensor(ti_list, dtype=torch.long)
                    tj = torch.tensor(tj_list, dtype=torch.long)
    
                    val = float(conf)
                    bias[b][ti[:, None], tj[None, :]] = val
                    bias[b][tj[:, None], ti[None, :]] = val
    
        return torch.tensor([ex['id'] for ex in batch], dtype=torch.int32), input_ids, attention_mask, bias, labels
    return collate

class SaDataset(Dataset):
    def __init__(self,
                 bv,
                 q_idx,
                 mode='train'):
        assert mode in ['train', 'val', 'test']
        data = bv.train_data if mode != 'test' else bv.test_data
        self.q_idx = q_idx
        self.data = data
        self.mode = mode
        self.max_len = 40
    
    def __len__(self):
      return len(self.q_idx)
     
    def __getitem__(self, index):
        idx = self.q_idx[index]
        row = self.data.iloc[idx]
        q1 = str(row["question1"])
        q2 = str(row["question2"])
        if self.mode == 'train':
            label = row['is_duplicate']
            return {
                    'id': row['id'],
                    'question1': q1,
                    'question2': q2,
                    'label': label
                    }
        else:
            return {
                    'id': row['test_id'],
                    'question1': q1,
                    'question2': q2
                    }

#%%
import numpy as np
from torch.utils.data import DataLoader
from models.utils.build_vocab import BuildVocab
# bv = BuildVocab('data/train.csv',
#                 'data/test.csv')
# train = bv.train_data
# test = bv.test_data
dataset = SaDataset(bv, np.arange(train.shape[0]),
                    mode='train')
collate = sa_collate_fn(
    tokenizer_name="microsoft/deberta-v3-base",
    neg_bias=2.0,
    max_len=80,
    mode='train',
    batch_size=64
    )
dl = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate
    )
# for batch in dl:
#     break
#%%
from tqdm import tqdm
# start = time.time()
# for batch in tqdm(dl, total=len(dl)):
#     pass
# end = time.time() - start
# end

B, T, C = batch[3].shape
(batch[3] > 0).sum() / (B*T*C)