import spacy
import torch
from torch.utils.data import Dataset

class QQPDataset(Dataset):
    def __init__(self,
                 data,
                 words_index,
                 max_len,
                 mode='train'):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.max_len = max_len
        self.words_index = words_index
        
        for c in ['question1', 'question2']:
            data[c] = data[c].fillna("").astype(str)
        
        if mode != 'test':
            columns = ['id', 'question1', 'question2', 'is_duplicate']
        else:
            columns = ['test_id', 'question1', 'question2']
        self.data = data[columns]
        nlp = spacy.blank('en')
        self.tokenizer = nlp.tokenizer
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        q1 = row['question1']
        q2 = row['question2']
        q1_tokens = self.tokenizer(q1)
        q2_tokens = self.tokenizer(q2)
        unk_idx = self.words_index['<unk>']
        q1_index = [self.words_index.get(token.text.lower(), unk_idx) for token in q1_tokens]
        q2_index = [self.words_index.get(token.text.lower(), unk_idx) for token in q2_tokens]
        q1_index = q1_index[:self.max_len]
        q2_index = q2_index[:self.max_len]
        q1_len = len(q1_index)
        q2_len = len(q2_index)
        if len(q1_index) < self.max_len:
            q1_index = q1_index + [self.words_index['<pad>']] * (self.max_len - len(q1_index))
        if len(q2_index) < self.max_len:
            q2_index = q2_index + [self.words_index['<pad>']] * (self.max_len - len(q2_index))
        q1_index = torch.tensor(q1_index, dtype=torch.long)
        q2_index = torch.tensor(q2_index, dtype=torch.long)
        
        if self.mode != 'test':
            id = row['id']
            y = row['is_duplicate']
            return id, q1_index, q2_index, q1_len, q2_len, y
        else:
            id = row['test_id']
            return id, q1_index, q2_index, q1_len, q2_len
    
    def __len__(self):
        return len(self.data)
        

#%%
