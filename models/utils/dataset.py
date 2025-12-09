from torch.utils.data import Dataset

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
        

#%%


