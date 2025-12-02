import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class DIIN(nn.Module):
    def __init__(self,
                 emb_dim,
                 char_dim,
                 hidden_dim,
                 vec_model,
                 vocab):
        super().__init__()
        word_to_idx = vocab.load_dict()
        vocab_size = len(word_to_idx)
        char_vocab_size = len(vocab.char_index)
        embedding_matrix = np.random.normal(scale=0.01, 
                                            size=(vocab_size, emb_dim)).astype(np.float32)
        embedding_matrix[word_to_idx['<pad>']] = np.zeros(emb_dim)
        for word, idx in word_to_idx.items():
            if word == '<pad>':
                continue
            if word in vec_model:
                embedding_matrix[idx] = vec_model[word]
        self.word_embedding = nn.Embedding(vocab_size, emb_dim,
                                           padding_idx=word_to_idx['<pad>'])
        self.word_embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.word_embedding.weight.requires_grad = True
        self.char_embedding = nn.Embedding(char_vocab_size, char_dim)
        self.char_dim = char_dim
        
        self.word_emb_dropout = nn.Dropout(0.05)
        self.char_emb_dropout = nn.Dropout(0.05)
        self.char_convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_dim,
                      out_channels=50,
                      kernel_size=kernel_size)
            for kernel_size in [3, 4, 5]
          ])
    
    def forward(self, batch):
        q1 = batch[1]
        q2 = batch[2]
        q1_len = batch[3]
        q2_len = batch[4]
        q1_char = batch[5].to(torch.long)
        q2_char = batch[6].to(torch.long)
        
        q1_emb = self.word_emb_dropout(self.word_embedding(q1))
        q2_emb = self.word_emb_dropout(self.word_embedding(q2))
        q1_char_emb = self._char_emb(q1_char)
        q2_char_emb = self._char_emb(q2_char)
        
        q1_emb = torch.cat([q1_emb, q1_char_emb], dim=-1)
        q2_emb = torch.cat([q2_emb, q2_char_emb], dim=-1)
        return q1_emb, q2_emb
    
    def _char_emb(self, q_char):
        B, T, C = q_char.shape
        q_char = q_char.reshape(-1, C)
        q_char = self.char_emb_dropout(self.char_embedding(q_char))
        q_char = q_char.transpose(1, 2)
        conv_outputs = []
        for conv in self.char_convs:
            h = conv(q_char)
            h = F.relu(h)
            h, _ = h.max(dim=2)
            conv_outputs.append(h)
        char_emb = torch.cat(conv_outputs, dim=-1)
        char_emb = char_emb.reshape(B, T, -1)
        return char_emb

#%%
# glove = {}
# path = 'artifacts/glove.840B.300d.txt'
# file_size = os.path.getsize(path)
# with open(path, 'r', encoding='utf8') as f, \
#     tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading GloVe") as pbar:
#         for line in f:
#             parts = line.rstrip().split(' ')
#             word = parts[0]
#             vec = np.array(parts[1:], dtype=np.float32)
#             glove[word] = vec
#             pbar.update(len(line.encode('utf8')))
# bv = BuildVocab('data/train.csv',
#                 'data/test.csv')
# bv.build_char_vocab()

model = DIIN(
    emb_dim=300,
    char_dim=100,
    hidden_dim=100,
    vec_model=glove,
    vocab=bv
  )
a, b = model(batch)

#%%
b.shape