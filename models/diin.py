import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class DIIN(nn.Module):
    def __init__(self,
                 vocab,
                 vec_model,
                 emb_dim,
                 char_dim,
                 hidden_dim,
                 highway_layers,
                 self_attn_layers,
                 ):
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
        self.highway_layers = highway_layers
        self.self_attn_layers = self_attn_layers
        
        self.word_emb_dropout = nn.Dropout(0.05)
        self.char_emb_dropout = nn.Dropout(0.05)
        self.char_convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_dim,
                      out_channels=50,
                      kernel_size=kernel_size)
            for kernel_size in [3, 4, 5]
          ])
        highway_dim = 50*3 + emb_dim
        self.highway_lienar = nn.Linear(highway_dim, highway_dim)
        self.self_attn_linears = nn.ModuleList([
            nn.Linear(highway_dim*3, 1)
            for _ in range(self_attn_layers)
          ])
    
    def forward(self, batch):
        q1 = batch[1]
        q2 = batch[2]
        q1_len = batch[3]
        q2_len = batch[4]
        q1_char = batch[5].to(torch.long)
        q2_char = batch[6].to(torch.long)
        q1_mask = self._padded_mask(q1_len, q1.shape[1])
        q2_mask = self._padded_mask(q2_len, q2.shape[1])
        
        q1_emb = self.word_emb_dropout(self.word_embedding(q1))
        q2_emb = self.word_emb_dropout(self.word_embedding(q2))
        q1_char_emb = self._char_emb(q1_char)
        q2_char_emb = self._char_emb(q2_char)
        
        q1_emb = torch.cat([q1_emb, q1_char_emb], dim=-1)
        q2_emb = torch.cat([q2_emb, q2_char_emb], dim=-1)
        for _ in range(self.highway_layers):
            q1_emb = self._emb_highway(q1_emb)
            q2_emb = self._emb_highway(q2_emb)
        
        for i in range(self.self_attn_layers):
            self_att_q1 = self._self_attention(q1_emb, q1_mask, i)
            q1_emb = self_att_q1 + q1_emb
            self_att_q2 = self._self_attention(q2_emb, q2_mask, i)
            q2_emb = self_att_q2 + q2_emb
            
        return self_att_q1, self_att_q2
    
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
    
    def _emb_highway(self, x):
        transform = F.relu(self.highway_lienar(x))
        gate = F.sigmoid(self.highway_lienar(x))
        h = gate * transform + (1 - gate) * x
        return h
    
    def _padded_mask(self, q_len, max_len):
        ref_len = torch.arange(max_len)
        mask = ref_len <  q_len[:, None]
        return mask
      
    def _self_attention(self, x, q_mask, i):
        seq_len = x.shape[1]
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        
        x_i = x_i.repeat(1, 1, seq_len, 1)
        x_j = x_j.repeat(1, seq_len, 1, 1)
        x_interaction = x_i * x_j
        
        x_cat = torch.cat([x_i, x_j, x_interaction], dim=-1)
        x_attn = self.self_attn_linears[i](x_cat).squeeze(-1)
        q_mask = q_mask.unsqueeze(1)
        invalid_mask = ~q_mask
        logits = x_attn.masked_fill(invalid_mask, -1e-9)
        attn_weights = torch.softmax(logits, dim=-1)
        self_att = torch.matmul(attn_weights, x)
        return self_att
        
    
    # def _self_attention(self, x, q_mask, i):
    #     B, L, D = x.shape
    
    #     linear = self.self_attn_linears[i]
    #     w = linear.weight.squeeze(0)          # (3D,)
    #     b = linear.bias                       
    
    #     w1, w2, w3 = torch.split(w, D, dim=0)
    
    #     # s1: (B, L)
    #     s1 = torch.matmul(x, w1)
    #     # s2: (B, L)
    #     s2 = torch.matmul(x, w2)
    
    #     # s3: (B, L, L)
    #     x_scaled = x * w3                     # (B, L, D)
    #     s3 = torch.matmul(x_scaled, x.transpose(1, 2))
    
    #     # logits: (B, L, L)
    #     logits = s1.unsqueeze(2) + s2.unsqueeze(1) + s3 + b
    
    #     # mask over keys (last dim)
    #     if q_mask is not None:
    #         key_mask = q_mask.bool().unsqueeze(1)        # (B, 1, L)
    #         invalid_mask = ~key_mask                     # True where pad
    #         logits = logits.masked_fill(invalid_mask, float('-inf'))
    
    #     attn_weights = torch.softmax(logits, dim=-1)     # (B, L, L)
    #     self_att = torch.matmul(attn_weights, x)         # (B, L, D)
    #     return self_att
            

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
    vocab=bv,
    highway_layers=2,
    self_attn_layers=4
  )
import time
start = time.time()
a = model(batch)
end = time.time() - start
print(end)

#%%
