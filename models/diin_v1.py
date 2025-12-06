import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import math

class DenseNetLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 growthrate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels,
                              growthrate,
                              kernel_size=3,
                              padding=1,
                              bias=False)
    
    def forward(self, x):
        out = F.relu(self.bn(x))
        out = self.conv(x)
        out = torch.cat([x, out], dim=1)
        return out

class DenseNetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 growthrate,
                 num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(DenseNetLayer(in_channels, growthrate))
            in_channels += growthrate
        self.block = nn.Sequential(*layers)
        self.out_channels = in_channels
    
    def forward(self, x):
        return self.block(x)

class DenseNetTransition(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = F.relu(self.bn(x))
        x = self.conv(x)
        x = self.pool(x)
        return x

class TransformerSentEncoder(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_layers,
                 num_heads,
                 model_dim,
                 max_len=100,
                 dropout=0.1):
        super().__init__()

        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.register_buffer("position_ids", torch.arange(max_len).unsqueeze(0))  # (1, L)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=model_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,   # x: (B, L, D)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, L, D = x.size()

        # Add positional embeddings
        pos_ids = self.position_ids[:, :L]        # (1, L)
        x = x + self.pos_emb(pos_ids)             # broadcast to (B, L, D)

        # Transformer expects True=PAD, so invert
        pad_mask = ~mask

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.dropout(x)

        return x
  
class DIIN(nn.Module):
    def __init__(self,
                 vocab,
                 vec_model,
                 char_dim: int = 100,
                 emb_dim: int = 300,
                 sa_num_layers: int = 2,
                 sa_num_heads: int = 8,
                 sa_ff_dim: int = 512,
                 sa_dropout: float = 0.1,
                 cnn_base_channels: int = 64,
                 dense_growth_rate: int = 16,
                 dense_layers_per_block: int = 4,
                 cnn_dropout: float = 0.3):
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
        self.char_embedding = nn.Embedding(char_vocab_size, char_dim,
                                           padding_idx=0)
        self.char_convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_dim,
                      out_channels=50,
                      kernel_size=kernel_size)
            for kernel_size in [3, 4, 5]
          ])
        for conv in self.char_convs:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)
        self.emb_proj = nn.Linear(3*50+emb_dim, 256)
        self.word_emb_dropout = nn.Dropout(0.05)
        self.char_emb_dropout = nn.Dropout(0.05)
        
        highway_dim = 3*50+emb_dim
        self.highway_proj = nn.Linear(highway_dim, highway_dim)
        self.highway_gate = nn.Linear(highway_dim, highway_dim)
        self.self_attn_linears = nn.ModuleList([
            nn.Linear(highway_dim*4, 1, bias=True)
            for _ in range(sa_num_layers)
          ])
        
        self.sentence_encoder = TransformerSentEncoder(
            emb_dim=256,
            num_layers=sa_num_layers,
            num_heads=sa_num_heads,
            model_dim=sa_ff_dim,
            dropout=sa_dropout,
            max_len=40
        )
        self.interaction_in_channels = highway_dim * 4
        self.cnn_pre = nn.Conv2d(
            in_channels=self.interaction_in_channels,
            out_channels=cnn_base_channels,
            kernel_size=1,
            bias=False,
        )
        
        self.dense_block1 = DenseNetBlock(
            in_channels=cnn_base_channels,
            growthrate=dense_growth_rate,
            num_layers=dense_layers_per_block,
        )
        ch1 = self.dense_block1.out_channels
        self.trans1 = DenseNetTransition(ch1, cnn_base_channels)

        self.dense_block2 = DenseNetBlock(
            in_channels=cnn_base_channels,
            growthrate=dense_growth_rate,
            num_layers=dense_layers_per_block,
        )
        ch2 = self.dense_block2.out_channels
        self.trans2 = DenseNetTransition(ch2, cnn_base_channels)
        self.cnn_final_norm = nn.BatchNorm2d(cnn_base_channels)
        self.cnn_final_conv = nn.Conv2d(
            in_channels=cnn_base_channels,
            out_channels=cnn_base_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.cnn_dropout = nn.Dropout(cnn_dropout)
        classifier_in_dim = 2 * cnn_base_channels
        self.fc1 = nn.Linear(classifier_in_dim, 1)
        self.fc2 = nn.Linear(256, 1)
        self.classifier_dropout = nn.Dropout(cnn_dropout)
        self._init_weights()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.sa_num_layers = sa_num_layers
        
        self.Wq = nn.Linear(highway_dim, 64, bias=False)
        self.Wk = nn.Linear(highway_dim, 64, bias=False)
        self.U = nn.Linear(highway_dim, 64)
        self.V = nn.Linear(highway_dim, 64)
        self.v = nn.Parameter(torch.randn(64))
        self.att_mix = nn.Linear(2, 1)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                if m.weight.requires_grad:
                    nn.init.uniform_(m.weight, -0.1, 0.1)
    
    def _padded_mask(self, q_len, max_len):
        ref_len = torch.arange(max_len).to(self.device)
        mask = ref_len < q_len[:, None]
        return mask
    
    def _emb_highway(self, x):
        transform = F.tanh(self.highway_proj(x))
        gate = torch.sigmoid(self.highway_gate(x))
        # h = gate * transform + (1 - gate) * x
        h =  gate * transform
        return h
    
    def build_interaction(self, q1, q2) -> torch.Tensor:
        B, L, D = q1.size()
        q1_exp = q1.unsqueeze(2).expand(-1, -1, L, -1)
        q2_exp = q2.unsqueeze(1).expand(-1, L, -1, -1)
    
        diff = q1_exp - q2_exp
        prod = q1_exp * q2_exp
        interaction = torch.cat([q1_exp, q2_exp, diff, prod], dim=-1)
        interaction = interaction.permute(0, 3, 1, 2).contiguous()
        return interaction
    
    def forward(self, batch, return_embedding=False):
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
        # for _ in range(1):
        # q1_emb = self._emb_highway(q1_emb)
        # q2_emb = self._emb_highway(q2_emb)

        for i in range(self.sa_num_layers):
            q1_enc = self._self_attention(q1_emb, q1_mask, i)
            q1_emb = q1_enc + q1_emb
            q2_enc = self._self_attention(q2_emb, q2_mask, i)
            q2_emb = q2_enc + q2_emb

        inter = self.build_interaction(q1_enc, q2_enc)
        x = self.cnn_pre(F.relu(inter))
        x = self.dense_block1(x)
        x = self.trans1(x)
        x = self.dense_block2(x)
        x = self.trans2(x)
        x = self.cnn_final_conv(F.relu(self.cnn_final_norm(x)))
        x = self.cnn_dropout(x)
        max_pool = F.adaptive_max_pool2d(x, output_size=1).squeeze(-1).squeeze(-1)
        avg_pool = F.adaptive_avg_pool2d(x, output_size=1).squeeze(-1).squeeze(-1)
        feat = torch.cat([max_pool, avg_pool], dim=-1)
        # h = self.classifier_dropout(F.relu(self.fc1(feat)))
        logits = self.fc1(F.relu(feat))
        if return_embedding:
            return logits, F.sigmoid(logits)
        else:
            return logits
    
    def _char_emb(self, q_char):
        B, T, C = q_char.shape
        q_char = q_char.reshape(-1, C)
        q_char = self.char_embedding(q_char)
        q_char = q_char.transpose(1, 2)
        conv_outputs = []
        for conv in self.char_convs:
            h = conv(q_char)
            h = F.relu(h)
            h, _ = h.max(dim=2)
            conv_outputs.append(h)
        char_emb = torch.cat(conv_outputs, dim=-1)
        char_emb = char_emb.reshape(B, T, -1)
        char_emb = self.char_emb_dropout(char_emb)
        return char_emb
    
    # def _self_attention(self, x, q_mask, i):
    #     B, L, D = x.shape

    #     linear = self.self_attn_linears[i]
    #     w = linear.weight.squeeze(0)      
    #     b = linear.bias                   
    #     w1, w2, w3 = torch.split(w, D, dim=0)   
    #     s1 = torch.matmul(x, w1)                
    #     s2 = torch.matmul(x, w2)                
    #     x_scaled = x * w3
    #     s3 = torch.matmul(x_scaled, x.transpose(1, 2))
    #     logits = s1.unsqueeze(2) + s2.unsqueeze(1) + s3 + b
    #     if q_mask is not None:
    #         key_mask = q_mask.bool().unsqueeze(1)
    #         invalid_mask = ~key_mask
    #         logits = logits.masked_fill(invalid_mask, -1e9)
        
    #     attn_weights = torch.softmax(logits, dim=-1)
    #     self_att = torch.matmul(attn_weights, x)
    
    #     return self_att
    
    # def _self_attention(self, x, q_mask, i):
    #     B, L, D = x.size()
    #     q = self.Wq(x)  # (B,L,d)
    #     k = self.Wk(x)  # (B,L,d)

    #     scores = torch.matmul(q, k.transpose(1, 2)) # (B,L,L)

    #     if q_mask is not None:
    #         key_mask = q_mask.unsqueeze(1)   # (B,1,L)
    #         mask_value = torch.finfo(scores.dtype).min
    #         scores = scores.masked_fill(~key_mask, mask_value)

    #     attn = torch.softmax(scores, dim=-1)  # (B,L,L)
    #     out = attn @ x                        # (B,L,D)
    #     return out
    
    def _self_attention(self, x, q_mask, i):
        B, L, D = x.shape
    
        linear = self.self_attn_linears[i]        # nn.Linear(4*D, 1)
        w = linear.weight.squeeze(0)              # (4D,)
        b = linear.bias
    
        # split into [x_i, x_j, x_i-x_j, x_i*x_j] parts
        u1, u2, u3, u4 = torch.split(w, D, dim=0)  # each (D,)
        w1 = u1 + u3      # for x_i
        w2 = u2 - u3      # for x_j
        w3 = u4           # for x_i * x_j term
    
        s1 = torch.matmul(x, w1)                  # (B, L)
        s2 = torch.matmul(x, w2)                  # (B, L)
    
        x_scaled = x * w3                         # (B, L, D)
        s3 = torch.matmul(x_scaled, x.transpose(1, 2))  # (B, L, L)
        logits1 = s1.unsqueeze(2) + s2.unsqueeze(1) + s3 + b
        
        q = self.Wq(x)  # (B,L,d)
        k = self.Wk(x)  # (B,L,d)
        logits2 = torch.matmul(q, k.transpose(1, 2)) # (B,L,L)

        # Q = self.U(x)
        # K = self.V(x)
        # Q = Q.unsqueeze(2)
        # K = K.unsqueeze(1)
        # z = torch.tanh(Q + K)
        # logits3 = torch.einsum('bijh,h -> bij', z, self.v)
        
        scores = torch.stack([logits1, logits2], dim=-1)
        scores = self.att_mix(scores).squeeze(dim=-1)
        
        if q_mask is not None:
            key_mask = q_mask.unsqueeze(1)   # (B,1,L)
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~key_mask, mask_value)

        attn = torch.softmax(scores, dim=-1)  # (B,L,L)
        out = attn @ x                        # (B,L,D)
        return out



#%%
# import os
# from tqdm import tqdm
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

# model = DIIN(
#     vocab=bv,
#     vec_model=glove,
#     char_dim=100,
#     emb_dim=300,
#     sa_num_layers=1,
#     sa_num_heads=4,
#     sa_ff_dim=512,
#     sa_dropout=0.1,
#     cnn_base_channels=64,
#     dense_growth_rate=16,
#     dense_layers_per_block=4,
#     cnn_dropout=0.1
# ).cuda()

# for i, v in enumerate(batch):
#     if isinstance(v, torch.Tensor):
#         batch[i] = v.cuda()
# import time
# start = time.time()
# a = model(batch)
# end = time.time() - start
# print(f'{end:.5f}')
#%%
# x = torch.rand(32,10,10)
# torch.stack([x,x],dim=-1).shape