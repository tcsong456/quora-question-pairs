import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TinyEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_ff=1024, dropout=0.1, layers=2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, x, pad_mask=None):
        return self.enc(x, src_key_padding_mask=pad_mask)

class TransformerDIIN(nn.Module):
    def __init__(self,
                 emb_dim,
                 words_index_dict,
                 max_len,
                 vec_model):
        super().__init__()
        embedding_matrix = np.zeros((len(words_index_dict), emb_dim), dtype=np.float32)
        for word, idx in words_index_dict.items():
            if word == '<pad>':
                continue
            vec = vec_model.wv.get_vector(word)
            embedding_matrix[idx] = vec
        embedding_matrix = torch.from_numpy(embedding_matrix)
        self.words_embedding = nn.Embedding(
            num_embeddings=len(words_index_dict),
            embedding_dim=emb_dim,
            padding_idx=words_index_dict['<pad>']
          )
        self.words_embedding.weight.data.copy_(embedding_matrix)
        self.words_embedding.weight.requires_grad = True
        
        self.proj = nn.Linear(emb_dim, 192)
        self.encoder = TinyEncoder(
                d_model=192,
                nhead=6,
                dim_ff=768,
                layers=1,
                dropout=0.1
            )
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU()
            )
        self.final_layer = nn.Linear(64, 1)
        
        self.max_len = max_len
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
        nn.init.xavier_uniform_(self.final_layer.weight, gain=0.1)
        nn.init.zeros_(self.final_layer.bias)
        
    def masked_softmax(self, logits, mask, dim):
        logits = logits.masked_fill(~mask, -1e4)
        return torch.softmax(logits, dim=dim)
    
    def dot_attention(self, q, k, k_mask):
        D = q.size(-1)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(D)
        attn = self.masked_softmax(scores, k_mask[:, None, :], dim=-1)
        return torch.matmul(attn, k)
    
    def build_similarity_channels(
        self,
        q1_enc, q2_enc,
        q1_mask, q2_mask,
        q1_att, q2_att,
        q1_ids, q2_ids
    ):
        q1e = F.normalize(q1_enc, dim=-1)
        q2e = F.normalize(q2_enc, dim=-1)
        q1a = F.normalize(q1_att, dim=-1)
        q2a = F.normalize(q2_att, dim=-1)
    
        S_enc = torch.matmul(q1e, q2e.transpose(1, 2))

        S_att = torch.matmul(q1a, q2e.transpose(1, 2))

        S_att2 = torch.matmul(q1e, q2a.transpose(1, 2))
        
        m = (q1_ids.unsqueeze(2) == q2_ids.unsqueeze(1))
        
        pair_mask = q1_mask[:, :, None] & q2_mask[:, None, :]
        S_enc = S_enc.masked_fill(~pair_mask, 0.0)
        S_att = S_att.masked_fill(~pair_mask, 0.0)
        S_att2 = S_att2.masked_fill(~pair_mask, 0.0)
        m = m & pair_mask
    
        return torch.stack([S_enc, S_att, S_att2, m], dim=1)
    
    def mask(self, q_len):
        ref_len = torch.arange(self.max_len).to(self.device)
        q_mask = ref_len[None] < q_len[:, None]
        return q_mask
    
    def _handle_lstm(self, x, x_len, lstm_func, dropout_func):
        x = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        x_packed_output, _ = lstm_func(x)
        x_output, _ = pad_packed_sequence(x_packed_output, batch_first=True, total_length=self.max_len)
        x_output = dropout_func(x_output)
        return x_output
    
    def token_metrics(self, q, q_att):
        abs_diff = torch.abs(q - q_att)
        xn = F.normalize(q, p=2, dim=-1, eps=1e-7)
        tn = F.normalize(q_att, p=2, dim=-1, eps=1e-7)
        cos = (xn * tn).sum(dim=-1, keepdim=True)
        l2 = torch.sqrt(((q - q_att)**2).sum(axis=-1, keepdim=True) + 1e-7)
        feats = torch.cat([q, q_att, abs_diff, cos, l2], dim=-1)
        return feats

    def forward(self, batch, return_embedding=False):
        q1 = batch[1]
        q2 = batch[2]
        q1_len = batch[3]
        q2_len = batch[4]
        
        q1_mask = self.mask(q1_len)
        q2_mask = self.mask(q2_len)
        q1_emb = self.proj(self.words_embedding(q1))
        q2_emb = self.proj(self.words_embedding(q2))
        
        q1_enc = self.encoder(q1_emb, pad_mask=~q1_mask)
        q2_enc = self.encoder(q2_emb, pad_mask=~q2_mask)
        q1_att = self.dot_attention(q1_enc, q2_enc, q2_mask)
        q2_att = self.dot_attention(q2_enc, q1_enc, q1_mask)
        x = self.build_similarity_channels(
            q1_enc, q2_enc,
            q1_mask, q2_mask,
            q1_att, q2_att,
            q1, q2
        )
        feat = self.conv(x)
        f_max = feat.amax(dim=(2,3))
        f_mean = feat.mean(dim=(2,3))
        cnn_feat = torch.cat([f_max, f_mean], dim=-1)
        
        logit = self.final_layer(F.relu(cnn_feat))
        if return_embedding:
            preds = F.sigmoid(logit)
            output = torch.cat([preds, cnn_feat], dim=-1)
            return logit, output
        else:
            return logit
