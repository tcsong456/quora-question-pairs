import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ESIM(nn.Module):
    def __init__(self,
                 vocab,
                 vec_model,
                 emb_dim=300,
                 char_dim=100,
                 hidden_dim=150):
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
        self.char_dropout = nn.Dropout(0.05)
        self.word_dropout = nn.Dropout(0.05)
        self.fm_dropout = nn.Dropout(0.1)
        self.final_dropout = nn.Dropout(0.1)
        self.char_lstm = nn.LSTM(
            input_size=char_dim,
            hidden_size=char_dim,
            batch_first=True
          )
        self.lstm_encoder = nn.LSTM(
            input_size=emb_dim+char_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
          )
        self.composition = nn.LSTM(hidden_dim,
                                   hidden_dim,
                                   batch_first=True,
                                   bidirectional=True)
        self.fm_compression = nn.Linear(8*hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(8*hidden_dim, 100)
        self.fc2 = nn.Linear(100, 1)
        
        self.char_dim = char_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, batch, return_embedding=False):
        q1 = batch[1]
        q2 = batch[2]
        q1_len = batch[3]
        q2_len = batch[4]
        q1_char = batch[5].to(torch.long)
        q2_char = batch[6].to(torch.long)
        B, T, C = q1_char.shape
        q1_mask = self._padded_mask(q1_len, q1.shape[1])
        q2_mask = self._padded_mask(q2_len, q2.shape[1])
        
        q1_char_len = (q1_char > 0).sum(dim=-1).view(-1)
        q2_char_len = (q2_char > 0).sum(dim=-1).view(-1)
        q1_char_mask = self._padded_mask(q1_char_len, C)
        q2_char_mask = self._padded_mask(q2_char_len, C)
        q1_char = q1_char.reshape(B*T, -1)
        q2_char = q2_char.reshape(B*T, -1)
        q1_char_len[q1_char_len==0] = 1
        q2_char_len[q2_char_len==0] = 1
        
        q1_char_emb = self.char_embedding(q1_char)
        q2_char_emb = self.char_embedding(q2_char)
        q1_char_temp_mask = (1 - q1_char_mask.float()) * -1e7
        q2_char_temp_mask = (1 - q2_char_mask.float()) * -1e7
        q1_char_max = (q1_char_emb * q1_char_mask.unsqueeze(-1) + q1_char_temp_mask.unsqueeze(-1)).max(dim=1)[0]
        q2_char_max = (q2_char_emb * q2_char_mask.unsqueeze(-1) + q2_char_temp_mask.unsqueeze(-1)).max(dim=1)[0]
        q1_char_emb = q1_char_max.reshape(B, T, self.char_dim)
        q2_char_emb = q2_char_max.reshape(B, T, self.char_dim)
        
        q1_emb = self.word_embedding(q1)
        q2_emb = self.word_embedding(q2)
        q1_emb = torch.cat([q1_emb, q1_char_emb], dim=-1)
        q2_emb = torch.cat([q2_emb, q2_char_emb], dim=-1)
        q1_enc = self._handle_lstm(q1_emb, q1_len, self.lstm_encoder, self.word_dropout)
        q2_enc = self._handle_lstm(q2_emb, q2_len, self.lstm_encoder, self.word_dropout)
        attended_q1 = self._cross_attention(q1_enc, q2_enc, q1_mask, q2_mask)
        attended_q2 = self._cross_attention(q2_enc, q1_enc, q2_mask, q1_mask)
        q1_inter_feature_map = torch.cat([q1_enc, attended_q1, q1_enc-attended_q1, q1_enc * attended_q1], dim=-1)
        q2_inter_feature_map = torch.cat([q2_enc, attended_q2, q2_enc-attended_q2, q2_enc * attended_q2], dim=-1)
        q1_inter = self.fm_dropout(F.relu(self.fm_compression(q1_inter_feature_map)))
        q2_inter = self.fm_dropout(F.relu(self.fm_compression(q2_inter_feature_map)))
        q1_inter = self._handle_lstm(q1_inter, q1_len, self.composition)
        q2_inter = self._handle_lstm(q2_inter, q2_len, self.composition)
        
        q1_avg = torch.sum(q1_mask.unsqueeze(1) * q1_inter.transpose(1, 2), dim=-1) / torch.sum(q1_mask, dim=1, keepdim=True)
        q2_avg = torch.sum(q2_mask.unsqueeze(1) * q2_inter.transpose(1, 2), dim=-1) / torch.sum(q2_mask, dim=1, keepdim=True)
        
        q1_temp_mask = -1e7 * (1 - q1_mask.float())
        q2_temp_mask = -1e7 * (1 - q2_mask.float())
        q1_max = (q1_inter * q1_mask.unsqueeze(-1) + q1_temp_mask.unsqueeze(-1)).max(dim=1)[0]
        q2_max = (q2_inter * q2_mask.unsqueeze(-1) + q2_temp_mask.unsqueeze(-1)).max(dim=1)[0]
        
        feature_map = torch.cat([q1_avg, q2_avg, q1_max, q2_max], dim=-1)
        h = self.final_dropout(feature_map)
        h = torch.tanh(self.fc1(h))
        h = self.final_dropout(h)
        logits = self.fc2(h) 
        if return_embedding:
            preds = F.sigmoid(logits)
            output = torch.cat([preds, h], dim=-1)
            return logits, output
        else:
            return logits
    
    def _handle_lstm(self, x, x_len, lstm_func, dropout_func=None):
        N, T, D = x.shape 
        x = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        x_packed_output, _ = lstm_func(x)
        x_output, _ = pad_packed_sequence(x_packed_output, batch_first=True, total_length=T)
        if dropout_func is not None:
            x = dropout_func(x_output)
        else:
            x = x_output
        return x
    
    def _padded_mask(self, q_len, max_len):
        ref_len = torch.arange(max_len).to(self.device)
        mask = ref_len < q_len[:, None]
        return mask
      
    def _cross_attention(self, x1, x2, x1_mask, x2_mask):
        att = x1.bmm(x2.transpose(1,2 ))
        x2_mask = x2_mask.bool().unsqueeze(1)
        logits = att.masked_fill(~x2_mask, -1e4)
        attn = torch.softmax(logits, dim=-1)
        out = attn @ x2
        out = out * x1_mask.bool().unsqueeze(-1)
        return out

#%%