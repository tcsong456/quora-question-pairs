import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from gensim.models.fasttext import load_facebook_model
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiMPM(nn.Module):
    def __init__(self,
                 emb_dim,
                 hidden_size,
                 words_index_dict,
                 mul_dim,
                 use_multi_head=False):
        super().__init__()
        vec_model = load_facebook_model('artifacts/cc.en.300.bin')
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
        self.hidden_size = hidden_size
        self.mul_dim = mul_dim
        self.use_multi_head = use_multi_head
        
        self.bi_lstm = nn.LSTM(emb_dim,
                               hidden_size,
                               batch_first=True,
                               bidirectional=True,
                               num_layers=1)
        self.dropout = nn.Dropout(0.2)
        self.full_matching_weights = nn.Parameter(torch.empty(mul_dim, hidden_size))
        nn.init.xavier_uniform_(self.full_matching_weights)
        self.pool_matching_weights = nn.Parameter(torch.empty(mul_dim, hidden_size))
        nn.init.xavier_uniform_(self.pool_matching_weights)
        self.attentive_matching_weights = nn.Parameter(torch.empty(mul_dim, hidden_size))
        nn.init.xavier_uniform_(self.attentive_matching_weights)
        if self.use_multi_head:
            self.multiplicative_linear = nn.Linear(hidden_size, 100)
            self.additive_linear = nn.Linear(hidden_size, 100)
            self.additive_dot_weights = nn.Parameter(torch.empty(1, 100))
            nn.init.xavier_uniform_(self.additive_dot_weights)
    
    def forward(self, batch):
        matches = []
        q1_input = batch[1]
        q2_input = batch[2]
        q1_lengths = batch[3]
        q2_lengths = batch[4]
        
        q1_emb = self.words_embedding(q1_input)
        q2_emb = self.words_embedding(q2_input)
        q1_input = pack_padded_sequence(q1_emb, q1_lengths, batch_first=True, enforce_sorted=False)
        q2_input = pack_padded_sequence(q2_emb, q2_lengths, batch_first=True, enforce_sorted=False)
        
        q1_packed_output, _ = self.bi_lstm(q1_input)
        q2_packed_output, _ = self.bi_lstm(q2_input)
        
        q1_output, _ = pad_packed_sequence(q1_packed_output)
        q2_output,_ = pad_packed_sequence(q2_packed_output)
        q1_output = q1_output.transpose(0, 1)
        q2_output = q2_output.transpose(0, 1)
        q1_fw, q1_bw = q1_output[:, :, :self.hidden_size], q1_output[:, :, self.hidden_size:]
        q2_fw, q2_bw = q2_output[:, :, :self.hidden_size], q2_output[:, :, self.hidden_size:]
        q1_fw, q1_bw = self.dropout(q1_fw), self.dropout(q1_bw)
        q2_fw, q2_bw = self.dropout(q2_fw), self.dropout(q2_bw)
        
        max_len = q1_fw.shape[1]
        self.max_len = max_len
        actual_q1_len = q1_lengths[:, None]
        actual_q2_len = q2_lengths[:, None]
        ref_len = torch.arange(max_len)[None]
        mask1 = ref_len < actual_q1_len
        mask2 = ref_len < actual_q2_len
        mask = (mask1[:, :, None] * mask2[:, None]).to(torch.long)
        
        batch_size = q1_lengths.shape[0]
        self.batch_size = batch_size
        batch_idx = torch.arange(batch_size)
        batch_idx = batch_idx[:,None]
        actual_q2_last = q2_lengths[:, None] - 1
        q2_idx = torch.cat([batch_idx, actual_q2_last], dim=1)
        last_q2_fw = q2_fw[q2_idx[:, 0], q2_idx[:, 1]]
        last_q2_bw = q2_bw[q2_idx[:, 0], q2_idx[:, 1]]
        
        fw_full_matching = self._full_matching(q1_fw, last_q2_fw)
        bw_full_matching = self._full_matching(q1_bw, last_q2_bw)
        matches.append(fw_full_matching)
        matches.append(bw_full_matching)
        
        fw_pool_matching = self._pooling_matching(q1_fw, q2_fw)
        bw_pool_matching = self._pooling_matching(q2_bw, q2_bw)
        matches.append(fw_pool_matching)
        matches.append(bw_pool_matching)
        
        fw_cosine_similarity_attn = self._cosine_attn(q1_fw, q2_fw, mask)
        bw_cosine_similarity_attn = self._cosine_attn(q1_bw, q2_bw, mask)
        
        if self.use_multi_head:
            fw_multiplicative_attn = self._multiplicative_attn(q1_fw, q2_fw, q2_lengths, mask)
            bw_multiplicative_attn = self._multiplicative_attn(q1_bw, q2_bw, q2_lengths, mask)
            
            fw_additive_attn = self._additive_attn(q1_fw, q2_fw, q2_lengths, mask)
            bw_additive_attn = self._additive_attn(q1_bw, q2_bw, q2_lengths, mask)
        
            return fw_additive_attn, bw_additive_attn
        else:
            return None, None
    
    def _cosine_attn(self, x1, x2, mask=None):
        x1 = x1[:, :, None]
        x2 = x2[:, None]
        cosine_similarity = self._cosine_similarity(x1, x2)
        cosine_similarity = cosine_similarity * mask
        return cosine_similarity
    
    def _cosine_similarity(self, x1, x2):
        cosine_numerator = torch.sum(x1 * x2, dim=-1)
        x1_norm_l1 = torch.norm(x1, dim=-1, p=1) + 1e-6
        x2_norm_l1 = torch.norm(x2, dim=-1, p=1) + 1e-6
        cosine_denominator = x1_norm_l1 * x2_norm_l1
        cosine_similarity = cosine_numerator / cosine_denominator
        return cosine_similarity
    
    def _mul_weights(self, x, weights, dim='1D'):
        assert dim in ['1D', '2D']
        if dim == '2D':
            x = x.reshape(-1, x.shape[-1])
        x = x[:, None]
        weights = weights[None]
        mul_x = x * weights
        return mul_x
    
    def _post_attn_process(self, logits, x2_len, pad_mask):
        ref_len = torch.arange(self.max_len)[None]
        mask = (ref_len < x2_len[:, None]).to(torch.long)
        mask = mask[:, None]
        logits = mask * logits + (1 - mask) * -1e9
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        attn = torch.exp(logits)
        attn = attn * pad_mask
        attn = attn / (torch.sum(attn, dim=-1, keepdim=True) + 1e-6)
        return attn
    
    def _multiplicative_attn(self, x1, x2, x2_len, pad_mask):
        x1 = self.multiplicative_linear(x1)
        x2 = self.multiplicative_linear(x2)
        logits = torch.matmul(x1, x2.transpose(1, 2))
        attn = self._post_attn_process(logits, x2_len, pad_mask)
        return attn
    
    def _additive_attn(self, x1, x2, x2_len, pad_mask):
        x1 = self.additive_linear(x1)
        x2 = self.additive_linear(x2)
        x1 = x1[:, :, None]
        x2 = x2[:, None]
        logits = F.tanh(x1 + x2)
        logits = torch.sum(self.additive_dot_weights * logits, dim=-1)
        attn = self._post_attn_process(logits, x2_len, pad_mask)
        return attn
    
    
      
    def _full_matching(self, x1, x2):
        q1_rep = self._mul_weights(x1, self.full_matching_weights, dim='2D')
        q2_rep = self._mul_weights(x2, self.full_matching_weights, dim='1D')
        q1_rep = q1_rep.reshape(self.batch_size, -1, self.mul_dim, self.hidden_size)
        q2_rep = q2_rep[:, None]
        word_to_sent_perspective = self._cosine_similarity(q1_rep, q2_rep)
        return word_to_sent_perspective
    
    def _pooling_matching(self, x1, x2):
        q1_rep = self._mul_weights(x1, self.pool_matching_weights, dim='2D')
        q2_rep = self._mul_weights(x2, self.pool_matching_weights, dim='2D')
        q1_rep = q1_rep.reshape(self.batch_size, -1, self.mul_dim, self.hidden_size)
        q2_rep = q2_rep.reshape(self.batch_size, -1, self.mul_dim, self.hidden_size)
        q1_rep = q1_rep[:, :, None]
        q2_rep = q2_rep[:, None]
        word_to_word_perspective = self._cosine_similarity(q1_rep, q2_rep)
        word_to_word_perspective = torch.mean(word_to_word_perspective, dim=2)
        return word_to_word_perspective

#%%
model = BiMPM(emb_dim=300,
              hidden_size=150,
              words_index_dict=words_index,
              mul_dim=20,
              use_multi_head=True)
a, b = model(batch)
# q2_fw.shape

#%%
# q1_len = batch[3]
# q2_len = batch[4]
x1 = torch.rand(64, 20, 1, 100)
x2 = torch.rand(64, 1, 20, 100)
x1 = torch.tile(x1, (1,1,20,1))
x2 = torch.tile(x2, (1,20,1,1))
torch.concat([x1, x2], dim=-1).shape