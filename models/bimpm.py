import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiMPM(nn.Module):
    def __init__(self,
                 emb_dim,
                 hidden_size,
                 words_index_dict,
                 mp_dim,
                 max_len,
                 batch_size,
                 vec_model,
                 device='cpu',
                 multi_attn_head=False):
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
        self.hidden_size = hidden_size
        self.mp_dim = mp_dim
        self.multi_attn_head = multi_attn_head
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device
        
        self.contextual_bilstm = nn.LSTM(emb_dim,
                                         hidden_size,
                                         batch_first=True,
                                         bidirectional=True,
                                         num_layers=1)
        self.matching_pools_bilstm = nn.LSTM(
            8*mp_dim,
            100,
            batch_first=True,
            bidirectional=True,
            num_layers=1
          )
        self.word_emb_dropout = nn.Dropout(0.1)
        self.encoder_lstm_dropout = nn.Dropout(0.2)
        self.matching_layer_dropout = nn.Dropout(0.3)
        self.decoder_lstm_dropout = nn.Dropout(0.2)
        self.dense_rep_dropout = nn.Dropout(0.3)
        
        self.full_matching_weights = nn.Parameter(torch.empty(mp_dim, hidden_size))
        nn.init.xavier_uniform_(self.full_matching_weights)
        self.pool_matching_weights = nn.Parameter(torch.empty(mp_dim, hidden_size))
        nn.init.xavier_uniform_(self.pool_matching_weights)
        self.attentive_matching_weights = nn.Parameter(torch.empty(mp_dim, hidden_size))
        nn.init.xavier_uniform_(self.attentive_matching_weights)
        self.max_att_matching_weights = nn.Parameter(torch.empty(mp_dim, hidden_size))
        nn.init.xavier_uniform_(self.max_att_matching_weights)
        if self.multi_attn_head:
            self.multiplicative_linear = nn.Linear(hidden_size, 100)
            self.additive_linear = nn.Linear(hidden_size, 100)
            self.additive_dot_weights = nn.Parameter(torch.empty(1, 100))
            nn.init.xavier_uniform_(self.additive_dot_weights)
            self.concat_linear = nn.Linear(2*hidden_size, 100)
            self.concat_dot_weights = nn.Parameter(torch.empty(1, 100))
            nn.init.xavier_uniform_(self.concat_dot_weights)
            self.multi_attentive_head = nn.Linear(4, 1)
            self.multi_mp_compression = nn.Linear(4*mp_dim, mp_dim)
        self.agg_rep_later = nn.Linear(4*100, 100)
        self.predict_layer = nn.Linear(100, 1)
    
    def forward(self, batch):
        q1 = batch[1]
        q2 = batch[2]
        q1_len = batch[3]
        q2_len = batch[4]
        
        agg_representations = []
        match_q1_to_q2 = self.matching_sequence(q1, q2, q1_len, q2_len)
        match_q2_to_q1 = self.matching_sequence(q2, q1, q2_len, q1_len)
        q1_q2_rep = self.matching_layer_dropout(torch.cat(match_q1_to_q2, dim=-1))
        q2_q1_rep = self.matching_layer_dropout(torch.cat(match_q2_to_q1, dim=-1))
        q1_q2_fw, q1_q2_bw = self._handle_lstm(q1_q2_rep, q1_len, self.matching_pools_bilstm, self.decoder_lstm_dropout, 100)
        q2_q1_fw, q2_q1_bw = self._handle_lstm(q2_q1_rep, q2_len, self.matching_pools_bilstm, self.decoder_lstm_dropout, 100)
        q1_q2_fw_state, _ = q1_q2_fw.max(dim=1)
        q1_q2_bw_state, _ = q1_q2_bw.max(dim=1)
        q2_q1_fw_state, _ = q2_q1_fw.max(dim=1)
        q2_q1_bw_state, _ = q2_q1_bw.max(dim=1)
        agg_representations.append(q1_q2_fw_state), agg_representations.append(q1_q2_bw_state), agg_representations.append(q2_q1_fw_state),\
        agg_representations.append(q2_q1_bw_state)
        agg_rep = torch.cat(agg_representations, dim=1)
        regroup_results = self.dense_rep_dropout(F.tanh(self.agg_rep_later(agg_rep)))
        y = F.sigmoid(self.predict_layer(regroup_results))
        return y
    
    def _handle_lstm(self, x, x_len, lstm_func, dropout_func, split_size):
        x = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        x_packed_output, _ = lstm_func(x)
        x_output, _ = pad_packed_sequence(x_packed_output, batch_first=True, total_length=self.max_len)
        x_fw, x_bw = x_output[:, :, :split_size], x_output[:, :, split_size:]
        x_fw, x_bw = dropout_func(x_fw), dropout_func(x_bw)
        return x_fw, x_bw
    
    def matching_sequence(self, q1_input, q2_input, q1_lengths, q2_lengths):
        matches = []
        q1_emb = self.word_emb_dropout(self.words_embedding(q1_input))
        q2_emb = self.word_emb_dropout(self.words_embedding(q2_input))
        
        q1_fw, q1_bw = self._handle_lstm(q1_emb, q1_lengths, self.contextual_bilstm, self.encoder_lstm_dropout, self.hidden_size)
        q2_fw, q2_bw = self._handle_lstm(q2_emb, q2_lengths, self.contextual_bilstm, self.encoder_lstm_dropout, self.hidden_size)
        
        max_len = q1_fw.shape[1]
        actual_q1_len = q1_lengths[:, None]
        actual_q2_len = q2_lengths[:, None]
        ref_len = torch.arange(max_len)[None].to(self.device)
        mask1 = ref_len < actual_q1_len
        mask2 = ref_len < actual_q2_len
        mask = (mask1[:, :, None] * mask2[:, None]).to(torch.long)
        
        batch_size = q1_lengths.shape[0]
        batch_idx = torch.arange(batch_size)
        batch_idx = batch_idx[:,None].to(self.device)
        actual_q2_last = q2_lengths[:, None] - 1
        q2_idx = torch.cat([batch_idx, actual_q2_last], dim=1)
        last_q2_fw = q2_fw[q2_idx[:, 0], q2_idx[:, 1]]
        last_q2_bw = q2_bw[q2_idx[:, 0], q2_idx[:, 1]]
        
        fw_full_matching = self._full_matching(q1_fw, last_q2_fw)
        bw_full_matching = self._full_matching(q1_bw, last_q2_bw)
        matches.append(fw_full_matching)
        matches.append(bw_full_matching)
        
        fw_pool_matching = self._pooling_matching(q1_fw, q2_fw)
        bw_pool_matching = self._pooling_matching(q1_bw, q2_bw)
        matches.append(fw_pool_matching)
        matches.append(bw_pool_matching)
        
        fw_cosine_attn = self._cosine_attn(q1_fw, q2_fw, q2_lengths, mask)
        bw_cosine_attn = self._cosine_attn(q1_bw, q2_bw, q2_lengths, mask)
        fw_cosine_sim = self._weight_sent_by_attn(q2_fw, fw_cosine_attn)
        bw_cosine_sim = self._weight_sent_by_attn(q2_bw, bw_cosine_attn)
        fw_cosine_matching = self._attentive_matching(q1_fw, fw_cosine_sim, self.attentive_matching_weights)
        bw_cosine_matching = self._attentive_matching(q1_bw, bw_cosine_sim, self.attentive_matching_weights)
        fw_cosine_max_att_matching = self._max_attentive_matching(q1_fw, q2_fw, fw_cosine_attn)
        bw_cosine_max_att_matching = self._max_attentive_matching(q1_bw, q2_bw, bw_cosine_attn)
        
        if self.multi_attn_head:
            fw_multiplicative_attn = self._multiplicative_attn(q1_fw, q2_fw, q2_lengths, mask)
            bw_multiplicative_attn = self._multiplicative_attn(q1_bw, q2_bw, q2_lengths, mask)
            fw_multiplicative_sim = self._weight_sent_by_attn(q2_fw, fw_multiplicative_attn)
            bw_multiplicative_sim = self._weight_sent_by_attn(q2_bw, bw_multiplicative_attn)
            fw_multiplicative_matching = self._attentive_matching(q1_fw, fw_multiplicative_sim, self.attentive_matching_weights)
            bw_multiplicative_matching = self._attentive_matching(q1_bw, bw_multiplicative_sim, self.attentive_matching_weights)
            
            fw_additive_attn = self._additive_attn(q1_fw, q2_fw, q2_lengths, mask)
            bw_additive_attn = self._additive_attn(q1_bw, q2_bw, q2_lengths, mask)
            fw_additive_sim = self._weight_sent_by_attn(q2_fw, fw_additive_attn)
            bw_additive_sim = self._weight_sent_by_attn(q2_bw, bw_additive_attn)
            fw_additive_matching = self._attentive_matching(q1_fw, fw_additive_sim, self.attentive_matching_weights)
            bw_additive_matching = self._attentive_matching(q1_bw, bw_additive_sim, self.attentive_matching_weights)
            
            fw_concat_attn = self._concat_attn(q1_fw, q2_fw, q2_lengths, mask)
            bw_concat_attn = self._concat_attn(q1_bw, q2_bw, q2_lengths, mask)
            fw_concat_sim = self._weight_sent_by_attn(q2_fw, fw_concat_attn)
            bw_concat_sim = self._weight_sent_by_attn(q2_bw, bw_concat_attn)
            fw_concat_matching = self._attentive_matching(q1_fw, fw_concat_sim, self.attentive_matching_weights)
            bw_concat_matching = self._attentive_matching(q1_bw, bw_concat_sim, self.attentive_matching_weights)
            
            # fw_attentive_matching = torch.stack([fw_cosine_matching, fw_multiplicative_matching,
            #                                      fw_additive_matching, fw_concat_matching], dim=-1)
            # bw_attentive_matching = torch.stack([bw_cosine_matching, bw_amultiplicative_matching,
            #                                      bw_aadditive_matching, bw_concat_matching], dim=-1)
            # fw_attentive_matching = self.multi_attentive_head(fw_attentive_matching).squeeze(dim=-1)
            # bw_attentive_matching = self.multi_attentive_head(bw_attentive_matching).squeeze(dim=-1)
            
            fw_attentive_matching = torch.cat([fw_cosine_matching, fw_multiplicative_matching,
                                                 fw_additive_matching, fw_concat_matching], dim=-1)
            bw_attentive_matching = torch.cat([bw_cosine_matching, bw_multiplicative_matching,
                                                 bw_additive_matching, bw_concat_matching], dim=-1)
            fw_attentive_matching = self.multi_mp_compression(fw_attentive_matching)
            bw_attentive_matching = self.multi_mp_compression(bw_attentive_matching)
            
            fw_max_attn = torch.stack([fw_cosine_attn, fw_multiplicative_attn,
                fw_additive_attn,fw_concat_attn], dim=-1)
            bw_max_attn = torch.stack([bw_cosine_attn, bw_multiplicative_attn,
                bw_additive_attn, bw_concat_attn], dim=-1)
            fw_max_att_score = self.multi_attentive_head(fw_max_attn).squeeze(dim=-1)
            bw_max_att_score = self.multi_attentive_head(bw_max_attn).squeeze(dim=-1)
            fw_max_att_score = fw_max_att_score.masked_fill(~mask.bool(), -1e9)
            bw_max_att_score = bw_max_att_score.masked_fill(~mask.bool(), -1e9)
            fw_max_att = torch.softmax(fw_max_att_score, dim=-1)
            bw_max_att = torch.softmax(bw_max_att_score, dim=-1)
            fw_max_att_matching = self._max_attentive_matching(q1_fw, q2_fw, fw_max_att)
            bw_max_att_matching = self._max_attentive_matching(q1_bw, q2_bw, bw_max_att)
        else:
            fw_attentive_matching = fw_cosine_matching
            bw_attentive_matching = bw_cosine_matching
            fw_max_att_matching = fw_cosine_max_att_matching
            bw_max_att_matching = bw_cosine_max_att_matching
        
        matches.append(fw_attentive_matching), matches.append(bw_attentive_matching)
        matches.append(fw_max_att_matching), matches.append(bw_max_att_matching)
        return matches
    
    def _cosine_attn(self, x1, x2, x2_len, pad_mask):
        x1 = x1[:, :, None]
        x2 = x2[:, None]
        logits = self._cosine_similarity(x1, x2)
        attn = self._post_attn_process(logits, x2_len, pad_mask)
        return attn
    
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
        ref_len = torch.arange(self.max_len)[None].to(self.device)
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
    
    def _concat_attn(self, x1, x2, x2_len, pad_mask):
        x1 = torch.tile(x1[:, :, None], (1, 1, self.max_len, 1))
        x2 = torch.tile(x2[:, None], (1, self.max_len, 1, 1))
        x = torch.concat([x1 ,x2], dim=-1)
        x = self.concat_linear(x)
        logits = torch.sum(x * self.concat_dot_weights, dim=-1)
        attn = self._post_attn_process(logits, x2_len, pad_mask)
        return attn
    
    def _weight_sent_by_attn(self, x, attn_mat):
        x = x[:, None]
        attn_mat = attn_mat[:, :, :, None]
        x = torch.sum(x * attn_mat, dim=2)
        return x
    
    def _full_matching(self, x1, x2):
        q1_rep = self._mul_weights(x1, self.full_matching_weights, dim='2D')
        q2_rep = self._mul_weights(x2, self.full_matching_weights, dim='1D')
        q1_rep = q1_rep.reshape(self.batch_size, -1, self.mp_dim, self.hidden_size)
        q2_rep = q2_rep[:, None]
        word_to_sent_perspective = self._cosine_similarity(q1_rep, q2_rep)
        return word_to_sent_perspective
    
    def _pooling_matching(self, x1, x2):
        q1_rep = self._mul_weights(x1, self.pool_matching_weights, dim='2D')
        q2_rep = self._mul_weights(x2, self.pool_matching_weights, dim='2D')
        q1_rep = q1_rep.reshape(self.batch_size, -1, self.mp_dim, self.hidden_size)
        q2_rep = q2_rep.reshape(self.batch_size, -1, self.mp_dim, self.hidden_size)
        q1_rep = q1_rep[:, :, None]
        q2_rep = q2_rep[:, None]
        word_to_word_perspective = self._cosine_similarity(q1_rep, q2_rep)
        word_to_word_perspective = torch.mean(word_to_word_perspective, dim=2)
        return word_to_word_perspective
    
    def _attentive_matching(self, x, m, attn_weights):
        x = self._mul_weights(x, attn_weights, dim='2D')
        m = self._mul_weights(m, attn_weights, dim='2D')
        word_to_word_weighted_perpective = self._cosine_similarity(x, m)
        word_to_word_weighted_perpective = word_to_word_weighted_perpective.reshape(self.batch_size, -1, self.mp_dim)
        return word_to_word_weighted_perpective
    
    def _max_attentive_matching(self, x1, x2, m):
        idx = m.argmax(dim=-1)
        batch_idx = torch.arange(idx.shape[0])[:, None].expand(-1, self.max_len)
        x2 = x2[batch_idx, idx, :]
        word_to_word_max_perspective = self._attentive_matching(x1, x2, self.max_att_matching_weights)
        return word_to_word_max_perspective
        

#%%

# bv = BuildVocab('data/train.csv',
#                 'data/test.csv')
# words_index = bv.load()

# train = pd.read_csv('data/train.csv')
# dataset = QQPDataset(
#     data=train,
#     words_index=words_index,
#     max_len=40,
#     mode='train'
#   )
# dl = DataLoader(dataset,
#                 shuffle=True,
#                 batch_size=128
#                 )
# for batch in dl:
#     break
  
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# for i, v in enumerate(batch):
#     if isinstance(v, torch.Tensor):
#         batch[i] = v.to(device)

# vec_model = load_facebook_model('artifacts/cc.en.300.bin')

# model = BiMPM(emb_dim=300,
#               hidden_size=150,
#               batch_size=128,
#               max_len=40,
#               words_index_dict=words_index,
#               mp_dim=20,
#               vec_model=vec_model,
#               device=device,
#               multi_attn_head=False).to(device)
# a = model(batch)
# q2_fw.shape

#%%




