import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SelfDesignV1(nn.Module):
    def __init__(self,
                 emb_dim,
                 hidden_size,
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
        self.bilsm = nn.LSTM(
                emb_dim,
                hidden_size,
                batch_first=True,
                bidirectional=True,
                num_layers=1
            )
        self.wp_lstm = nn.LSTM(
                4*hidden_size,
                hidden_size,
                batch_first=True,
                bidirectional=False,
                num_layers=1
            )
        self.wp_fc = nn.Linear(hidden_size, hidden_size//2)
        self.wp_out = nn.Linear(hidden_size//2, 1)
        self.cross_linear = nn.Linear(2*hidden_size, 2*hidden_size)
        self.diff_fc1 = nn.Linear(4*hidden_size+1, 128)
        self.diff_fc2 = nn.Linear(128, 16)
        self.cnn1 = nn.Conv2d(1, 3, kernel_size=3, padding=0)
        self.cnn1_bn = nn.BatchNorm2d(3)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.cnn2 = nn.Conv2d(3, 3, kernel_size=3, padding=0)
        self.cnn2_bn = nn.BatchNorm2d(3)
        self.pool2 = nn.MaxPool2d(kernel_size=3)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, max_len, max_len)
            z = self.pool1(self.cnn1_bn(self.cnn1(dummy)))
            z = self.pool2(self.cnn2_bn(self.cnn2(z)))
            flat_dim = z.numel()

        self.cnn_fc = nn.Linear(flat_dim, 10)
        self.cnn_fc_bn = nn.BatchNorm1d(10)
        self.cnn_out = nn.Linear(10, 1)
        self.fc_main = nn.Linear(16, 1)
        # self.final_layer = nn.Linear(27, 1)
 
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
         
            self.cross_dropout = nn.Dropout(0.1)
            self.enc_dropout = nn.Dropout(0.1)
            self.enc1_dropout = nn.Dropout(0.05)
            self.diff_dropout = nn.Dropout(0.15)
            
        self.max_len = max_len
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _mask(self, q_len):
        ref_len = torch.arange(self.max_len).to(self.device)
        q_mask = ref_len[None] < q_len[:, None]
        q_mask = q_mask.to(torch.long)
        return q_mask
    
    def _handle_lstm(self, x, x_len, lstm_func, dropout_func):
        x = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        x_packed_output, _ = lstm_func(x)
        x_output, _ = pad_packed_sequence(x_packed_output, batch_first=True, total_length=self.max_len)
        x_output = dropout_func(x_output)
        return x_output
    
    def _diff_mask(self, q1, q2):
        q1_mask_diff = torch.zeros(q1.shape[0], self.max_len, dtype=torch.long)
        q2_mask_diff = torch.zeros(q2.shape[0], self.max_len, dtype=torch.long)
        for i, (row1, row2) in enumerate(zip(q1.cpu().numpy(), q2.cpu().numpy())):
            row11 = set(row1)
            row22 = set(row2)
            unique_q1 = list(set(row11) - set(row22))
            unique_q2 = list(set(row22) - set(row11))
            for k in unique_q1:
                idx = np.where(row1==k)[0]
                q1_mask_diff[i, idx] = 1
            for k in unique_q2:
                idx = np.where(row2==k)[0]
                q2_mask_diff[i, idx] = 1
        return q1_mask_diff.to(self.device), q2_mask_diff.to(self.device)
    
    def _cross_attn(self, q1, q2, q1_mask, q2_mask):
        q1 = self.cross_linear(q1)
        q2 = self.cross_linear(q2)
        logits = q1 @ q2.transpose(1,2)
        logits = q2_mask[:, None] * logits + (1 - q2_mask[:, None]) * -1e9
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        attn = torch.exp(logits)
        attn = attn / (torch.sum(attn, dim=-1, keepdim=True) + 1e-6)
        attn = attn * q1_mask[:, :, None]
        return attn
    
    # def _diff_head(self, x1, x2):
    #     x = torch.cat([x1, x2], dim=1)
    #     h = F.relu(self.diff_fc1(x))
    #     h = self.diff_dropout(h)
    #     diff1 = torch.tanh(self.diff_fc2(h))
    #     diff2 = F.cosine_similarity(x1, x2, dim=-1).unsqueeze(-1)
    #     return [diff1, diff2]
    
    def _diff_head(self, x1, x2):
        x1n = F.normalize(x1, dim=-1, eps=1e-8)
        x2n = F.normalize(x2, dim=-1, eps=1e-8)

        absdiff = torch.abs(x1n - x2n)
        prod    = x1n * x2n
        cos     = (x1n * x2n).sum(dim=-1, keepdim=True)
        feats = torch.cat([absdiff, prod, cos], dim=-1)
        h = F.relu(self.diff_fc1(feats))
        h = self.diff_dropout(h)
        diff = self.diff_fc2(h)
        return [diff]
        
    def _mean_pool(self, q, mask):
        mean_q = (q * mask[:, :, None]).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-4)
        return mean_q
    
    def _max_pool(self, q, mask):
        mask = mask[:, :, None]
        q = q * mask
        mask = (1 - mask) * -1e9
        q_max = (q + mask).max(dim=1)[0]
        return q_max
    
    def _att_pool(self, att_matrix, q1_mask, q2_mask):
        att_matrix = att_matrix * q2_mask.unsqueeze(1) + (1 - q2_mask.unsqueeze(1)) * -1e9
        att_matrix = torch.softmax(att_matrix, dim=1)
        att_matrix = att_matrix.max(dim=-1)[0]
        f = (att_matrix * q1_mask).sum(dim=1) / q1_mask.sum(dim=1)
        return f
    
    def forward(self, batch, return_embedding=False):
        q1 = batch[1]
        q2 = batch[2]
        q1_len = batch[3]
        q2_len = batch[4]
        q1_mask = self._mask(q1_len)
        q2_mask = self._mask(q2_len)
        q1_mask_diff, q2_mask_diff = self._diff_mask(q1, q2)
        
        q1_emb = self.words_embedding(q1)
        q2_emb = self.words_embedding(q2)
        
        x1 = self._handle_lstm(q1_emb, q1_len.cpu(), self.bilsm, self.enc_dropout)
        x2 = self._handle_lstm(q2_emb, q2_len.cpu(), self.bilsm, self.enc_dropout)
        q1_raw = self._mean_pool(x1, q1_mask)
        q2_raw = self._mean_pool(x2, q2_mask)
        q1_atted = self.enc_dropout(self._cross_attn(x1, x2, q1_mask, q2_mask) @ q2_emb)
        q2_atted = self.enc_dropout(self._cross_attn(x2, x1, q2_mask, q1_mask) @ q1_emb)
        q1_diff = self._mean_pool(x1, q1_mask_diff)
        q2_diff = self._mean_pool(x2, q2_mask_diff)
        q1_mean_atted = self._mean_pool(q1_atted, q1_mask)
        q2_mean_atted = self._mean_pool(q2_atted, q2_mask)
        
        diffs = []
        avg_sent_diff = self._diff_head(q1_raw, q2_raw)
        diffs += avg_sent_diff
        # max_atted_diff = self._diff_head(self._max_pool(q1_atted, q1_mask), self._max_pool(q2_atted, q2_mask))
        # diffs += max_atted_diff
        # mean_diff_sent = self._diff_head(q1_diff, q2_diff)
        # diffs += mean_diff_sent
        # mean_atted_diff = self._diff_head(q1_mean_atted, q2_mean_atted)
        # diffs += mean_atted_diff
        # raw_mean_atted_1 = self._diff_head(q1_raw, q1_mean_atted)
        # diffs += raw_mean_atted_1
        # raw_mean_atted_2 = self._diff_head(q2_raw, q2_mean_atted)
        # diffs += raw_mean_atted_2
        # raw_diff_mean_1 = self._diff_head(q1_raw, q2_diff)
        # diffs += raw_diff_mean_1
        # raw_diff_mean_2 = self._diff_head(q2_raw, q1_diff)
        # diffs += raw_diff_mean_2
        
        # concated_q1 = torch.cat([q1_atted, x1], dim=-1)
        # concated_q2 = torch.cat([q2_atted, x2], dim=-1)
        # wp1 = self._handle_lstm(concated_q1, q1_len.cpu(), self.wp_lstm, self.enc_dropout)
        # wp2 = self._handle_lstm(concated_q2, q2_len.cpu(), self.wp_lstm, self.enc_dropout)
        # h1 = self.enc1_dropout(F.relu(self.wp_fc(wp1)))
        # h2 = self.enc1_dropout(F.relu(self.wp_fc(wp2)))
        # tok1 = torch.tanh(self.wp_out(h1))
        # tok2 = torch.tanh(self.wp_out(h2))
        # q1_pred = self._mean_pool(tok1, q1_mask)
        # q2_pred = self._mean_pool(tok2, q2_mask)
        
        # att_matrix = self.cross_linear(x2) @ self.cross_linear(x1).transpose(1, 2)
        # x = torch.tanh(att_matrix.unsqueeze(1))
        # x = self.cnn1_bn(self.cnn1(x))
        # x = F.dropout(x, p=0.3, training=self.training)
        # x = self.pool1(x)
        
        # x = self.cnn2_bn(self.cnn2(x))
        # x = F.dropout(x, p=0.3, training=self.training)
        # x = self.pool2(x)
        
        # x = x.flatten(1)
        # x = F.relu(self.cnn_fc_bn(self.cnn_fc(x)))
        # x = F.dropout(x, p=0.4, training=self.training)
        # cnn_pred = torch.tanh(self.cnn_out(x))
        
        # att_matrix_t = self.cross_linear(x1) @ self.cross_linear(x2).transpose(1, 2)
        # att_q1_q2 = self._att_pool(att_matrix_t, q1_mask, q2_mask)
        # att_q2_q1 = self._att_pool(att_matrix, q2_mask, q1_mask)
        # diffs += [att_q1_q2.unsqueeze(1), att_q2_q1.unsqueeze(1)]
        
        concated = torch.cat(diffs, dim=-1)
        concated = F.dropout(concated, p=0.05, training=self.training)
        logits = F.relu(self.fc_main(concated))
        h = logits
        # h = [h] + [q1_pred, q2_pred, cnn_pred]
        # h = torch.cat(h, dim=1)
        # logits = self.final_layer(h)
        if return_embedding:
            preds = torch.sigmoid(logits)
            output = torch.cat([preds, h], dim=-1)
            return logits, output
        else:
            return logits


#%%
# from gensim.models.fasttext import load_facebook_model
# vec_model = load_facebook_model('artifacts/cc.en.300.bin')
# bv = BuildVocab('data/train.csv',
#                 'data/test.csv')
# words_index_dict = bv.load_dict()

# for i, v in enumerate(batch):
#     if isinstance(v, torch.Tensor):
#         batch[i] = v.cuda()

# model = SelfDesignV1(
#         emb_dim=300,
#         hidden_size=150,
#         words_index_dict=words_index_dict,
#         max_len=40,
#         vec_model=vec_model
#     ).cuda()
# a, b = model(batch, return_embedding=True)
#%%

