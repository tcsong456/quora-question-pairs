import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class Dense_net_block(nn.Module):
    def __init__(self, outChannels, growth_rate, kernel_size):
        super(Dense_net_block, self).__init__()
        self.bn = nn.BatchNorm2d(outChannels)
        self.conv = nn.Conv2d(outChannels, growth_rate, kernel_size=kernel_size, bias=False, padding=1)

    def forward(self, x):
        x = self.bn(x)
        ft = F.relu(self.conv(x))
        out = torch.cat((x, ft), dim=1)
        return out

# class Dense_net_block_no_bn(nn.Module):
#     def __init__(self, outChannels, growth_rate, kernel_size):
#         super().__init__()
#         self.conv = nn.Conv2d(outChannels, growth_rate, kernel_size=kernel_size, bias=False, padding=1)

#     def forward(self, x):
#         ft = F.relu(self.conv(x))
#         out = torch.cat((x, ft), dim=1)
#         return out

class Dense_net_transition(nn.Module):
    def __init__(self, nChannels, outChannels):
        super(Dense_net_transition, self).__init__()
        self.bn = nn.BatchNorm2d(nChannels)
        self.conv = nn.Conv2d(nChannels, outChannels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn(x))
        out = self.conv(x)
        # out = F.max_pool2d(out, (2,2), (2,2), padding=0)
        out = self.pool(out)
        return out
  
# class Dense_net_transition_no_bn(nn.Module):
#     def __init__(self, nChannels, outChannels):
#         super().__init__()
#         self.conv = nn.Conv2d(nChannels, outChannels, kernel_size=1, bias=False)

#     def forward(self, x):
#         out = self.conv(x)
#         out = F.max_pool2d(out, (2,2), (2,2), padding=0)
#         return out

class DenseNet(nn.Module):
    def __init__(self, nChannels, growthRate, reduction, nDenseBlocks, kernel_size):
        super(DenseNet, self).__init__()
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, kernel_size)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Dense_net_transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, kernel_size)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Dense_net_transition(nChannels, nOutChannels)
        # nChannels = nOutChannels
       
        # self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, kernel_size)
        # nChannels += nDenseBlocks*growthRate
        # nOutChannels = int(math.floor(nChannels*reduction))
        # self.trans3 = Dense_net_transition(nChannels, nOutChannels)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, kernel_size, block_cls=Dense_net_block):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(block_cls(nChannels, growthRate, kernel_size))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.trans1(self.dense1(x))
        out = self.trans2(self.dense2(out))
        # out = self.trans3(self.dense3(out))
        return out

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
        self.char_embedding = nn.Embedding(char_vocab_size, char_dim,
                                           padding_idx=0)
        self.char_dim = char_dim
        self.highway_layers = highway_layers
        self.self_attn_layers = self_attn_layers
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.word_emb_dropout = nn.Dropout(0.05)
        self.char_emb_dropout = nn.Dropout(0.05)
        self.char_convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_dim,
                      out_channels=60,
                      kernel_size=kernel_size)
            for kernel_size in [3, 4, 5]
          ])
        for conv in self.char_convs:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)

        highway_dim = 60*3 + emb_dim
        self.highway_proj = nn.Linear(highway_dim, highway_dim)
        self.highway_gate = nn.Linear(highway_dim, highway_dim)
        self.self_attn_linears = nn.ModuleList([
            nn.Linear(highway_dim*3, 1, bias=True)
            for _ in range(self_attn_layers)
          ])
        self.gating_q1 = nn.ModuleList([
            nn.Linear(highway_dim, highway_dim)
            for _ in range(4)
          ])
        self.gating_q2 = nn.ModuleList([
            nn.Linear(highway_dim, highway_dim)
            for _ in range(4)
          ])
        self._self_attention = TransformerSentEncoder(
            emb_dim=highway_dim,
            num_layers=2,
            num_heads=6,
            model_dim=512,
            dropout=0.1,
            max_len=40
        )
        
        self.cnn_final_conv = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bi_self_att_dropout = nn.Dropout(0.075)
        self.interaction_conv = nn.Conv2d(highway_dim*4, 64, kernel_size=1, padding=0)
        self.dense_net = DenseNet(nChannels=64, growthRate=16, reduction=0.5, nDenseBlocks=4, kernel_size=3)
        # self.prediction_layer = nn.Linear(64*10*10, 1)
        self.cnn_dropout = nn.Dropout(0.1)
        self.cnn_final_norm = nn.BatchNorm2d(64)
        classifier_in_dim = 2 * 64
        self.fc1 = nn.Linear(classifier_in_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.classifier_dropout = nn.Dropout(0.1)
    
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
        for _ in range(self.highway_layers):
            q1_emb = self._emb_highway(q1_emb)
            q2_emb = self._emb_highway(q2_emb)
        
        # for i in range(self.self_attn_layers):
        #     self_att_q1 = self._self_attention(q1_emb, q1_mask, i)
        #     q1_emb = self_att_q1 + q1_emb
        #     self_att_q2 = self._self_attention(q2_emb, q2_mask, i)
        #     q2_emb = self_att_q2 + q2_emb
        self_att_q1 = self._self_attention(q1_emb, q1_mask)
        self_att_q2 = self._self_attention(q2_emb, q2_mask)
        
        # bi_att_mx = self_att_q1.unsqueeze(2) * self_att_q2.unsqueeze(1)
        bi_att_mx = self.build_interaction(self_att_q1, self_att_q2)
        bi_att_mx = self.bi_self_att_dropout(bi_att_mx) #.permute(0, 3, 1, 2).contiguous()
        interaction_feat_map = F.relu(self.interaction_conv(bi_att_mx))
        output = self.dense_net(interaction_feat_map)
        output = self.cnn_final_conv(F.relu(self.cnn_final_norm(output)))
        # print(output.shape)
        # output = output.reshape(output.shape[0], -1)
        # logits = self.prediction_layer(output)
        x = self.cnn_final_conv(output)
        x = self.cnn_dropout(x)
        max_pool = F.adaptive_max_pool2d(x, output_size=1).squeeze(-1).squeeze(-1)
        avg_pool = F.adaptive_avg_pool2d(x, output_size=1).squeeze(-1).squeeze(-1)
        feat = torch.cat([max_pool, avg_pool], dim=-1)
        h = self.classifier_dropout(F.relu(self.fc1(feat)))
        logits = self.fc2(h)
        
        if return_embedding:
            return logits, F.sigmoid(logits)
        else:
            return logits
    
    def build_interaction(self, q1, q2) -> torch.Tensor:
        B, L, D = q1.size()

        q1_t = q1.transpose(1, 2)
        q2_t = q2.transpose(1, 2)

        q1_exp = q1_t.unsqueeze(3)
        q2_exp = q2_t.unsqueeze(2)

        q1_exp = q1_exp.expand(-1, -1, L, L)
        q2_exp = q2_exp.expand(-1, -1, L, L)
    
        diff = q1_exp - q2_exp
        prod = q1_exp * q2_exp

        interaction = torch.cat([q1_exp, q2_exp, diff, prod], dim=1)
        return interaction
    
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
    
    def _emb_highway(self, x):
        transform = F.relu(self.highway_proj(x))
        gate = torch.sigmoid(self.highway_gate(x))
        h = gate * transform + (1 - gate) * x
        return h
    
    def _padded_mask(self, q_len, max_len):
        ref_len = torch.arange(max_len).to(self.device)
        mask = ref_len < q_len[:, None]
        return mask
    
    def _gating_layer(self, x, x_att, mode='q1'):
        assert mode in ['q1', 'q2']
        if mode == 'q1':
            gating_layer = self.gating_q1
        else:
            gating_layer = self.gating_q2
        
        xh = gating_layer[0](x)
        xc = gating_layer[1](x_att)
        z = torch.tanh(xh + xc)
        
        xh2 = gating_layer[2](x)
        xc2 = gating_layer[3](x_att)
        f = torch.sigmoid(xh2 + xc2)
        out = f * x + (1 - f) * z
        return out
    
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


#%%
# import os
# from tqdm import tqdm
# from models.utils.build_vocab import BuildVocab
# from gensim.models.fasttext import load_facebook_model
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
#     emb_dim=300,
#     char_dim=100,
#     hidden_dim=100,
#     highway_layers=2,
#     self_attn_layers=4,
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
# a