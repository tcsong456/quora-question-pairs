import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class DIINConvHead(nn.Module):
    def __init__(self, in_channels, base_channels=96, dropout=0.1):
        super().__init__()
        self.pre = nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False)

        def block(ch):
            return nn.Sequential(
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            )

        self.block1a = block(base_channels)
        self.block1b = block(base_channels)
        self.pool1  = nn.MaxPool2d(2)

        self.block2a = block(base_channels)
        self.block2b = block(base_channels)
        self.pool2  = nn.MaxPool2d(2)

        self.final  = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * base_channels, 1)

    def forward(self, inter):
        x = self.pre(inter)

        x = self.block1a(x) + x
        x = self.block1b(x) + x
        x = self.pool1(x)

        x = self.block2a(x) + x
        x = self.block2b(x) + x
        x = self.pool2(x)

        x = self.final(x)
        x = self.dropout(x)

        max_pool = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
        avg_pool = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        feat = torch.cat([max_pool, avg_pool], dim=-1)

        logits = self.fc(F.relu(feat))
        return logits, feat

class DIIN(nn.Module):
    def __init__(self,
                 vocab,
                 vec_model,
                 att_layers=1,
                 char_dim=100,
                 emb_dim=300,
                 cnn_base_channels=64,
                 dense_growth_rate=16,
                 dense_layers_per_block=4,
                 cnn_dropout=0.3):
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
        self.att_layers = att_layers
        
        highway_dim = 3*50+emb_dim
        self.highway_proj = nn.Linear(highway_dim, highway_dim)
        self.highway_gate = nn.Linear(highway_dim, highway_dim)
        self.self_attn_linears = nn.ModuleList([
            nn.Linear(highway_dim*4, 1, bias=True)
            for _ in range(att_layers)
          ])
        
        self.interaction_in_channels = highway_dim * 4        
        self.head = DIINConvHead(
            in_channels=self.interaction_in_channels,
            base_channels=cnn_base_channels,
            dropout=cnn_dropout
          )

        self._init_weights()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.additive_dot_weights = nn.Parameter(torch.empty(1, 100))
        nn.init.xavier_uniform_(self.additive_dot_weights)
        self.cross_attention_w = nn.Linear(highway_dim, 100)
        self.Wq = nn.Linear(highway_dim, 64, bias=False)
        self.Wk = nn.Linear(highway_dim, 64, bias=False)
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
        
        for i in range(self.att_layers):
            q1_enc = self._self_attention(q1_emb, q1_mask, i)
            q1_emb = q1_enc + q1_emb
            q2_enc = self._self_attention(q2_emb, q2_mask, i)
            q2_emb = q2_enc + q2_emb
        
        inter = self.build_interaction(q1_enc, q2_enc)        
        logits, feat = self.head(F.relu(inter))
        if return_embedding:
            preds = F.sigmoid(logits)
            output = torch.cat([preds, feat], dim=-1)
            return logits, output
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
    
    def _self_attention(self, x, q_mask, i):
        B, L, D = x.shape
    
        linear = self.self_attn_linears[i]
        w = linear.weight.squeeze(0)
        b = linear.bias

        u1, u2, u3, u4 = torch.split(w, D, dim=0)
        w1 = u1 + u3 
        w2 = u2 - u3
        w3 = u4
    
        s1 = torch.matmul(x, w1)
        s2 = torch.matmul(x, w2)
    
        x_scaled = x * w3 
        s3 = torch.matmul(x_scaled, x.transpose(1, 2))
        logits1 = s1.unsqueeze(2) + s2.unsqueeze(1) + s3 + b
        
        q = self.Wq(x)
        k = self.Wk(x)
        logits2 = torch.matmul(q, k.transpose(1, 2))
        
        scores = torch.stack([logits1, logits2], dim=-1)
        scores = self.att_mix(scores).squeeze(dim=-1)
        
        if q_mask is not None:
            key_mask = q_mask.unsqueeze(1)
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~key_mask, mask_value)

        attn = torch.softmax(scores, dim=-1)
        out = attn @ x
        return out
