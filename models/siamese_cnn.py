import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class SiameseCNN(nn.Module):
    def __init__(self,
                 vocab,
                 vec_model,
                 emb_dim=300,
                 out_channels=64,
                 kernel_sizes=[2]):
        super().__init__()
        word_to_idx = vocab.load_dict()
        vocab_size = len(word_to_idx)
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
        self.word_embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
                nn.Conv1d(in_channels=emb_dim,
                          out_channels=out_channels,
                          kernel_size=k,
                          padding=0)
                for k in kernel_sizes
            ])
        self.proj = nn.Sequential(
            nn.Linear(out_channels * len(kernel_sizes), 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.emb_dropout = nn.Dropout(0.1)
        self.linear_dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(4*256, 128)
        self.ln = nn.LayerNorm(128)
        self.gelu = nn.GELU()
        self.final_layer = nn.Linear(128, 1)
        
        self.max_len = 40
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def _padded_mask(self, q_len):
        ref_len = torch.arange(self.max_len).to(self.device)
        mask = ref_len < q_len[:, None]
        return mask
    
    def pair_features(self, e1, e2):
        diff = torch.abs(e1 - e2)
        prod = e1 * e2
        return torch.cat([e1, e2, diff, prod], dim=1)
    
    def forward(self, batch, return_embedding=False):
        q1 = batch[1]
        q2 = batch[2]
        q1_len = batch[3]
        q2_len = batch[4]
        q1_mask = self._padded_mask(q1_len)
        q2_mask = self._padded_mask(q2_len)
        
        q1_emb = self.emb_dropout(self.word_embedding(q1))
        q2_emb = self.emb_dropout(self.word_embedding(q2))
        x1 = q1_emb.transpose(1, 2)
        x2 = q2_emb.transpose(1, 2)
        x1 = q1_mask.unsqueeze(1).to(x1.dtype) * x1
        x2 = q2_mask.unsqueeze(1).to(x2.dtype) * x2
        
        pooled_1 = []
        for conv in self.convs:
            h = F.relu(conv(x1))
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)
            pooled_1.append(h)
        x1_cnn = torch.cat(pooled_1, dim=-1)
        
        pooled_2 = []
        for conv in self.convs:
            h = F.relu(conv(x2))
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)
            pooled_2.append(h)
        x2_cnn = torch.cat(pooled_2, dim=-1)
        
        q1_cnn = self.proj(x1_cnn)
        q2_cnn = self.proj(x2_cnn)
        pf = self.pair_features(q1_cnn, q2_cnn)
        h = self.fc(pf)
        h = self.linear_dropout(self.gelu(self.ln(h)))
        logit = self.final_layer(h)
        if return_embedding:
            preds = F.sigmoid(logit)
            output = torch.cat([preds, h], dim=-1)
            return logit, output
        else:
            return logit

