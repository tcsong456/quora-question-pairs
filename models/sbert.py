import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

class SBertEncoder(nn.Module):
    def __init__(self,
                 model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
    
    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask)
        emb = self.mean_pool(out.last_hidden_state, attention_mask)
        return emb

class SBERT(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = SBertEncoder(model_name)
        hidden_size = self.encoder.model.config.hidden_size

        combined_dim = hidden_size * 4 + 1
        self.compress_layer = nn.Linear(combined_dim, 128)
        self.final_layer = nn.Linear(128, 1)

    def forward(self,
                batch,
                return_embedding=False):
        q1_input = batch[1]
        q2_input = batch[2]
        q1_attention_mask = batch[3]
        q2_attention_mask = batch[4]
        u = self.encoder(q1_input, q1_attention_mask)
        v = self.encoder(q2_input, q2_attention_mask) 

        diff = torch.abs(u - v)
        prod = u * v
        cos = F.cosine_similarity(u, v, dim=-1, eps=1e-8).unsqueeze(-1)
        feats = torch.cat([u, v, diff, prod, cos], dim=-1)

        h = F.relu(self.compress_layer(feats))
        logits = self.final_layer(h)
        if return_embedding:
            preds = torch.sigmoid(logits)
            output = torch.cat([preds, h], dim=-1)
            return logits, output
        else:
            return logits



#%%


