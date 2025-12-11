import torch
from torch import nn
from transformers import AutoModel
from torch.nn import functional as F

class DeBertaV3(nn.Module):
    def __init__(self,
                 model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.compress_layer = nn.Linear(hidden_size, 128)
        self.final_layer = nn.Linear(128, 1)
    
    def forward(self, batch, return_embedding=False):
        q_ids = batch[1]
        att_mask = batch[2]
        outputs = self.encoder(
            input_ids=q_ids,
            attention_mask=att_mask
          )
        cls_rep = outputs.last_hidden_state[:, 0, :]
        h = F.relu(self.compress_layer(cls_rep))
        logits = self.final_layer(h)
        if return_embedding:
            preds = torch.sigmoid(logits)
            output = torch.cat([preds, h], dim=-1)
            return logits, output
        else:
            return logits
            


