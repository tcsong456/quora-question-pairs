import torch
import numpy as np
from torch import nn
from gensim.models.fasttext import load_facebook_model

class BiMPM(nn.Module):
    def __init__(self,
                 emb_dim,
                 words_index_dict):
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

#%%
# model = BiMPM(300,
#               words_index)
# model.words_embedding(torch.tensor([100]))
