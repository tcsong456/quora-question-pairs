


#%%
import numpy as np
from torch.utils.data import DataLoader
from models.utils.build_vocab import BuildVocab
# bv = BuildVocab(
#          'data/train.csv',
#          'data/test.csv'
#      )
# train = bv.train_data
ds = InfoNCEDataset(
        bv,
        np.arange(train.shape[0]),
        mode='train'
    )
collate_fn = make_collate_fn(mode='train')
dl = DataLoader(ds, batch_size=128, shuffle=True,
                collate_fn=collate_fn)
for batch in dl:
    break