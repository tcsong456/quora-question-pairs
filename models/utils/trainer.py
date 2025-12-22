import os
import torch
import argparse
import warnings
warnings.filterwarnings(action='ignore')
from transformers import logging
logging.set_verbosity_error()
import numpy as np
from torch import nn
from tqdm import tqdm
from torch import optim
from models.bimpm import BiMPM
from models.diin import DIIN
from models.esim import ESIM
from models.sbert import SBERT
from models.deberta import DeBertaV3
from models.self_design_1 import SelfDesignV1
from models.utils.build_vocab import BuildVocab
from models.utils.dataset import QQPDataset, SBERTDataset, DeBERTaV3Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from gensim.models.fasttext import load_facebook_model
from transformers import get_linear_schedule_with_warmup

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.cnt = 0
        self.sum = 0
        self.average = 0
    
    def update(self, value, n):
        self.cnt += n
        self.sum += value * n
        self.average = self.sum / max(self.cnt, 1)

class Trainer:
    def __init__(self,
                 vocab,
                 vec_model=None,
                 batch_size=32,
                 test_batch_size=20,
                 epochs=50,
                 amp=True,
                 model_name='bimpm',
                 early_stopping=1):
        train = vocab.train_data
        test = vocab.test_data
        words_index_dict = vocab.load_dict()
        self.words_index_dict = words_index_dict
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.amp = amp
        self.epochs = epochs
        self.vocab = vocab
        self.suffix = ''
        self.collate_fn = None
        
        self.data_train, self.data_val = [], []
        self.models, self.optimizers = [], []
        y = train['is_duplicate'].values
        x = train.drop('is_duplicate', axis=1)
        skf = StratifiedKFold(n_splits=5, random_state=7610, shuffle=True)
        for train_idx, val_idx in skf.split(x, y):            
            self.data_train.append(train_idx)
            self.data_val.append(val_idx)
            if model_name == 'bimpm':
                assert vec_model is not None 
                model = BiMPM(
                              emb_dim=300,
                              hidden_size=150,
                              max_len=40,
                              words_index_dict=words_index_dict,
                              mp_dim=20,
                              vec_model=vec_model,
                              device=device,
                              multi_attn_head=True
                            ).to(device)
                self.suffix = '_multi_head'
            elif model_name == 'diin':
                assert vec_model is not None 
                model = DIIN(
                      vocab=bv,
                      vec_model=vec_model,
                      att_layers=2,
                      char_dim=100,
                      emb_dim=300,
                      cnn_base_channels=96,
                      cnn_dropout=0.1
                  ).to(device)
            elif model_name == 'esim':
                assert vec_model is not None 
                model = ESIM(
                    vocab=bv,
                    vec_model=vec_model,
                    emb_dim=300,
                    char_dim=100,
                    hidden_dim=200
                  ).to(device)
            elif model_name == 'self_1':
                model = SelfDesignV1(
                        emb_dim=300,
                        hidden_size=150,
                        words_index_dict=words_index_dict,
                        max_len=40,
                        vec_model=vec_model
                    ).to(device)
            elif model_name == 'sbert':
                model = SBERT(
                    model_name="sentence-transformers/all-mpnet-base-v2"
                  ).to(device)
            elif model_name == 'deberta':
                model = DeBertaV3(
                      model_name="microsoft/deberta-v3-base"
                    ).to(device)
            
            if model_name not in ['sbert', 'deberta']:
                optimizer = optim.Adam(model.parameters(),
                                        lr=0.002)
                dataset = QQPDataset
            else:
                encoder_lr, head_lr = 2e-5, 5e-4
                no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
                encoder_decay, encoder_nodecay = [], []
                head_decay, head_nodecay = [], []
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    in_encoder = name.startswith("encoder")
                    use_nodecay = any(nd in name for nd in no_decay)
                    if in_encoder:
                        if use_nodecay:
                            encoder_nodecay.append(param)
                        else:
                            encoder_decay.append(param)
                    else:
                        if use_nodecay:
                            head_nodecay.append(param)
                        else:
                            head_decay.append(param)
                weight_decay = 0.01
                optimizer = optim.AdamW(
                      [
                        {"params": encoder_decay,   "lr": encoder_lr, "weight_decay": weight_decay},
                        {"params": encoder_nodecay, "lr": encoder_lr, "weight_decay": 0.0},
                        {"params": head_decay,      "lr": head_lr,    "weight_decay": weight_decay},
                        {"params": head_nodecay,    "lr": head_lr,    "weight_decay": 0.0},
                    ]
                    )
                if model_name == 'deberta':
                    dataset = DeBERTaV3Dataset
                elif model_name == 'sbert':
                    dataset = SBERTDataset
                    
            self.optimizers.append(optimizer)
            self.models.append(model)
            self.dataset = dataset
            self.test_batch_size = test_batch_size

        os.makedirs('checkpoints', exist_ok=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.early_stopping = early_stopping
        self.model_name = model_name
        self.test = test
        self.train_data = train
        self.batch_size = batch_size
    
    def train(self, fold, warm_start=False):
        model = self.models[fold]
        optimizer = self.optimizers[fold]
        checkpoint_path = f'checkpoints/{self.model_name}_{fold}{self.suffix}.pth'
        best_loss = np.inf
        bad_epoch = 0
        start_epoch = 0
        if warm_start:
            ckpt = torch.load(checkpoint_path)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch']
            best_loss = ckpt['best_loss']
            
        scaler = GradScaler(enabled=self.amp)
        train_idx = self.data_train[fold]
        val_idx = self.data_val[fold]
        loss_meter_tr = AverageMeter()
        loss_meter_val = AverageMeter()
        
        train_dataset = self.dataset(bv=self.vocab,
                                   q_idx=train_idx,
                                   mode='train')
        val_dataset = self.dataset(bv=self.vocab,
                                 q_idx=val_idx,
                                 mode='val')
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
          )
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=self.test_batch_size,
            collate_fn=self.collate_fn
          )
        
        os.makedirs('artifacts/training', exist_ok=True)
        if self.model_name in ['sbert', 'deberta']:
            total_steps = 2 * int(self.train_data.shape[0] * 0.8 // self.batch_size + 1)
            warmup_steps = int(0.1 * total_steps)
            scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        for epoch in range(start_epoch, self.epochs):
            model.train()
            train_dl = tqdm(train_dataloader,
                            total=len(train_dataloader),
                            desc=f'running fold {fold} in training')
            for step, batch in enumerate(train_dl):
                current_lr = optimizer.param_groups[0]['lr']
                for i, v in enumerate(batch):
                    if isinstance(v, torch.Tensor):
                        batch[i] = v.to(self.device)
                
                optimizer.zero_grad()
                y_true = batch[-1]
                with autocast(enabled=self.amp):
                    y_pred = model(batch)
                    loss = self.loss_fn(y_pred.view(-1), y_true.float())
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if self.model_name in ['sbert', 'deberta']:
                    scheduler.step()
                
                loss_meter_tr.update(loss.item(), 1)
                loss = loss_meter_tr.average
                train_dl.set_postfix({
                    f'epoch {epoch} loss': f'{loss:.5f}',
                    'lr': f'{current_lr: .5f}'
                    } 
                  )
            
            with torch.no_grad():
                model.eval()
                val_dl = tqdm(val_dataloader,
                              total=len(val_dataloader),
                              desc=f'running fold {fold} in evaluation')
                features,ids = [], []
                for batch in val_dl:
                    for i, v in enumerate(batch):
                        if isinstance(v, torch.Tensor):
                            batch[i] = v.to(self.device)
                
                    yt = batch[-1]
                    with autocast(enabled=self.amp):
                        yp, feature = model(batch, return_embedding=True)
                        val_loss = self.loss_fn(yp.view(-1), yt.float())
                        features.append(feature)
                    id = batch[0].cpu().numpy()
                    ids.append(id)
                    loss_meter_val.update(val_loss, 1)
                    val_loss = loss_meter_val.average
                    val_dl.set_postfix({
                        f'epoch {epoch} loss': f'{val_loss: .5f}'
                      })
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    bad_epoch = 0
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'epoch': epoch
                      }, checkpoint_path)
                    ids = np.concatenate(ids)
                    features = torch.cat(features,dim=0)
                    features = features.detach().cpu().numpy()
                    features = np.concatenate([ids[:, None], features], axis=1)
                    np.save(f'artifacts/training/{self.model_name}_features_{fold}{self.suffix}.npy', features.astype(np.float32))
                else:
                    bad_epoch += 1
            if bad_epoch == self.early_stopping:
                print(f'early stopping reaches at epoch: {epoch}')
                break
    
    @torch.no_grad()
    def predict(self):
        os.makedirs('artifacts/prediction', exist_ok=True)
        for fold in range(5):
            checkpoint = f'checkpoints/{self.model_name}_{fold}{self.suffix}.pth'
            model = self.models[fold]
            ckpt = torch.load(checkpoint)
            model.load_state_dict(ckpt['model'])
            
            test_index = np.arange(len(self.test))
            test_dataset = self.dataset(
                bv=self.vocab,
                q_idx=test_index,
                mode='test'
              )
            test_dataloader = DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=self.test_batch_size
              )
            test_dl = tqdm(test_dataloader,
                          total=len(test_dataloader),
                          desc=f'predicting fold {fold}')
            
            model.eval()
            ids, features = [], []
            for batch in test_dl:
                for i, v in enumerate(batch):
                    if isinstance(v, torch.Tensor):
                        batch[i] = v.to(self.device)
                with autocast(enabled=self.amp):
                    _, feature = model(batch, return_embedding=True)
                    features.append(feature)
                id = batch[0].cpu().numpy()
                ids.append(id)
            ids = np.concatenate(ids)
            features = torch.cat(features,dim=0)
            features = features.detach().cpu().numpy()
            features = np.concatenate([ids[:, None], features], axis=1)
            np.save(f'artifacts/prediction/{self.model_name}_features_{fold}{self.suffix}.npy', features.astype(np.float32))
      
    def merge(self):
        print('merging scattered features')
        train_feats = []
        for fold in range(5):
            prediction_path = f'artifacts/prediction/{self.model_name}_features_{fold}{self.suffix}.npy'
            train_path = f'artifacts/training/{self.model_name}_features_{fold}{self.suffix}.npy'
            test_features = np.load(prediction_path)
            train_features = np.load(train_path)
            if fold == 0:
                total_features = np.zeros([*test_features.shape], dtype=np.float32)
            total_features += test_features
            train_feats.append(train_features.astype(np.float32))
            os.remove(prediction_path)
            os.remove(train_path)
            
        total_features /= 5
        train_features = np.concatenate(train_feats, axis=0)
        sorted_index = train_features[:, 0].argsort()
        train_features = train_features[sorted_index]
        sorted_index = total_features[:, 0].argsort()
        test_features = total_features[sorted_index]
        
        np.save(f'artifacts/training/{self.model_name}_features{self.suffix}.npy', train_features)
        np.save(f'artifacts/prediction/{self.model_name}_features{self.suffix}.npy', total_features)

def parse_args():
    def int_list(arg):
        arg = arg.strip("[]")
        return [int(x) for x in arg.split(",") if x.strip()]
    
    def binary_list(arg):
        nums = []
        lst = int_list(arg)
        for v in lst:
            if v not in [0, 1]:
                raise argparse.ArgumentTypeError("Values must be 0 or 1.")
            nums.append(v)
        return nums
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--warm-start-folds', type=binary_list)
    parser.add_argument('--model-name', type=str, default='none')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--early-stopping', type=int, default=3)
    parser.add_argument('--training-folds', type=int_list)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--test-batch-size', type=int, default=512)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    bv = BuildVocab('data/train.csv',
                    'data/test.csv')
    bv.build_char_vocab()
    if args.model_name in ['bimpm', 'self_1']:
        vec_model = load_facebook_model('artifacts/cc.en.300.bin')
    elif args.model_name in ['diin', 'esim']:
        glove = {}
        path = 'artifacts/glove.840B.300d.txt'
        file_size = os.path.getsize(path)
        with open(path, 'r', encoding='utf8') as f, \
            tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading GloVe") as pbar:
                for line in f:
                    parts = line.rstrip().split(' ')
                    word = parts[0]
                    vec = np.array(parts[1:], dtype=np.float32)
                    glove[word] = vec
                    pbar.update(len(line.encode('utf8')))
        vec_model = glove
    else:
        vec_model = None
    
    trainer = Trainer(
        vocab=bv,
        vec_model=vec_model,
        amp=args.amp,
        model_name=args.model_name,
        early_stopping=args.early_stopping,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size
      )
    for fold, warm_start in zip(args.training_folds, args.warm_start_folds):
        trainer.train(fold, warm_start=warm_start)
    trainer.predict()
    trainer.merge()