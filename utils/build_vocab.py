import os
import json
import spacy
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm

class BuildVocab:
    def __init__(self,
                 train_path,
                 test_path,
                 max_len=30):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        train = self._mask_data(train)
        test = self._mask_data(test)
        self.train_data = train
        self.test_data = test
        self.words_index = {}
        self.words_index['<pad>'] = 0
        self.words_index['<unk>'] = 1
        if not os.path.exists('artifacts'):
            raise FileNotFoundError('artifacts folder must exist, create it first')
        self.save_path = 'artifacts/words_index.json'
        self.vocab_exe = False
        self.max_len = max_len
    
    def _mask_data(self, data):
        mask = pd.isnull(data['question1']) | pd.isnull(data['question2'])
        data = data[~mask]
        data = data.reset_index().drop('index', axis=1)
        return data
    
    def vocab(self):
        ind = 2
        nlp = spacy.blank("en") 
        tokenizer = nlp.tokenizer
        self.tokenizer = tokenizer
        columns = ['question1', 'question2']
        self.columns = columns
        questions = np.array([])
        
        for q in columns:
            self.train_data[q] = self.train_data[q].fillna("").astype(str)
            self.test_data[q] = self.test_data[q].fillna("").astype(str)
            questions = np.hstack([questions, self.train_data[q].values])
            questions = np.hstack([questions, self.test_data[q].values])
        
        tq_qt = tqdm(questions, total=len(questions), desc='building vocabulary')
        for q in tq_qt:
            tokens = tokenizer(q)
            for token in tokens:
                w = token.text.lower()
                if w not in self.words_index:
                    self.words_index[w] = ind
                    ind += 1
        self.vocab_exe = True
    
    def convert_sent_to_index(self):
        if not self.vocab_exe:
            raise ValueError('please run vocab method first to build the vocabulary')
        
        unk_idx = self.words_index['<unk>']
        for q in self.columns:
            for i, d in enumerate([self.train_data, self.test_data]):
                sentences, sent_lens = [], []
                cur_data = 'train' if i == 0 else 'test'
                tq_ts = tqdm(d[q].values, total=d.shape[0], desc=f'converting sentence to word index for {cur_data} {q}')
                for sent in tq_ts:
                    tokens = self.tokenizer(sent)
                    sent_index = [self.words_index.get(token.text.lower(), unk_idx) for token in tokens]
                    sent_index = sent_index[:self.max_len]
                    sent_len = len(sent_index)
                    sent_lens.append(sent_len)
                    if len(sent_index) < self.max_len:
                        sent_index += [self.words_index['<pad>']] * (self.max_len - len(sent_index))
                    sentences.append(sent_index)
                sentences = np.array(sentences)
                np.save(f'artifacts/{cur_data}_{q}.npy', sentences)
                np.save(f'artifacts/{cur_data}_{q}_len.npy', np.array(sent_lens))
    
    def save(self):
        if len(self.words_index) <= 1:
            warnings.warn('build the vocab dictionary before saving')
        with open(self.save_path, 'w', encoding='utf8') as f:
            json.dump(self.words_index, f, ensure_ascii=False)
    
    def load_dict(self):
        if not os.path.exists(self.save_path):
            raise ValueError('no vocab dictionary at the location')
        with open(self.save_path, 'r', encoding='utf8') as f:
            words_index = json.load(f)
            self.words_index = words_index
            return words_index
    
    def load_arrays(self):
        nps = {}
        for q in ['question1', 'question2']:
            for d in ['train', 'test']:
                base_path = f'artifacts/{d}_{q}'
                cur_path = f'{base_path}.npy'
                len_path = f'{base_path}_len.npy'
                if not os.path.exists(cur_path):
                    raise ValueError(f'{cur_path} does not exist, please run convert_sent_to_index method first')
                data = np.load(cur_path)
                data_len = np.load(len_path)
                f_name = os.path.basename(base_path)
                f_len_name = os.path.basename(len_path)[:-4]
                nps[f_name] = data
                nps[f_len_name] = data_len
        return nps

if __name__ == '__main__':
    build_vocab = BuildVocab(
        'data/train.csv',
        'data/test.csv',
        max_len=40
      )
    build_vocab.vocab()
    build_vocab.save()
    build_vocab.convert_sent_to_index()

#%%

