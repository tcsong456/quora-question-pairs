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
                 max_len=30,
                 max_char_len=10):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        train = self._mask_data(train)
        test = self._mask_data(test)
        test['test_id'] = test['test_id'].map(int)
        test = test.drop_duplicates('test_id')
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
        self.max_char_len = max_char_len
    
    def _mask_data(self, data):
        mask = pd.isnull(data['question1']) | pd.isnull(data['question2'])
        data = data[~mask]
        data = data.reset_index().drop('index', axis=1)
        return data
    
    def build_vocab(self):
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
    
    def build_char_vocab(self):
        ind = 2
        self.char_index = {}
        self.char_index['<pad>'] = 0
        self.char_index['<unk>'] = 1
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789.,!?;:\'"-()[]{}@#$%^&*_+=/\\'
        for c in chars:
            self.char_index[c] = ind
            ind += 1
    
    def convert_sent_to_index(self):
        if not self.vocab_exe:
            raise ValueError('please run vocab method first to build the vocabulary')

        unk_idx = self.words_index['<unk>']
        unk_char_idx = self.char_index['<unk>']
        for q in self.columns:
            for i, d in enumerate([self.train_data, self.test_data]):
                sentences, sent_lens = [], []
                words = []
                cur_data = 'train' if i == 0 else 'test'
                tq_ts = tqdm(d[q].values, total=d.shape[0], desc=f'converting sentence to word index for {cur_data} {q}')
                for sent in tq_ts:
                    tokens = self.tokenizer(sent)
                    char_indices = [[self.char_index.get(c, unk_char_idx) for c in w.text.lower()] for w in tokens]
                    char_indices = [indice+[self.char_index['<pad>']]*(self.max_char_len-len(indice)) if len(indice)<self.max_char_len else indice[:self.max_char_len] for indice in char_indices]
                    sent_index = [self.words_index.get(token.text.lower(), unk_idx) for token in tokens]
                    sent_index = sent_index[: self.max_len]
                    char_indices = char_indices[: self.max_len]
                    sent_len = len(sent_index)
                    sent_lens.append(sent_len)
                    if len(sent_index) < self.max_len:
                        padding_char = [self.char_index['<pad>']] * self.max_char_len
                        for _ in range(self.max_len - len(sent_index)):
                            char_indices.append(padding_char)
                        sent_index += [self.words_index['<pad>']] * (self.max_len - len(sent_index))
                    sentences.append(sent_index)
                    words.append(np.array(char_indices))
                words = np.stack(words, axis=0).astype(np.int8)
                sentences = np.array(sentences)
                np.save(f'artifacts/{cur_data}_{q}.npy', sentences)
                np.save(f'artifacts/{cur_data}_{q}_len.npy', np.array(sent_lens))
                np.save(f'artifacts/{cur_data}_{q}_char.npy', words)
    
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
                char_path = f'{base_path}_char.npy'
                if not os.path.exists(cur_path):
                    raise ValueError(f'{cur_path} does not exist, please run convert_sent_to_index method first')
                data = np.load(cur_path)
                for path in [len_path, char_path]:
                    f_len_name = os.path.basename(path)[:-4]
                    df = np.load(path)
                    nps[f_len_name] = df
                f_name = os.path.basename(base_path)
                nps[f_name] = data
        return nps

if __name__ == '__main__':
    build_vocab = BuildVocab( 
        'data/train.csv',
        'data/test.csv',
        max_len=40,
        max_char_len=20
      )
    # build_vocab.build_vocab()
    build_vocab.build_char_vocab()
    # build_vocab.save()
    # build_vocab.convert_sent_to_index()

#%%

