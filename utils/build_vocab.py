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
                 test_path):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        self.train_data = train
        self.test_data = test
        self.words_index = {}
        self.words_index['<pad>'] = 0
        self.words_index['<unk>'] = 1
        if not os.path.exists('artifacts'):
            raise FileNotFoundError('artifacts folder must exist, create it first')
        self.save_path = 'artifacts/words_index.json'
    
    def vocab(self):
        ind = 2
        nlp = spacy.blank("en") 
        tokenizer = nlp.tokenizer
        columns = ['question1', 'question2']
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
    
    def save(self):
        if len(self.words_index) <= 1:
            warnings.warn('build the vocab dictionary before saving')
        with open(self.save_path, 'w', encoding='utf8') as f:
            json.dump(self.words_index, f, ensure_ascii=False)
    
    def load(self):
        if not os.path.exists(self.save_path):
            raise ValueError('no vocab dictionary at the location')
        with open(self.save_path, 'r', encoding='utf8') as f:
            words_index = json.load(f)
            return words_index

if __name__ == '__main__':
    build_vocab = BuildVocab(
        'data/train.csv',
        'data/test.csv'
      )
    build_vocab.vocab()
    build_vocab.save()

#%%
