import re
import spacy
nlp = spacy.load("en_core_web_sm") 

_WORD_RE = re.compile(r'[a-z]+|\d+(?:\.\d+)?', re.IGNORECASE)
def tokenize_with_offsets(text):
    tokens, offsets = [], []
    for m in _WORD_RE.finditer(text):
        tok = m.group(0)
        tokens.append(tok.lower())
        offsets.append((m.start(), m.end()))
    return tokens, offsets

def char_span_to_token_span(token_offsets, start, end):
    start_tok, end_tok = None, None
    for i, (ts, te) in enumerate(token_offsets):
        if te <= start:
            continue
        if ts >= end:
            break
        if start_tok is None:
            start_tok = i
        end_tok = i + 1
        
    if start_tok is None:
        return None
    return start_tok, end_tok

def build_ner_span(text):
    spans = []
    tokens, offsets = tokenize_with_offsets(text)
    doc = nlp(text)
    for ent in doc.ents:
        token_span = char_span_to_token_span(offsets, ent.start_char, ent.end_char)
        if token_span is None:
            continue
        s, e = token_span
        if s >= e:
            continue
        spans.append({
                'text': ent.text,
                'label': ent.label_,
                'start': s,
                'end': e
            })
    
    seen = set()
    dedup = []
    for sp in spans:
        key = (sp["start"], sp["end"], sp["label"], sp["text"].strip().lower())
        if key not in seen:
            seen.add(key)
            dedup.append(sp)
    return dedup

#%%
_, token_offsets = tokenize_with_offsets(q1)
import pandas as pd
# train = pd.read_csv('data/train.csv')
# q1 = train.loc[100, 'question1']
# q2 = train.loc[100, 'question2']
# for m in _WORD_RE.finditer(q):
#     tok = m.group(0)
#     print(m.start(), m.end())
token_offsets
    
#%%


doc = nlp('Bank of England increased the interest rate, people will have to pay more morgages')
for ent in doc.ents:
    print(ent.start_char, ent.end_char, ent.label_)