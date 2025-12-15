import re
import spacy
import torch
import numpy as np
from collections import defaultdict

nlp = spacy.load("en_core_web_sm") 
_num_re = re.compile(r"^\d+(\.\d+)?$")

# _WORD_RE = re.compile(r'[a-z]+|\d+(?:\.\d+)?', re.IGNORECASE)
# def tokenize_with_offsets(text):
#     tokens, offsets = [], []
#     for m in _WORD_RE.finditer(text):
#         tok = m.group(0)
#         tokens.append(tok.lower())
#         offsets.append((m.start(), m.end()))
#     return tokens, offsets

def tokenize_with_offsets(text):
    doc = nlp(text)
    tokens, offsets = [], []
    for tok in doc:
        if tok.pos_ == 'PUNCT':
            continue
        tokens.append(tok.text.lower())
        offsets.append((tok.idx, tok.idx+len(tok.text)))
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
                'text': ent.text.lower(),
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

def norm_tok(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
    return t

def is_number(t: str) -> bool:
    return _num_re.match(t) is not None

def add_rule(P, i, j, conf):
    if conf >  P[i, j]:
        P[i, j] = conf
        
def get_tokens_lemmas_pos(text):
    toks = nlp(text)
    lemma = [tok.lemma_ for tok in toks if tok.pos_ != 'PUNCT']
    pos = [tok.pos_ for tok in toks if tok.pos_ != 'PUNCT']
    return lemma, pos

def norm_ent_text(s):
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    return s

def pos_compatible(p1, p2):
    if p1 is None or p2 is None:
        return True
    if p1 == p2:
        return True
    if {p1, p2} <= {"AUX", "VERB"}:
        return True
    if {p1, p2} <= {"NOUN", "PROPN"}:
        return True
    return False

def build_alignment(tokens1, tokens2,
                    lemma1, lemma2,
                    pos1, pos2,
                    ner_span1, ner_span2,
                    device, topk):
    L1, L2 = len(tokens1), len(tokens2)
    P = torch.zeros([L1, L2], device=device, dtype=torch.float32)
    
    t1 = [norm_tok(tok) for tok in tokens1]
    t2 = [norm_tok(tok) for tok in tokens2]
    
    tok2_pos = defaultdict(list)
    for j, w in enumerate(t2):
        if w:
            tok2_pos[w].append(j)
    
    lemma2_pos = defaultdict(list)
    for j, w in enumerate(lemma2):
        w = norm_tok(w)
        if w:
            lemma2_pos[w].append(j)
    
    for i, w in enumerate(t1):
        if not w:
            continue
        
        if is_number(w) and w in tok2_pos:
            for j in tok2_pos[w]:
                add_rule(P, i, j, 1.0)
        
        if len(w) >= 2 and w in tok2_pos:
            for j in tok2_pos[w]:
                add_rule(P, i, j, 0.9)
    
    ent2 = defaultdict(list)
    for sp in ner_span2:
        key = (norm_ent_text(sp['text']), sp['label'])
        ent2[key] = sp
    
    for sp1 in ner_span1:
        key = (norm_ent_text(sp1['text']), sp1['label'])
        if key not in ent2:
            continue
        
        label = sp1["label"]
        if label in {"DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"}:
            c = 0.95
        else:
            c = 0.85

        for sp2 in ent2.values():
            for i in range(sp1['start'], sp1['end']):
                for j in range(sp2['start'], sp2['end']):
                    add_rule(P, i, j, c)
    
    for i, lem in enumerate(lemma1):
        lem = norm_tok(lem)
        if not lem or len(lem) < 4:
            continue
        if lem not in lemma2_pos:
            continue
        
        if not pos_compatible(pos1[i], pos2[j]):
            continue
        add_rule(P, i, j, 0.45)
    
    if topk is not None and topk < L2:
        vals, idx = torch.topk(P, k=topk, dim=1)
        P2 = torch.zeros_like(P)
        P2.scatter_(1, idx, vals)
        P = P2
    return P

            
#%%
q = "I cant believe Bank of England increased the interest rate again, more morgages are awaiting33!"

q1 = train.loc[1001, 'question1']
q2 = train.loc[1001, 'question2']
tokens1 = [tok.text.lower() for tok in nlp(q1) if tok.pos_ != 'PUNCT']
tokens2 = [tok.text.lower() for tok in nlp(q2) if tok.pos_ != 'PUNCT']
lemma1, pos1 = get_tokens_lemmas_pos(q1)
lemma2, pos2 = get_tokens_lemmas_pos(q2)
ner_span1 = build_ner_span(q1)
ner_span2 = build_ner_span(q2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p = build_alignment(tokens1, tokens2,
                    lemma1, lemma2,
                    pos1, pos2,
                    ner_span1, ner_span2,
                    device=device, topk=3)
    
#%%
q1 = train.loc[1003, 'question1']
q2 = train.loc[1003, 'question2']
q1, q2
# p
