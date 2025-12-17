import re
import spacy
from tqdm import tqdm
from collections import defaultdict

nlp = spacy.load("en_core_web_sm",
                 disable=["parser"]) 
_num_re = re.compile(r"^\d+(\.\d+)?$")

def norm_tok(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
    return t

def is_number(t: str) -> bool:
    return _num_re.match(t) is not None

def add_rule(P, i, j, conf):
    if conf >  P[i, j]:
        P[i, j] = conf

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
                    ner_span1, ner_span2):
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
    
    pairs = []
    for i, w in enumerate(t1):
        if not w:
            continue
        
        if is_number(w) and w in tok2_pos:
            for j in tok2_pos[w]:
                pairs.append((i, j , 1.0))
            continue
        
        if len(w) >= 2 and w in tok2_pos:
            for j in tok2_pos[w]:
                pairs.append((i, j, 0.9))
    
    ent2 = defaultdict(list)
    for sp in ner_span2:
        key = (norm_ent_text(sp['text']), sp['label'])
        ent2[key] = sp
    
    for sp1 in ner_span1:
        key = (norm_ent_text(sp1['text']), sp1['label'])
        sp2 = ent2.get(key)
        if sp2 is None:
            continue
        
        label = sp1["label"]
        if label in {"DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"}:
            c = 0.95
        else:
            c = 0.85

        for i in range(sp1['start'], sp1['end']):
            for j in range(sp2['start'], sp2['end']):
                pairs.append((i, j, c))
    
    for i, lem in enumerate(lemma1):
        lem = norm_tok(lem)
        if not lem or len(lem) < 4:
            continue
        js = lemma2_pos.get(lem)
        if not js:
            continue
        
        p1 = pos1[i]
        for j in js:
            if pos_compatible(p1, pos2[j]): 
                pairs.append((i, j, 0.45))
    
    best_conf = {}
    for a, b, v in pairs:
        key = (a, b)
        if key not in best_conf or v > best_conf[key]:
            best_conf[key] = v
    pairs = [(a, b, v) for (a, b), v in best_conf.items()]
    return pairs

def featurize_doc(doc, drop_punct=True):
    tokens, offsets, lemmas, pos = [], [], [], []
    sp2w = {}

    for tok in doc:
        if tok.is_space:
            continue
        if drop_punct and tok.is_punct:
            continue

        sp2w[tok.i] = len(tokens)
        tokens.append(tok.text.lower())
        offsets.append((tok.idx, tok.idx + len(tok.text)))
        lemmas.append(tok.lemma_.lower())
        pos.append(tok.pos_)

    ner_spans = []
    for ent in doc.ents:
        mapped = [sp2w[i] for i in range(ent.start, ent.end) if i in sp2w]
        if not mapped:
            continue
        ner_spans.append({
            "text": ent.text,
            "label": ent.label_,
            "start": min(mapped),
            "end": max(mapped) + 1
        })

    return tokens, offsets, lemmas, pos, ner_spans

def build_alignments_for_batch(q1_list, q2_list, feat_cache):
    aligned_pairs = []
    for q1, q2 in zip(q1_list, q2_list):
        tokens1, offsets1, lemma1, pos1, ner1 = feat_cache[q1]
        tokens2, offsets2, lemma2, pos2, ner2 = feat_cache[q2]

        P = build_alignment(tokens1, tokens2, lemma1, lemma2, pos1, pos2, ner1, ner2)
        aligned_pairs.append(P)

    return aligned_pairs

def build_question_feature_cache(texts, nlp, batch_size=2048):
    cache = {}
    uniq = list(dict.fromkeys(texts))

    for doc, text in zip(
        tqdm(nlp.pipe(uniq, batch_size=batch_size), total=len(uniq)),
        uniq):
        cache[text] = featurize_doc(doc, drop_punct=True)

    return cache
    
