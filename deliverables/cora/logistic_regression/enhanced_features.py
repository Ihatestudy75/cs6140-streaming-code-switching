"""Extra causal features derived solely from a frozen tokenizer vocabulary.

Keep original token hashes unchanged in a separate block, so the feature
ablation does not silently change collisions in the original representation.
"""
from collections import Counter
from functools import lru_cache
import unicodedata
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import FeatureHasher
from model import vectorize_prefixes as base_vectorize

EXTRA_DIM = 2**14


@lru_cache(maxsize=None)
def describe(token):
    body = token[2:] if token.startswith('##') else token
    chars = [c for c in body if c.isalpha()]
    scripts = set()
    for c in chars:
        name = unicodedata.name(c, '')
        group = next((g for g in ('LATIN','CYRILLIC','ARABIC','DEVANAGARI','HIRAGANA','KATAKANA','HANGUL','THAI','HEBREW','GREEK') if g in name), None)
        scripts.add(group or ('HAN' if 'CJK' in name or 'IDEOGRAPH' in name else 'OTHER'))
    return dict(continuation=token.startswith('##'),
        punct=bool(body) and all(unicodedata.category(c).startswith('P') for c in body),
        digit=bool(body) and all(c.isdigit() for c in body),
        unknown=token=='[UNK]', capital=bool(body) and body[0].isupper(),
        length=min(len(body),8), script='+'.join(sorted(scripts)) or 'NONE')


def extra_features(ids, window, vocab, mode):
    if mode not in ('morph','script','all'):
        raise ValueError(mode)
    info=[describe(vocab[int(i)]) for i in ids[-window:]]
    if not info:
        raise ValueError('Empty prefix')
    out={}
    if mode in ('morph','all'):
        for pos, d in enumerate(reversed(info[-3:])):
            for k in ('continuation','punct','digit','unknown','capital','length'):
                out[f'morph:p{pos}:{k}={d[k]}']=1.
        for k in ('continuation','punct','digit','unknown','capital'):
            out[f'morph:rate:{k}']=sum(d[k] for d in info)/len(info)
        distance=next((i for i,d in enumerate(reversed(info)) if d['punct']),None)
        bucket='none_in_window' if distance is None else str(min(distance,8))
        out['morph:punct_distance='+bucket]=1.
    if mode in ('script','all'):
        scripts=[d['script'] for d in info]
        for pos,s in enumerate(reversed(scripts[-3:])):
            out[f'script:p{pos}={s}']=1.
        for s,n in Counter(scripts).items():
            out['script:rate:'+s]=n/len(scripts)
        for n in (2,3):
            if len(scripts)>=n:
                out[f'script:suffix{n}='+','.join(scripts[-n:])]=1.
        letters=[s for s in scripts if s!='NONE']
        out['script:changes']=sum(a!=b for a,b in zip(letters,letters[1:]))/max(1,len(letters)-1)
        out['script:distinct='+str(min(3,len(set(letters))))]=1.
    return out


def vectorize(prefixes, window, vocab, mode):
    prefixes=list(prefixes)
    base=base_vectorize(prefixes,window)
    if mode=='base':
        return base
    extra=FeatureHasher(n_features=EXTRA_DIM,input_type='dict',alternate_sign=False,dtype=np.float64).transform(
        extra_features(ids,window,vocab,mode) for ids in prefixes).tocsr()
    return hstack([base,extra],format='csr')
