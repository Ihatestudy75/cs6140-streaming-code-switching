"""Observed punctuation at a token boundary; a descriptive proxy, not syntax."""
import unicodedata

SENTENCE_ENDINGS = set('.!?。！？؟।॥…')
CLAUSE_SEPARATORS = set(',;:，；：、،؛—–-')


def boundary_kind(text, offsets, position):
    """Use only observed characters through token t, never token t+1."""
    if position < 0 or position >= len(offsets):
        raise ValueError('Invalid token position')
    start, end = offsets[position]
    if not 0 <= start <= end <= len(text):
        raise ValueError('Invalid text offsets')
    observed = text[:end].rstrip()
    if not observed:
        return 'unclassified'
    char = observed[-1]
    if char in SENTENCE_ENDINGS:
        return 'sentence_punctuation'
    if char in CLAUSE_SEPARATORS:
        return 'clause_punctuation'
    if unicodedata.category(char).startswith('P'):
        return 'other_punctuation'
    return 'non_punctuation'


def boundary_group(kind):
    if kind in ('sentence_punctuation','clause_punctuation','other_punctuation'):
        return 'punctuation'
    if kind == 'non_punctuation':
        return 'within_clause_proxy'
    return 'unclassified'
