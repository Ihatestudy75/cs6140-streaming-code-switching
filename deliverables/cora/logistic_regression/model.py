"""Causal sparse token features and two logistic-regression classifiers."""
from collections import Counter
import numpy as np
from sklearn.feature_extraction import FeatureHasher

HASH_FEATURES = 2**18


def token_features(input_ids, window):
    """Only pass token IDs available at prediction time, ending at position t."""
    if window < 1 or len(input_ids) == 0:
        raise ValueError('A positive window and nonempty prefix are required.')
    ids = tuple(int(i) for i in input_ids[-window:])
    values = {f'position={offset}:token={token}': 1.0
              for offset, token in enumerate(reversed(ids))}
    for token, count in Counter(ids).items():
        values[f'bag:token={token}'] = count / len(ids)
    for size in (2, 3):
        if len(ids) >= size:
            values[f'suffix={size}:' + ','.join(map(str, ids[-size:]))] = 1.0
    return values


def vectorize_prefixes(prefixes, window):
    hasher = FeatureHasher(n_features=HASH_FEATURES, input_type='dict',
                           alternate_sign=False, dtype=np.float64)
    return hasher.transform(token_features(ids, window) for ids in prefixes).tocsr()


def at_position(input_ids, position, window):
    """Slice before feature construction: future IDs cannot enter the features."""
    if position < 0 or position >= len(input_ids):
        raise ValueError('Position is outside the sequence.')
    return token_features(input_ids[max(0, position + 1 - window):position + 1], window)


def predict_prefixes(bundle, prefixes):
    switch_x = vectorize_prefixes(prefixes, bundle['switch_config']['window'])
    scores = bundle['switch_model'].predict_proba(switch_x)[:, 1]
    duration_window = bundle['duration_config']['window']
    duration_x = (switch_x if duration_window == bundle['switch_config']['window']
                  else vectorize_prefixes(prefixes, duration_window))
    durations = bundle['duration_model'].predict(duration_x)
    return scores, durations
