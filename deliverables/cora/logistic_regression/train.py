"""Fit the new primary baseline locally using the existing, frozen data split.

This runner reads the licensed data in place and never downloads or uploads it.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import csv
import gc
import hashlib
import json
import platform
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from metrics import choose_threshold, evaluate
from model import vectorize_prefixes, predict_prefixes, HASH_FEATURES

DEFAULT_REFERENCE = Path(os.environ.get('CS6140_REFERENCE_ROOT', 'reference'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def read_jsonl(path):
    with Path(path).open() as file:
        return [json.loads(line) for line in file if line.strip()]


def load_data(reference, maximum_window):
    run = reference / 'runs/all-v3'
    data_path = run / 'combined/labeled_tokens.jsonl'
    splits_path = run / 'combined/splits.jsonl'
    previous_manifest = json.loads((run / 'run_manifest.json').read_text())
    assert sha(data_path) == previous_manifest['data_sha256'], 'Source data changed.'
    assert sha(splits_path) == previous_manifest['split_sha256'], 'Source split changed.'
    entries = read_jsonl(splits_path)
    entries_by_id = {s['sequence_id']: s for s in entries}
    assert len(entries) == len(entries_by_id)
    for field in ('record_id', 'group_id'):
        groups = defaultdict(set)
        for entry in entries:
            groups[entry[field]].add(entry['split'])
        assert all(len(v) == 1 for v in groups.values()), f'{field} crosses splits.'
    pilots = set((run / 'all_pilot_record_ids.txt').read_text().split())
    assert all(e['split'] == 'train' for e in entries if e['record_id'] in pilots)
    types = {}
    for owner in ('shaoyu', 'yulun', 'yufan'):
        for file in sorted((reference / 'source/deliverables' / owner).glob(f'*/cs_type_{owner}.jsonl')):
            for row in read_jsonl(file):
                types[row['sequence_id']] = row['cs_type']
    samples = {key: [] for key in ('train', 'validation', 'test')}
    seen = set()
    for row in read_jsonl(data_path):
        sid = row['sequence_id']
        assert sid not in seen and sid in entries_by_id and sid in types
        seen.add(sid)
        entry = entries_by_id[sid]
        assert entry['record_id'] == row['record_id']
        ids = row['input_ids']
        assert len(ids) == len(row['switch_valid_mask']) == len(row['switch_labels'])
        assert len(ids) == len(row['duration_labels']) == len(row['duration_valid_mask'])
        assert row['switch_valid_mask'] == [y >= 0 for y in row['switch_labels']]
        assert row['duration_valid_mask'] == [d >= 0 for d in row['duration_labels']]
        for t, valid in enumerate(row['switch_valid_mask']):
            if not valid:
                continue
            y, d = row['switch_labels'][t], row['duration_labels'][t]
            assert y in (0, 1) and d in (-1, 0, 1, 2)
            assert d < 0 or y == 1
            samples[entry['split']].append({
                'inputs': {'input_ids': ids[max(0, t+1-maximum_window):t+1]},
                'targets': {'switch': y, 'duration': d},
                'provenance': {'sequence_id': sid, 'position': t,
                               'language_pair': row['language_pair']}})
    assert seen == set(entries_by_id)
    audit = {'data_sha256': sha(data_path), 'split_sha256': sha(splits_path),
             'reference_runner_sha256': sha(reference / 'run.py'),
             'sequences': dict(Counter(s['split'] for s in entries)),
             'positions': {k: len(v) for k, v in samples.items()},
             'positive_switches': {k: sum(s['targets']['switch'] for s in v) for k, v in samples.items()},
             'known_durations': {k: sum(s['targets']['duration'] >= 0 for s in v) for k, v in samples.items()},
             'checks': {'source_hashes_match': True, 'record_and_group_disjoint': True,
                        'pilot_train_only': True, 'valid_masks': True}}
    return samples, types, audit


def fit_lr(x, y, c, class_weight):
    model = LogisticRegression(C=c, class_weight=class_weight, solver='liblinear',
                               max_iter=500, tol=1e-4, random_state=6140)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(x, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError('Training did not converge. Increase max_iter before evaluating test.')
    return model, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'runs/logistic-v1')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new output directory to preserve prior results.')
    args.out.mkdir(parents=True)
    windows, c_values, weights = [5, 20], [0.1, 1.0], [None, 'balanced']
    samples, types, audit = load_data(args.reference_root, max(windows))
    write_json(args.out / 'audit.json', audit)
    print('Fixed split loaded:', audit['positions'], flush=True)
    train, valid = samples['train'], samples['validation']
    train_y = np.array([s['targets']['switch'] for s in train])
    train_d = np.array([s['targets']['duration'] for s in train])
    valid_d = np.array([s['targets']['duration'] for s in valid])
    train_known, valid_known = train_d >= 0, valid_d >= 0
    history = []
    best_switch = best_duration = None
    best_switch_metric = best_duration_metric = -1.0
    for window in windows:
        started = time.perf_counter()
        x_train = vectorize_prefixes((s['inputs']['input_ids'] for s in train), window)
        x_valid = vectorize_prefixes((s['inputs']['input_ids'] for s in valid), window)
        duration_train = x_train[train_known]
        duration_valid = x_valid[valid_known]
        print(f'Window {window}: {x_train.shape}, {x_train.nnz:,} nonzeros; '
              f'features built in {time.perf_counter()-started:.1f}s', flush=True)
        for c in c_values:
            for weight in weights:
                config = {'window': window, 'C': c, 'class_weight': weight}
                switch_model, switch_seconds = fit_lr(x_train, train_y, c, weight)
                scores = switch_model.predict_proba(x_valid)[:, 1]
                choice = choose_threshold(valid, scores.tolist(), types)
                metric = choice['validation_macro_pair_type_switch_f1']
                duration_model, duration_seconds = fit_lr(duration_train, train_d[train_known], c, weight)
                predictions = duration_model.predict(duration_valid)
                duration_metric = f1_score(valid_d[valid_known], predictions, labels=[0,1,2], average='macro', zero_division=0)
                row = dict(config, **choice, validation_duration_macro_f1=float(duration_metric),
                           switch_fit_seconds=switch_seconds, duration_fit_seconds=duration_seconds,
                           switch_iterations=switch_model.n_iter_.tolist(), duration_iterations=duration_model.n_iter_.tolist())
                history.append(row)
                if metric > best_switch_metric:
                    best_switch_metric = metric
                    best_switch = (switch_model, dict(config, **choice))
                if duration_metric > best_duration_metric:
                    best_duration_metric = float(duration_metric)
                    best_duration = (duration_model, dict(config, validation_duration_macro_f1=float(duration_metric)))
                write_json(args.out / 'validation_search.json', history)
                print(row, flush=True)
        del x_train, x_valid, duration_train, duration_valid
        gc.collect()
    bundle = {'switch_model': best_switch[0], 'switch_config': best_switch[1],
              'duration_model': best_duration[0], 'duration_config': best_duration[1],
              'hash_features': HASH_FEATURES, 'feature_version': 'causal-position-bag-suffix-v1',
              'duration_classes': ['1-2', '3-6', '7+']}
    joblib.dump(bundle, args.out / 'logistic_model.joblib', compress=3)
    results = {'config': {'switch': bundle['switch_config'], 'duration': bundle['duration_config']}}
    # Only now, after settings are frozen, score the held-out test set.
    for split in ('train', 'validation', 'test'):
        current = samples[split]
        started = time.perf_counter()
        scores, durations = predict_prefixes(bundle, [s['inputs']['input_ids'] for s in current])
        elapsed = time.perf_counter() - started
        results[split] = evaluate(current, list(zip(scores.tolist(), durations.tolist())),
                                  bundle['switch_config']['threshold'], types)
        results[split]['batch_feature_and_predict_seconds'] = elapsed
        if split == 'test':
            with (args.out / 'test_predictions.jsonl').open('w') as file:
                for s, prob, d in zip(current, scores, durations):
                    file.write(json.dumps(dict(s['provenance'], probability=float(prob),
                        predicted_switch=int(prob >= bundle['switch_config']['threshold']),
                        predicted_duration=int(d), gold_switch=s['targets']['switch'],
                        gold_duration=s['targets']['duration'])) + '\n')
        print(split, {'macro_pair_type_switch_f1': results[split]['macro_pair_type_switch_f1'],
                      **results[split]['pooled']}, flush=True)
    write_json(args.out / 'results.json', results)
    old = json.loads((args.reference_root / 'runs/all-v3/results.json').read_text())
    comparison = []
    for name, ev in [('Original token counts', old['current_token']['test']),
                     ('Original prefix counts', old['causal_prefix']['test']),
                     ('Logistic Regression (new primary)', results['test'])]:
        comparison.append({'model': name, 'switch_macro_pair_type_f1': ev['macro_pair_type_switch_f1'],
                           **{k: ev['pooled'][k] for k in ['switch_f1','precision','recall','duration_macro_f1_oracle_switch','joint_duration_macro_f1']}})
    with (args.out / 'comparison.csv').open('w', newline='') as file:
        writer=csv.DictWriter(file, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    with (args.out / 'metrics.csv').open('w', newline='') as file:
        fields=['model','split','scope','switch_f1','precision','recall','duration_macro_f1_oracle_switch','joint_duration_macro_f1','positions']
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for split in ('train','validation','test'):
            ev=results[split]
            for scope, v in [('pooled',ev['pooled'])]+list(ev['by_pair'].items())+list(ev['by_cs_type'].items())+list(ev['by_pair_type'].items()):
                writer.writerow(dict(model='logistic_regression',split=split,scope=scope,**{k:v[k] for k in fields[3:]}))
    write_json(args.out / 'run_manifest.json', {
        'reference_root':str(args.reference_root.resolve()), 'data':audit,
        'python':platform.python_version(), 'sklearn':sklearn.__version__,
        'numpy':np.__version__, 'scipy':scipy.__version__, 'seed':6140,
        'source_sha256':{f:sha(Path(__file__).parent / f) for f in ['train.py','model.py','metrics.py']},
        'input_contract':'Only current and past token IDs. No labels, language IDs, cs_type, language pair, future tokens or full text as input.',
        'selection':'Switch model and threshold maximize validation equal-weight F1 across 42 pair/type groups. Duration model maximizes validation gold-switch duration macro F1. Ties keep earlier candidates.',
        'grid':{'windows':windows,'C':c_values,'class_weight':weights,'hashed_features':HASH_FEATURES},
        'test_caveat':'Existing test results were viewed during previous work; this is an exploratory comparison, not a new blind test.',
        'feature_caveat':'Hashing is stateless and can cause collisions. Existing subword segmentation is reused; this is not character-stream inference.'})
    print('Saved:',args.out,flush=True)

if __name__=='__main__':
    main()
