"""Rerun the frozen LR feature/grid baseline on data-v1.1; validation only.

Two independent classifiers predict switch and duration. This runner exports the
team prediction contract and uses the same pooled-F1 threshold grid as predict.py.
"""
import argparse
import gc
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from train import fit_lr, sha, read_jsonl, write_json
from model import vectorize_prefixes, HASH_FEATURES
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.metrics import f1_score

TOKENIZER_HASH = '78b44c32238e3b5e83d1deaa63c3fa1466ddf73547ce641a36f75180079ab45f'
THRESHOLDS = np.round(np.arange(.05, .951, .01), 2)


def load_combined(combined, max_len):
    if max_len < 3:
        raise ValueError('max_len must reserve CLS/SEP and at least one subword')
    entries = read_jsonl(combined / 'splits.jsonl')
    split = {r['sequence_id']: r for r in entries}
    manifest = json.loads((combined / 'split_manifest.json').read_text())
    assert len(entries) == len(split)
    assert manifest['seed'] == 6140
    assert dict(Counter(r['split'] for r in entries)) == manifest['counts']
    for key in ('record_id', 'group_id'):
        groups = defaultdict(set)
        for r in entries:
            groups[r[key]].add(r['split'])
        assert all(len(v) == 1 for v in groups.values()), key
    pilots = set(manifest['forced_train_records'])
    assert pilots <= {r['record_id'] for r in entries}
    assert all(r['split'] == 'train' for r in entries if r['record_id'] in pilots)
    samples = {'train': [], 'validation': []}
    coverage = {s: Counter() for s in ('train', 'validation', 'test')}
    seen = set()
    for r in read_jsonl(combined / 'labeled_tokens.jsonl'):
        sid = r['sequence_id']
        assert sid not in seen and sid in split
        seen.add(sid)
        entry = split[sid]
        assert entry['record_id'] == r['record_id']
        assert r['tokenizer_sha256'] == TOKENIZER_HASH
        assert r['neutral_policy'] == 'inherit_previous' and r['end_policy'] == 'observed'
        ids, ys, ds = r['input_ids'], r['switch_labels'], r['duration_labels']
        assert len(ids) == len(ys) == len(ds)
        assert r['switch_valid_mask'] == [y >= 0 for y in ys]
        assert r['duration_valid_mask'] == [d >= 0 for d in ds]
        s = entry['split']
        cov = coverage[s]
        cov['n_sequences'] += 1
        cov['n_valid_labels_total'] += sum(y >= 0 for y in ys)
        cov['n_dropped_by_truncation'] += sum(y >= 0 for y in ys[max_len-2:])
        # Test labels are audited for coverage, never fitted or evaluated.
        if s not in samples:
            continue
        for t in range(min(len(ids), max_len - 2)):
            y, d = ys[t], ds[t]
            if y < 0:
                continue
            assert y in (0, 1) and d in (-1, 0, 1, 2)
            assert d < 0 or y == 1
            samples[s].append(dict(sequence_id=sid, position=t,
                language_pair=r['language_pair'], y_switch=y, y_duration=d,
                prefix=ids[max(0, t-19):t+1]))
    assert seen == set(split)
    for cov in coverage.values():
        cov['n_positions'] = cov['n_valid_labels_total'] - cov['n_dropped_by_truncation']
    return samples, coverage


def threshold_choice(y, scores):
    values = [float(f1_score(y, scores >= t, zero_division=0)) for t in THRESHOLDS]
    idx = int(np.argmax(values))  # same ascending-grid tie handling as predict.py
    return float(THRESHOLDS[idx]), values[idx]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--combined', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-len', type=int, required=True)
    a = p.parse_args()
    samples, coverage = load_combined(a.combined, a.max_len)
    a.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    config = dict(backbone='logreg', model_id='causal-hashed-logistic-regression',
        data_tag='data-v1.1', data='mbert/combined/labeled_tokens.jsonl',
        splits='mbert/combined/splits.jsonl', tokenizer_sha256=TOKENIZER_HASH,
        data_sha256=sha(a.combined/'labeled_tokens.jsonl'),
        split_sha256=sha(a.combined/'splits.jsonl'),
        split_manifest_sha256=sha(a.combined/'split_manifest.json'),
        seed=6140, max_len=a.max_len, coverage=coverage,
        python=platform.python_version(), sklearn=sklearn.__version__,
        numpy=np.__version__, scipy=scipy.__version__,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        source_sha256={f: sha(Path(__file__).parent/f) for f in
                       ('train_v1_1.py', 'train.py', 'model.py', 'metrics.py')},
        grid=dict(windows=[5,20], C=[.1,1.], class_weight=[None,'balanced']),
        hash_features=HASH_FEATURES, feature_version='causal-position-bag-suffix-v1',
        switch_selection='validation pooled F1, thresholds 0.05..0.95 step 0.01; ties keep first',
        duration_selection='validation macro F1 at gold switches with known duration',
        input_contract='current and past token IDs only; no gold language IDs or future tokens')
    write_json(a.out/'config.json', config)
    print('Fixed split and frozen mBERT checks PASS:', dict(coverage), flush=True)
    tr, va = samples['train'], samples['validation']
    ty = np.array([r['y_switch'] for r in tr]); vy = np.array([r['y_switch'] for r in va])
    td = np.array([r['y_duration'] for r in tr]); vd = np.array([r['y_duration'] for r in va])
    tk, vk = td >= 0, vd >= 0
    assert set(td[tk]) == {0,1,2}
    best_sw = best_du = None
    best_sf = best_df = -1.
    history = []
    for window in (5,20):
        x = vectorize_prefixes((r['prefix'] for r in tr), window)
        xv = vectorize_prefixes((r['prefix'] for r in va), window)
        xd, xvd = x[tk], xv[vk]
        print('Features ready:', window, x.shape, 'nnz', x.nnz, flush=True)
        for c in (.1,1.):
            for weight in (None,'balanced'):
                sw, st = fit_lr(x,ty,c,weight)
                # predict.py exports six decimal places before threshold tuning.
                scores = np.round(sw.predict_proba(xv)[:,1],6)
                threshold,sf = threshold_choice(vy,scores)
                du,dt = fit_lr(xd,td[tk],c,weight)
                df = float(f1_score(vd[vk],du.predict(xvd),labels=[0,1,2],average='macro',zero_division=0))
                setting=dict(window=window,C=c,class_weight=weight)
                entry=dict(split='validation',**setting,threshold=threshold,switch_f1=sf,
                    duration_macro_f1=df,switch_fit_seconds=st,duration_fit_seconds=dt)
                history.append(entry)
                with (a.out/'train_log.jsonl').open('a') as log:
                    log.write(json.dumps(entry)+'\n')
                print(entry,flush=True)
                if sf > best_sf:
                    best_sf=sf;best_sw=(sw,dict(setting,threshold=threshold))
                if df > best_df:
                    best_df=df;best_du=(du,setting.copy())
        del x,xv,xd,xvd
        gc.collect()
    bundle=dict(switch_model=best_sw[0],switch_config=best_sw[1],
        duration_model=best_du[0],duration_config=best_du[1],hash_features=HASH_FEATURES,
        feature_version=config['feature_version'],duration_classes=['1-2','3-6','7+'])
    joblib.dump(bundle,a.out/'logistic_model.joblib',compress=3)
    xv=vectorize_prefixes((r['prefix'] for r in va),best_sw[1]['window'])
    scores=np.round(best_sw[0].predict_proba(xv)[:,1],6)
    if best_du[1]['window'] != best_sw[1]['window']:
        xv=vectorize_prefixes((r['prefix'] for r in va),best_du[1]['window'])
    assert best_du[0].classes_.tolist()==[0,1,2]
    probs=best_du[0].predict_proba(xv);threshold=best_sw[1]['threshold']
    with (a.out/'predictions_validation.jsonl').open('w') as out:
        for r,score,prob in zip(va,scores,probs):
            record={k:v for k,v in r.items() if k!='prefix'}
            record.update(p_switch=float(score),switch_pred=int(score>=threshold),
                p_duration=[round(float(v),6) for v in prob],duration_pred=int(np.argmax(prob)))
            out.write(json.dumps(record)+'\n')
    write_json(a.out/'threshold.json',dict(threshold=threshold,validation_switch_f1=best_sf,
        validation_switch_f1_at_0_5=float(f1_score(vy,scores>=.5)),grid=[.05,.95,.01]))
    write_json(a.out/'predictions_validation_meta.json',dict(run=a.out.name,backbone='logreg',
        model_id=config['model_id'],split='validation',threshold=threshold,
        **coverage['validation'],n_true_switches=int(vy.sum())))
    config.update(selected_switch=best_sw[1],selected_duration=best_du[1])
    write_json(a.out/'config.json',config)
    print('DONE. Validation switch F1:',best_sf,'duration macro F1:',best_df,flush=True)


if __name__ == '__main__':
    main()
