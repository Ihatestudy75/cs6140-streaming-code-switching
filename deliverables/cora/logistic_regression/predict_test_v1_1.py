"""One-time test inference from a frozen validation-selected LR bundle. No fitting or tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from model import vectorize_prefixes
from enhanced_features import vectorize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--combined', type=Path, required=True)
    p.add_argument('--max-len', type=int, required=True)
    a = p.parse_args()
    config = json.loads((a.model_run/'config.json').read_text())
    threshold = json.loads((a.model_run/'threshold.json').read_text())['threshold']
    for name in ('config.json', 'threshold.json'):
        assert json.loads((a.out/name).read_text()) == json.loads((a.model_run/name).read_text())
    assert a.max_len == config['max_len'] == 128
    for name, key in [('labeled_tokens.jsonl','data_sha256'),('splits.jsonl','split_sha256')]:
        assert sha(a.combined/name) == config[key], key
    for name in ('predictions_test.jsonl','predictions_test_meta.json','metrics_test.json','test_inference_receipt.json'):
        if (a.out/name).exists():
            raise FileExistsError(f'Refusing to repeat test inference: {a.out/name}')
    bundle = joblib.load(a.model_run/'logistic_model.joblib')
    assert bundle['feature_version'] == config['feature_version']
    assert bundle['switch_config'] == config['selected_switch']
    assert bundle['duration_config'] == config['selected_duration']
    assert threshold == bundle['switch_config']['threshold']
    for model in ('switch','duration'):
        assert bundle[model+'_model'].classes_.tolist() == ([0,1] if model == 'switch' else [0,1,2])
    for name in ('model.py', 'enhanced_features.py'):
        if name in config['source_sha256']:
            assert sha(Path(__file__).parent/name) == config['source_sha256'][name]
    splits = [json.loads(l) for l in (a.combined/'splits.jsonl').read_text().splitlines()]
    wanted = {r['sequence_id'] for r in splits if r['split']=='test'}
    rows=[]; seen=set(); total=dropped=dropped_sw=0
    with (a.combined/'labeled_tokens.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['sequence_id'] not in wanted: continue
            assert r['sequence_id'] not in seen
            seen.add(r['sequence_id'])
            assert r['tokenizer_sha256']==config['tokenizer_sha256']
            ids,ys,ds=r['input_ids'],r['switch_labels'],r['duration_labels']
            assert len(ids)==len(ys)==len(ds)
            total+=sum(y>=0 for y in ys)
            dropped+=sum(y>=0 for y in ys[a.max_len-2:])
            dropped_sw+=sum(y==1 for y in ys[a.max_len-2:])
            if 'vocab' in bundle:
                assert [bundle['vocab'][i] for i in ids]==r['tokens']
            for t in range(min(len(ids),a.max_len-2)):
                if ys[t]<0: continue
                rows.append(dict(sequence_id=r['sequence_id'],position=t,language_pair=r['language_pair'],
                                 y_switch=ys[t],y_duration=ds[t],prefix=ids[max(0,t-19):t+1]))
    assert seen==wanted and len(seen)==2148
    assert len(rows)==config['coverage']['test']['n_positions']
    def features(setting):
        prefixes=(r['prefix'] for r in rows)
        if 'vocab' in bundle:
            return vectorize(prefixes,setting['window'],bundle['vocab'],setting['mode'])
        return vectorize_prefixes(prefixes,setting['window'])
    # Reserve the run before either classifier is invoked; accidental reruns fail closed.
    receipt=dict(status='started',model_sha256=sha(a.model_run/'logistic_model.joblib'),
                 threshold=threshold,threshold_source='validation',fit_calls=0,
                 data_sha256=config['data_sha256'],split_sha256=config['split_sha256'])
    receipt_path=a.out/'test_inference_receipt.json'
    with receipt_path.open('x') as f: json.dump(receipt,f,indent=2)
    scores=np.round(bundle['switch_model'].predict_proba(features(bundle['switch_config']))[:,1],6)
    probs=bundle['duration_model'].predict_proba(features(bundle['duration_config']))
    with (a.out/'predictions_test.jsonl').open('x') as f:
        for row,score,prob in zip(rows,scores,probs):
            result={k:v for k,v in row.items() if k!='prefix'}
            result.update(p_switch=float(score),switch_pred=int(score>=threshold),
                          p_duration=[round(float(v),6) for v in prob],duration_pred=int(np.argmax(prob)))
            f.write(json.dumps(result)+'\n')
    meta=dict(run=a.out.name,backbone='logreg',model_id=config['model_id'],split='test',threshold=threshold,
              n_sequences=len(seen),n_positions=len(rows),n_true_switches=sum(r['y_switch'] for r in rows),
              max_len=a.max_len,n_valid_labels_total=total,n_dropped_by_truncation=dropped,
              n_switches_dropped_by_truncation=dropped_sw)
    (a.out/'predictions_test_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    receipt.update(status='complete',predictions_sha256=sha(a.out/'predictions_test.jsonl'),
                   inference_calls={'switch_predict_proba':1,'duration_predict_proba':1})
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(f'{a.out.name}: exported {len(rows)} test positions at frozen threshold {threshold}')


if __name__=='__main__':
    main()
