"""Run the token baseline and prefix model locally.

Use a new output folder: python3 run.py --out runs/reproduction
Source data is read-only. No network access is used.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE / 'update1_data_standard'))
import pipeline as p
from loader import iter_samples

DELIVERIES = [(pair, 'shaoyu') for pair in ('yue_en', 'ko_en', 'ru_en', 'ms_en')] + [(pair, 'yulun') for pair in ('ar_en', 'hi_en', 'zh_en', 'ja_en', 'es_en', 'vi_en')]
DELIVERIES += [(pair, 'yufan') for pair in ('fr_en', 'de_en', 'it_en', 'philippines_en')]
PAIRS = [pair for pair, owner in DELIVERIES]
SCREENED_PAIRS = {'fr_en', 'de_en', 'it_en', 'philippines_en'}
SOURCE_COMMITS = {'shaoyu': '440cbb032b57b010f909ac30d6deb223f7e642dc', 'yulun': '76a524ddb25ee21ca09a1c07928fbb1ec8c22543', 'yufan': 'c4061b60567a5aa6544874a8e471912da87e934e'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    report, types, pilots, all_rows, inputs = {}, {}, set(), [], []
    inventory = json.loads((SOURCE / 'deliverables/yufan/file_inventory_yufan.json').read_text())
    for item in inventory['files']:
        path = SOURCE / 'deliverables/yufan' / item['path']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    for pair, owner in DELIVERIES:
        base = SOURCE / 'deliverables' / owner / pair
        inp = base / ('tokenized/labeled_tokens_' + owner + '.jsonl')
        inputs.append(str(inp))
        rows = p.read_jsonl(inp)
        texts_path = base / ('extraction/texts_' + owner + '.jsonl')
        lid_path = base / ('annotation/lid_' + owner + '.jsonl')
        texts = p.unique(p.read_jsonl(texts_path), 'sequence_id')
        lids = p.unique(p.read_jsonl(lid_path), 'sequence_id')
        cs = p.unique(p.read_jsonl(base / ('cs_type_' + owner + '.jsonl')), 'sequence_id')
        assert set(texts) == set(lids) == set(cs) == set(p.unique(rows, 'sequence_id'))
        manifest = json.loads((base / ('tokenized/manifest_' + owner + '.json')).read_text())
        if owner == 'yufan':
            outputs = manifest['output_sha256']
            for path in [texts_path, lid_path, inp, base / 'cs_type_yufan.jsonl']:
                assert sha(path) == outputs[path.name]
            generation = manifest['source_manifest']['source_tokenization_run']
        else:
            assert sha(texts_path) == manifest['texts_sha256']
            assert sha(lid_path) == manifest['lid_sha256']
            generation = manifest
        # Accept LF and CRLF line endings in the script hash.
        script_path = SOURCE / 'update1_data_standard/pipeline.py'
        script_hashes = {sha(script_path), hashlib.sha256(script_path.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()}
        assert generation['pipeline_sha256'] in script_hashes
        pilot = set((base / ('annotation/pilot_record_ids_' + owner + '.txt')).read_text().split())
        if owner != 'yufan':
            assert len(pilot) == 20
        else:
            assert pilot == set(json.loads((base / 'annotation/development_record_manifest_yufan.json').read_text())['present_in_delivery'])
        pilots.update(pilot)
        for row in rows:
            p.validate_row(row)
            sid = row['sequence_id']
            assert row['language_pair'] == pair
            assert row['text'] == texts[sid]['text']
            chars, spans = p.validate_lid(texts[sid], lids[sid])
            langs, notes = p.align(row['text'], chars, row['offsets'], row['input_ids'], 100, row['neutral_policy'])
            assert langs == row['language_ids'] and notes == row['alignment_notes']
            assert spans == row['language_spans']
            assert row['review_status'] == lids[sid]['review_status']
            if owner == 'yufan':
                assert row['review_status'] == 'unreviewed'
            else:
                assert (row['review_status'] == 'reviewed') == (row['record_id'] in pilot)
            assert cs[sid]['record_id'] == row['record_id']
            # Yufan's type map uses IDs without a language_pair field.
            if 'language_pair' in cs[sid]:
                assert cs[sid]['language_pair'] == pair
            assert cs[sid]['cs_type'] in ('Intersentential', 'Intrasentential', 'Extra-sentential / Tag switching')
            types[sid] = cs[sid]['cs_type']
        records = {r['record_id'] for r in rows}
        assert pilot <= records
        if owner == 'yufan':
            sample = json.loads((base / 'extraction/sampling_manifest_yufan.json').read_text())
            assert len(records) == sample['sampled_records'] == 500
            assert records == set(sample['selected_record_ids'])
            assert len(rows) == sample['sampled_sequences']
            templates = p.unique(p.read_jsonl(base / 'extraction/lid.template_yufan.jsonl'), 'sequence_id')
            assert set(templates) == set(texts)
            assert all(templates[k]['text_sha256'] == texts[k]['text_sha256'] for k in templates)
        stats = p.summary(rows)
        assert stats == json.loads((base / ('tokenized/stats_' + owner + '.json')).read_text())
        issues = p.read_jsonl(base / ('extraction/extract_issues_' + owner + '.jsonl'))
        report[pair] = dict(stats, owner=owner,
                            sampling_population=('screened_retained_set' if owner == 'yufan' else 'full_raw_records'),
                            extraction_issue_scope=('full_extraction_not_sample' if owner == 'yufan' else 'sample'),
                            records=len({r['record_id'] for r in rows}),
                            pilot_records=len(pilot), extraction_issues=dict(Counter(r['severity'] for r in issues)),
                            extraction_errors=[r for r in issues if r['severity'] == 'error'],
                            tokenized_sha256=sha(inp), manifest=manifest,
                            total_tokens=sum(len(r['input_ids']) for r in rows),
                            unk_token_count=sum(r['input_ids'].count(100) for r in rows))
        all_rows.extend(rows)
    assert len({(r['tokenizer_sha256'], r['neutral_policy'], r['end_policy']) for r in all_rows}) == 1
    return report, types, pilots, all_rows, inputs


def features(prefix, order):
    return tuple(prefix[-order:])


class Counts:
    """Predict from token IDs. Use metadata only for scoring."""
    def __init__(self, samples, order=1, min_count=1):
        self.order, self.min_count = order, min_count
        self.tables = [defaultdict(lambda: [0, 0, [0, 0, 0]]) for _ in range(order)]
        total, positive, durations = 0, 0, [0, 0, 0]
        for s in samples:
            ids, y, d = s['inputs']['input_ids'], s['targets']['switch'], s['targets']['duration']
            total += 1
            positive += y
            if d >= 0:
                durations[d] += 1
            for n in range(1, min(order, len(ids)) + 1):
                c = self.tables[n-1][features(ids, n)]
                c[0] += 1
                c[1] += y
                if d >= 0:
                    c[2][d] += 1
        self.prior = positive / total
        self.duration_prior = max(range(3), key=lambda d: durations[d])

    def predict(self, input_ids):
        score, duration = self.prior, self.duration_prior
        score_found, duration_found = False, False
        for n in range(min(self.order, len(input_ids)), 0, -1):
            c = self.tables[n-1].get(features(input_ids, n))
            if c is None:
                continue
            if not score_found and c[0] >= self.min_count:
                score, score_found = c[1] / c[0], True
            if not duration_found and sum(c[2]) >= self.min_count:
                duration = max(range(3), key=lambda d: c[2][d])
                duration_found = True
        return score, duration


def group(s, types):
    return (s['provenance']['language_pair'], types[s['provenance']['sequence_id']])


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def choose_threshold(samples, scores, types):
    """Choose the threshold with the best validation macro F1.

    On a tie, keep the higher threshold. Group equal scores together.
    """
    totals = Counter(group(s, types) for s in samples if s['targets']['switch'])
    groups = sorted({group(s, types) for s in samples})
    counts = {g: [0, 0, totals[g]] for g in groups}
    best = (0.0, math.nextafter(max(scores), math.inf))
    ranked = sorted(zip(scores, samples), key=lambda x: x[0], reverse=True)
    i = 0
    while i < len(ranked):
        score = ranked[i][0]
        while i < len(ranked) and ranked[i][0] == score:
            s = ranked[i][1]
            c = counts[group(s, types)]
            if s['targets']['switch']:
                c[0] += 1
                c[2] -= 1
            else:
                c[1] += 1
            i += 1
        value = sum(f1(*c) for c in counts.values()) / len(groups)
        if value > best[0]:
            best = value, score
    return {'threshold': best[1], 'validation_macro_pair_type_switch_f1': best[0]}


def score_group(items, threshold):
    tp = fp = fn = tn = correct_duration = duration_n = 0
    cm = [[0] * 3 for _ in range(3)]
    # Joint scoring counts missed and false switches and wrong length classes.
    # Skip gold switches whose duration is unknown.
    joint_tp, joint_fp, joint_fn = [0]*3, [0]*3, [0]*3
    for s, score, pred_d in items:
        y, d = s['targets']['switch'], s['targets']['duration']
        pred = score >= threshold
        tp += int(pred and y == 1)
        fp += int(pred and y == 0)
        fn += int(not pred and y == 1)
        tn += int(not pred and y == 0)
        if d >= 0:
            duration_n += 1
            correct_duration += int(d == pred_d)
            cm[d][pred_d] += 1
        if y == 1 and d < 0:
            continue
        if pred:
            if d == pred_d:
                joint_tp[pred_d] += 1
            else:
                joint_fp[pred_d] += 1
        if d >= 0 and (not pred or d != pred_d):
            joint_fn[d] += 1
    duration_f1 = [f1(cm[k][k], sum(cm[j][k] for j in range(3) if j != k), sum(cm[k][j] for j in range(3) if j != k)) for k in range(3)]
    return {'positions': len(items), 'gold_switches': tp+fn, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp/(tp+fp) if tp+fp else 0.0, 'recall': tp/(tp+fn) if tp+fn else 0.0,
            'switch_f1': f1(tp, fp, fn), 'duration_support': duration_n,
            'duration_accuracy_oracle_switch': correct_duration/duration_n if duration_n else None,
            'duration_macro_f1_oracle_switch': sum(duration_f1)/3,
            'duration_confusion_matrix': cm,
            'joint_duration_macro_f1': sum(f1(joint_tp[k], joint_fp[k], joint_fn[k]) for k in range(3))/3}


def evaluate(samples, predictions, threshold, types):
    cells, pairs, cstypes = defaultdict(list), defaultdict(list), defaultdict(list)
    items = []
    for s, (prob, d) in zip(samples, predictions):
        item = (s, prob, d)
        pair, typ = group(s, types)
        items.append(item)
        cells[pair + ' | ' + typ].append(item)
        pairs[pair].append(item)
        cstypes[typ].append(item)
    by_cell = {k: score_group(v, threshold) for k, v in sorted(cells.items())}
    cohorts = {}
    for name, screened in [('full_record_sample', False), ('screened_retained_sample', True)]:
        chosen = [item for item in items if (item[0]['provenance']['language_pair'] in SCREENED_PAIRS) == screened]
        chosen_cells = [v for k, v in by_cell.items() if (k.split(' | ')[0] in SCREENED_PAIRS) == screened]
        if chosen:
            cohorts[name] = {'pooled': score_group(chosen, threshold), 'macro_pair_type_switch_f1': sum(v['switch_f1'] for v in chosen_cells)/len(chosen_cells)}
    return {'pooled': score_group(items, threshold), 'by_pair_type': by_cell, 'by_sampling_cohort': cohorts,
            'macro_pair_type_switch_f1': sum(v['switch_f1'] for v in by_cell.values()) / len(by_cell),
            'by_pair': {k: score_group(v, threshold) for k, v in sorted(pairs.items())},
            'by_cs_type': {k: score_group(v, threshold) for k, v in sorted(cstypes.items())}}


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--out', type=Path, default=ROOT / 'runs/v1')
    args = a.parse_args()
    out = p.new_dir(args.out.resolve())
    audit_report, types, pilots, rows, inputs = audit()
    p.write_json(out / 'audit.json', audit_report)
    (out / 'all_pilot_record_ids.txt').write_text('\n'.join(sorted(pilots)) + '\n')
    p.split(SimpleNamespace(inputs=inputs, seed=6140, train_records=str(out / 'all_pilot_record_ids.txt'), out=str(out / 'combined')))
    # Save input hashes with relative paths.
    split_manifest_path = out / 'combined/split_manifest.json'
    split_manifest = json.loads(split_manifest_path.read_text())
    split_manifest['inputs'] = {Path(k).relative_to(ROOT).as_posix(): v
                                for k, v in split_manifest['inputs'].items()}
    p.write_json(split_manifest_path, split_manifest)
    data, splits = out / 'combined/labeled_tokens.jsonl', out / 'combined/splits.jsonl'
    entries = p.read_jsonl(splits)
    split_by_sid = {r['sequence_id']: r['split'] for r in entries}
    sets = {k: {r['record_id'] for r in entries if r['split'] == k} for k in ('train', 'validation', 'test')}
    assert pilots <= sets['train'] and not (sets['train'] & sets['test'] or sets['train'] & sets['validation'] or sets['test'] & sets['validation'])
    exact = defaultdict(set)
    for r in rows:
        exact[' '.join(unicodedata.normalize('NFC', r['text']).split())].add(split_by_sid[r['sequence_id']])
    assert all(len(v) == 1 for v in exact.values())
    samples = {s: list(iter_samples(data, splits, s, context='prefix', max_context=5)) for s in sets}
    print('Audit and split passed.', flush=True)
    # Use validation data to choose settings. Keep test data out.
    candidates = []
    baseline_model = best_prefix_model = None
    best_prefix_index = None
    for order, min_count in [(1, 1)] + [(n, c) for n in (2, 3, 5) for c in (1, 5, 20)]:
        model = Counts(samples['train'], order, min_count)
        preds = [model.predict(s['inputs']['input_ids']) for s in samples['validation']]
        choice = choose_threshold(samples['validation'], [v[0] for v in preds], types)
        candidate = dict(order=order, min_count=min_count, **choice)
        candidates.append(candidate)
        index = len(candidates) - 1
        if order == 1:
            baseline_model = model
        elif best_prefix_index is None or choice['validation_macro_pair_type_switch_f1'] > candidates[best_prefix_index]['validation_macro_pair_type_switch_f1']:
            best_prefix_index, best_prefix_model = index, model
        print(candidate, flush=True)
    # On a tie, keep the shorter history, then the earlier setting.
    best_prefix = best_prefix_index
    selected = {'current_token': 0, 'causal_prefix': best_prefix}
    results = {}
    for name, index in selected.items():
        config = candidates[index]
        model = baseline_model if name == 'current_token' else best_prefix_model
        results[name] = {'config': config}
        for split in ('validation', 'test'):
            preds = [model.predict(s['inputs']['input_ids']) for s in samples[split]]
            results[name][split] = evaluate(samples[split], preds, config['threshold'], types)
            if split == 'test':
                p.write_jsonl(out / (name + '_test_predictions.jsonl'),
                              [dict(s['provenance'], probability=prob, predicted_switch=int(prob >= config['threshold']), predicted_duration=d,
                                    gold_switch=s['targets']['switch'], gold_duration=s['targets']['duration'])
                               for s, (prob, d) in zip(samples[split], preds)])
        # Save counts for later predictions.
        p.write_json(out / (name + '_model.json'), {'config': config, 'prior': model.prior, 'duration_prior': model.duration_prior,
                     'tables': [[[list(k), v] for k, v in sorted(tab.items())] for tab in model.tables]})
    for name, prob in [('always_no_switch', 0.0), ('always_switch', 1.0)]:
        results[name] = {'test': evaluate(samples['test'], [(prob, baseline_model.duration_prior)] * len(samples['test']), 0.5, types)}
    p.write_json(out / 'results.json', results)
    p.write_json(out / 'validation_search.json', candidates)
    p.write_json(out / 'run_manifest.json', {'source_branches': ['data/shaoyu', 'data/yulun', 'data/yufan'],
                 'source_commits': SOURCE_COMMITS, 'python': platform.python_version(), 'runner_sha256': sha(__file__),
                 'split_sha256': sha(splits), 'data_sha256': sha(data), 'seed': 6140,
                 'evaluation': 'Pseudo-label agreement. Reviewed pilot records are model-assisted and training-only.',
                 'selection_metric': 'Validation positive switch F1 averaged equally over 42 language-pair x CS-type cells; mixed sampling cohorts, exploratory.',
                 'input_contract': 'Token IDs at positions <=t only. No language IDs, cs_type, labels, text or future token at inference.',
                 'checks': {'pilot_train_only': True, 'no_record_overlap': True, 'no_normalized_exact_text_overlap': True},
                 'not_verified': ['Raw CSV absent', 'Independent human LID quality', 'Near duplicates/template leakage', 'Retokenization from frozen tokenizer assets (assets absent)']})
    with (out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'split', 'scope', 'switch_f1', 'precision', 'recall', 'duration_macro_f1_oracle_switch', 'joint_duration_macro_f1', 'positions'])
        for name, res in results.items():
            for split in ('validation', 'test'):
                if split not in res:
                    continue
                ev = res[split]
                for scope, v in [('pooled', ev['pooled'])] + list(ev['by_pair'].items()) + list(ev['by_cs_type'].items()) + list(ev['by_pair_type'].items()):
                    writer.writerow([name, split, scope] + [v[k] for k in ('switch_f1', 'precision', 'recall', 'duration_macro_f1_oracle_switch', 'joint_duration_macro_f1', 'positions')])
    print('Saved:', out, flush=True)


if __name__ == '__main__':
    main()
