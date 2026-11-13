"""Turn a run's predictions into metrics_<split>.json (IO_STANDARD.md §5).

Usage (from repo root):
  python project_update_2/evaluate.py --run project_update_2/runs/<run_id> --split validation \
      --data D/labeled_tokens.jsonl
  python project_update_2/evaluate.py --run project_update_2/runs/<run_id> --split test \
      --data D/labeled_tokens.jsonl

Reads   <run>/predictions_<split>.jsonl and <run>/predictions_<split>_meta.json (for the threshold);
        --data (optional, local only): the labeled_tokens.jsonl the run used, for the boundary groups
Writes  <run>/metrics_<split>.json (aggregate numbers only, no text)

All switch/duration numbers come from score_group() in the frozen baseline file
deliverables/cora/logistic_regression/metrics.py, and the punctuation groups come from
boundary_kind() / boundary_group() in boundary_rules.py next to it. So the transformer runs
and the logistic regression baseline are scored by the same code.
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'deliverables', 'cora', 'logistic_regression'))
from boundary_rules import boundary_group, boundary_kind  # noqa: E402
from metrics import SCREENED_PAIRS, score_group  # noqa: E402

LID_GROUPS = {
    'script_rule': ['ar_en', 'zh_en', 'ja_en', 'yue_en', 'ko_en', 'ru_en'],
    'llm': ['es_en', 'vi_en', 'ms_en', 'hi_en'],
    'lingua': ['fr_en', 'de_en', 'it_en', 'philippines_en'],
}


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def to_item(row):
    """One prediction line -> the (sample, score, predicted duration) tuple score_group expects."""
    sample = {'targets': {'switch': row['y_switch'], 'duration': row['y_duration']}}
    return sample, row['p_switch'], row['duration_pred']


def boundary_groups(rows, data_path):
    """Boundary group of each prediction row, using the observed text through subword t only.

    Reads text and offsets from the local data file; nothing from them is written out.
    """
    wanted = {r['sequence_id'] for r in rows}
    raw = {}
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r['sequence_id'] in wanted:
                    raw[r['sequence_id']] = (r['text'], r['offsets'], r['switch_labels'])
    missing = wanted - set(raw)
    if missing:
        raise SystemExit(f'{len(missing)} predicted sequences are not in {data_path}, e.g. {sorted(missing)[0]}')
    groups = []
    for r in rows:
        text, offsets, labels = raw[r['sequence_id']]
        t = r['position']
        if labels[t] != r['y_switch']:
            raise SystemExit(f"{r['sequence_id']} position {t}: label in data does not match the prediction file")
        groups.append(boundary_group(boundary_kind(text, offsets, t)))
    return groups


def sigma(per_pair, pairs):
    """Population std (ddof=0) and mean of per-pair switch F1 over the given pairs.

    With fewer than 2 pairs a spread across pairs cannot be measured, so value is None
    (pstdev of a single number would be 0.0, which reads as "perfectly universal").
    """
    values = [per_pair[p]['switch_f1'] for p in pairs if p in per_pair]
    if not values:
        return None
    spread = statistics.pstdev(values) if len(values) >= 2 else None
    return {'value': spread, 'mean': statistics.fmean(values), 'n_pairs': len(values)}


def fmt(s):
    return 'n/a' if s is None or s['value'] is None else f"{s['value']:.4f} over {s['n_pairs']} pairs"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True, help='run directory with predictions_<split>.jsonl')
    p.add_argument('--split', choices=['validation', 'test'], required=True)
    p.add_argument('--data', help='local labeled_tokens.jsonl used by the run; enables by_boundary')
    a = p.parse_args()

    # 1. Read predictions and the threshold they were made with.
    rows = read_jsonl(os.path.join(a.run, f'predictions_{a.split}.jsonl'))
    with open(os.path.join(a.run, f'predictions_{a.split}_meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    threshold = meta['threshold']

    # 2. Pooled: all positions together.
    pooled = score_group([to_item(r) for r in rows], threshold)

    # 3. Per pair and per LID group.
    by_pair_items = defaultdict(list)
    for r in rows:
        by_pair_items[r['language_pair']].append(to_item(r))
    per_pair = {pair: score_group(items, threshold) for pair, items in sorted(by_pair_items.items())}
    by_lid_method = {
        group: score_group([it for pair in pairs for it in by_pair_items.get(pair, [])], threshold)
        for group, pairs in LID_GROUPS.items() if any(pair in by_pair_items for pair in pairs)
    }

    # 3b. Punctuation vs within-clause proxy (needs the local data file for text and offsets).
    by_boundary = None
    if a.data:
        items_by_group = defaultdict(list)
        for r, g in zip(rows, boundary_groups(rows, a.data)):
            items_by_group[g].append(to_item(r))
        by_boundary = {g: score_group(items, threshold) for g, items in sorted(items_by_group.items())}
    else:
        print('No --data given: by_boundary is skipped.')

    # 4. Universality: spread of per-pair switch F1.
    all_pairs = sorted(per_pair)
    core10 = [pair for pair in all_pairs if pair not in SCREENED_PAIRS]
    universality = {
        'metric': 'per_pair.switch_f1', 'ddof': 0,
        'sigma_all': sigma(per_pair, all_pairs),
        'sigma_core10': sigma(per_pair, core10),
        'sigma_by_lid_method': {g: sigma(per_pair, pairs) for g, pairs in LID_GROUPS.items()},
    }

    # 5. Write metrics_<split>.json.
    known = {pair for pairs in LID_GROUPS.values() for pair in pairs}
    unknown = sorted(set(all_pairs) - known)
    if unknown:
        print(f'WARNING: pairs not in any LID group: {unknown}')
    absent = sorted(known - set(all_pairs))
    if absent:
        print(f'WARNING: {len(absent)} of 14 pairs have no {a.split} positions: {absent}')
    out = {'io_version': '0.3', 'run': meta.get('run', a.run), 'backbone': meta.get('backbone'),
           'split': a.split, 'threshold': threshold, 'n_positions': len(rows),
           'pooled': pooled, 'per_pair': per_pair, 'by_lid_method': by_lid_method, 'by_boundary': by_boundary,
           'universality': universality}
    path = os.path.join(a.run, f'metrics_{a.split}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'{a.split}: switch F1 {pooled["switch_f1"]:.4f} (P {pooled["precision"]:.4f} R {pooled["recall"]:.4f}) '
          f'| duration acc {pooled["duration_accuracy_oracle_switch"]} macro F1 {pooled["duration_macro_f1_oracle_switch"]:.4f} '
          f'| joint {pooled["joint_duration_macro_f1"]:.4f} | sigma_all {fmt(universality["sigma_all"])}')
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
