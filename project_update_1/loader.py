"""Pure Python batches for the current-token baseline or token-prefix experiments.

Only input_ids belong in the predictor. Targets and provenance stay outside it.
This operates on already segmented subwords, not a character-by-character stream.
"""
import argparse
from collections import defaultdict
from pipeline import read_jsonl, unique, validate_row


def iter_samples(data_path, split_path, split_name, context='current', max_context=None):
    if context not in ('current', 'prefix'):
        raise ValueError('context must be current or prefix')
    if max_context is not None and max_context <= 0:
        raise ValueError('max_context must be positive')
    rows = read_jsonl(data_path)
    by_id = unique(rows, 'sequence_id')
    splits = unique(read_jsonl(split_path), 'sequence_id')
    if len({(r['tokenizer_sha256'], r['neutral_policy'], r['end_policy']) for r in rows}) > 1:
        raise ValueError('Mixed tokenizer or labeling policies')
    if set(by_id) != set(splits):
        raise ValueError('Split IDs do not match data IDs')
    records, groups = defaultdict(set), defaultdict(set)
    for sid, entry in splits.items():
        if entry['record_id'] != by_id[sid]['record_id']:
            raise ValueError('Split record mismatch')
        if entry['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Invalid split name')
        records[entry['record_id']].add(entry['split'])
        groups[entry['group_id']].add(entry['split'])
    if any(len(v) != 1 for v in list(records.values()) + list(groups.values())):
        raise ValueError('A record or duplicate group crosses splits')
    for row in rows:
        validate_row(row)
        if splits[row['sequence_id']]['split'] != split_name:
            continue
        for t, valid in enumerate(row['switch_valid_mask']):
            if not valid:
                continue
            start = t if context == 'current' else (max(0, t+1-max_context) if max_context else 0)
            yield {'inputs': {'input_ids': row['input_ids'][start:t+1]},
                   'targets': {'switch': row['switch_labels'][t],
                               'duration': row['duration_labels'][t]},
                   'provenance': {'sequence_id': row['sequence_id'], 'position': t,
                                  'language_pair': row['language_pair']}}


def batches(samples, batch_size=32):
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    batch = []
    for item in samples:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='train')
    p.add_argument('--context', choices=['current', 'prefix'], default='current')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-context', type=int)
    a = p.parse_args()
    import json
    samples = iter_samples(a.data, a.splits, a.split, a.context, a.max_context)
    print(json.dumps(next(batches(samples, a.batch_size), []), ensure_ascii=False, indent=2))
