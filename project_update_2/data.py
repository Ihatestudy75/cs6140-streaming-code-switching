"""Sequence-level data loading and batch collation for the dual-head causal model.

One training example is one whole sequence. The causal mask (see model.py) makes
every position t see only x_1..x_t, so one forward pass scores all positions at once.

Label alignment
  pipeline.py tokenizes with add_special_tokens=False, so row['input_ids'][i] is
  subword i and row['switch_labels'][i] already refers to subword i+1.
  The collator adds a start token in front and an end token at the back:
      model position : 0      1      2     ...  n      n+1    n+2 ...
      input          : CLS    x_0    x_1   ...  x_n-1  SEP    PAD ...
      labels         : -1     y_0    y_1   ...  y_n-1  -1     -1  ...
  So the labels shift right by one position. CLS, SEP and PAD positions get -1,
  which the loss ignores (ignore_index=-1).

Truncation
  A sequence longer than max_len keeps only its first max_len - 2 subwords.
  Each kept label is still correct: the label at t only depends on the true
  sentence, and the model input at t only uses x_1..x_t.

This is the team collator. It was reviewed against the alignment checklist in
IO_STANDARD.md (special token IDs from the tokenizer, -1 at CLS/SEP/PAD, labels
shifted right by one, truncation applied to inputs and labels together).
tests/check_collator.py checks the alignment and the padding behaviour.
"""
import argparse
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'project_update_1'))
from pipeline import read_jsonl, unique, validate_row  # noqa: E402

SPLITS = ('train', 'validation', 'test')


def load_rows(data_path, split_path, split_name):
    """Return the rows of one split, with the same safety checks as loader.py."""
    if split_name not in SPLITS:
        raise ValueError(f'split must be one of {SPLITS}')
    rows = read_jsonl(data_path)
    by_id = unique(rows, 'sequence_id')
    splits = unique(read_jsonl(split_path), 'sequence_id')
    if len({(r['tokenizer_sha256'], r['neutral_policy'], r['end_policy']) for r in rows}) > 1:
        raise ValueError('Mixed tokenizer or labeling policies')
    if set(by_id) != set(splits):
        raise ValueError('Split IDs do not match data IDs')
    out = []
    for row in rows:
        if splits[row['sequence_id']]['split'] != split_name:
            continue
        validate_row(row)
        # Keep only the fields the model and the evaluation need. No raw text.
        out.append({k: row[k] for k in ('sequence_id', 'language_pair', 'input_ids',
                                         'switch_labels', 'duration_labels')})
    return out


class SequenceDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    """Turn a list of rows into padded tensors for DualHeadModel."""

    def __init__(self, cls_id, sep_id, pad_id, max_len=256):
        if max_len < 3:
            raise ValueError('max_len must leave room for CLS, SEP and one subword')
        self.cls_id, self.sep_id, self.pad_id, self.max_len = cls_id, sep_id, pad_id, max_len

    @classmethod
    def from_tokenizer(cls, tokenizer, max_len=256):
        # Read special token IDs from the tokenizer; they differ between models
        # (mBERT: CLS=101, SEP=102, PAD=0; XLM-R: <s>=0, </s>=2, <pad>=1).
        return cls(tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, max_len)

    def __call__(self, rows):
        keep = [min(len(r['input_ids']), self.max_len - 2) for r in rows]
        T = max(keep) + 2
        B = len(rows)
        input_ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, T), dtype=torch.long)
        switch = torch.full((B, T), -1, dtype=torch.long)
        duration = torch.full((B, T), -1, dtype=torch.long)
        for b, (r, n) in enumerate(zip(rows, keep)):
            input_ids[b, 0] = self.cls_id
            input_ids[b, 1:n + 1] = torch.tensor(r['input_ids'][:n])
            input_ids[b, n + 1] = self.sep_id
            attention_mask[b, :n + 2] = 1                       # CLS + subwords + SEP are real
            switch[b, 1:n + 1] = torch.tensor(r['switch_labels'][:n])      # shift right by 1
            duration[b, 1:n + 1] = torch.tensor(r['duration_labels'][:n])
        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'switch_labels': switch, 'duration_labels': duration,
                'sequence_id': [r['sequence_id'] for r in rows],
                'language_pair': [r['language_pair'] for r in rows],
                'n_subwords': keep}


def length_stats(rows, max_len):
    """Per pair: subword-length percentiles and how many sequences/labels max_len cuts."""
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r['language_pair']].append(r)
    by_pair['ALL'] = rows
    table = []
    for pair, items in sorted(by_pair.items(), key=lambda kv: (kv[0] == 'ALL', kv[0])):
        lengths = sorted(len(r['input_ids']) for r in items)
        pct = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))]  # noqa: E731
        cut = sum(n > max_len - 2 for n in lengths)
        valid = sum(v >= 0 for r in items for v in r['switch_labels'])
        lost = sum(v >= 0 for r in items for v in r['switch_labels'][max_len - 2:])
        table.append((pair, len(items), pct(.5), pct(.95), pct(.99), lengths[-1], cut, lost / max(valid, 1)))
    return table


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Print sequence-length statistics for choosing max_len.')
    p.add_argument('--data', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--max-len', type=int, nargs='+', default=[128, 192, 256])
    a = p.parse_args()
    rows = [r for s in SPLITS for r in load_rows(a.data, a.splits, s)]
    for m in a.max_len:
        print(f'\nmax_len = {m} (= {m - 2} subwords + CLS + SEP)')
        print(f'{"pair":8s} {"seqs":>5s} {"p50":>5s} {"p95":>5s} {"p99":>5s} {"max":>5s} {"cut":>5s} {"labels lost":>12s}')
        for pair, n, p50, p95, p99, mx, cut, lost in length_stats(rows, m):
            print(f'{pair:8s} {n:5d} {p50:5d} {p95:5d} {p99:5d} {mx:5d} {cut:5d} {lost:11.2%}')
