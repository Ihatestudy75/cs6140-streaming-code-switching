"""Write per-position predictions of a trained run, and choose the switch threshold on validation.

Usage (from repo root):
  # 1. Validation: write predictions and choose the threshold that maximizes switch F1.
  .venv-train/Scripts/python.exe project_update_2/predict.py --run project_update_2/runs/mbert \
      --data D/labeled_tokens.jsonl --splits D/splits.jsonl --split validation --tune-threshold
  # 2. Test (run once): reuses the threshold saved by step 1.
  .venv-train/Scripts/python.exe project_update_2/predict.py --run project_update_2/runs/mbert \
      --data D/labeled_tokens.jsonl --splits D/splits.jsonl --split test

Outputs in --run:
  predictions_<split>.jsonl       one line per position with a valid switch label
  predictions_<split>_meta.json   run, split, threshold, counts, and coverage:
                                  n_valid_labels_total (valid switch labels before truncation),
                                  n_dropped_by_truncation (= n_valid_labels_total - n_positions),
                                  n_switches_dropped_by_truncation
  threshold.json                  written only by --tune-threshold on validation

Each prediction line:
  sequence_id, position (index of x_t in the original subword list, 0-based),
  language_pair, p_switch = P(LID(x_{t+1}) != LID(x_t)), switch_pred (p_switch >= threshold),
  p_duration = [P(Small), P(Medium), P(Large)], duration_pred (argmax),
  y_switch, y_duration (-1 = no duration label).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Collator, SequenceDataset, load_rows  # noqa: E402
from model import BACKBONES, DualHeadModel  # noqa: E402
from provenance import check_tokenizer  # noqa: E402

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            sw, dur = model(ids, mask)
        p_sw = sw.float().softmax(-1)[..., 1].cpu()
        p_dur = dur.float().softmax(-1).cpu()
        for b, n in enumerate(batch['n_subwords']):
            for i in range(n):
                y_sw = int(batch['switch_labels'][b, i + 1])     # model position i+1 = subword i
                if y_sw == -1:
                    continue
                pd = p_dur[b, i + 1].tolist()
                out.append({'sequence_id': batch['sequence_id'][b], 'position': i,
                            'language_pair': batch['language_pair'][b],
                            'p_switch': round(float(p_sw[b, i + 1]), 6),
                            'p_duration': [round(x, 6) for x in pd],
                            'duration_pred': int(np.argmax(pd)),
                            'y_switch': y_sw, 'y_duration': int(batch['duration_labels'][b, i + 1])})
    return out


def switch_f1(preds, threshold):
    y = np.array([p['y_switch'] for p in preds])
    yhat = np.array([p['p_switch'] >= threshold for p in preds], dtype=int)
    return f1_score(y, yhat, zero_division=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True, help='run directory with best.pt')
    p.add_argument('--data', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--split', choices=['validation', 'test'], required=True)
    p.add_argument('--tune-threshold', action='store_true', help='validation only: choose the threshold')
    p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()
    if a.tune_threshold and a.split != 'validation':
        raise SystemExit('The threshold may only be chosen on the validation split.')
    threshold_file = os.path.join(a.run, 'threshold.json')
    if not a.tune_threshold and not os.path.exists(threshold_file):
        raise SystemExit('No threshold.json: run --split validation --tune-threshold first.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(os.path.join(a.run, 'best.pt'), map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    tokenizer = AutoTokenizer.from_pretrained(BACKBONES[cfg['backbone']])
    check_tokenizer(a.data, tokenizer, cfg['backbone'])    # stop before loading the model weights
    collate = Collator.from_tokenizer(tokenizer, cfg['max_len'])
    model = DualHeadModel(cfg['backbone'], dropout=0.0)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    rows = load_rows(a.data, a.splits, a.split)
    preds = predict(model, DataLoader(SequenceDataset(rows), batch_size=a.batch_size, collate_fn=collate), device)

    if a.tune_threshold:
        scores = {float(t): switch_f1(preds, t) for t in THRESHOLDS}
        best = max(scores, key=scores.get)
        with open(threshold_file, 'w', encoding='utf-8') as f:
            json.dump({'threshold': best, 'validation_switch_f1': scores[best],
                       'validation_switch_f1_at_0.5': scores[0.5], 'grid': [0.05, 0.95, 0.01]}, f, indent=2)
        print(f'threshold 0.50 -> F1 {scores[0.5]:.4f}')
        print(f'best threshold {best:.2f} -> F1 {scores[best]:.4f} (saved to {threshold_file})')
    threshold = json.load(open(threshold_file, encoding='utf-8'))['threshold']
    for r in preds:
        r['switch_pred'] = int(r['p_switch'] >= threshold)

    path = os.path.join(a.run, f'predictions_{a.split}.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        for r in preds:
            f.write(json.dumps(r) + '\n')
    # Coverage: positions cut by max_len get no prediction, so count them here.
    n_valid_total = sum(sum(y != -1 for y in r['switch_labels']) for r in rows)
    n_switches_total = sum(sum(y == 1 for y in r['switch_labels']) for r in rows)
    n_true_switches = sum(r['y_switch'] for r in preds)
    meta = {'run': a.run, 'backbone': cfg['backbone'], 'model_id': cfg['model_id'], 'split': a.split,
            'threshold': threshold, 'n_sequences': len(rows), 'n_positions': len(preds),
            'n_true_switches': n_true_switches, 'checkpoint_epoch': ckpt['epoch'],
            'max_len': cfg['max_len'], 'n_valid_labels_total': n_valid_total,
            'n_dropped_by_truncation': n_valid_total - len(preds),
            'n_switches_dropped_by_truncation': n_switches_total - n_true_switches}
    with open(os.path.join(a.run, f'predictions_{a.split}_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    # Quick sanity summary. The official metrics come from the team evaluation code.
    by_pair = defaultdict(list)
    for r in preds:
        by_pair[r['language_pair']].append(r)
    dur = [r for r in preds if r['y_duration'] != -1]
    print(f'{a.split}: {len(preds)} positions, threshold {threshold:.2f}, '
          f'switch F1 {switch_f1(preds, threshold):.4f}, '
          f'duration acc {np.mean([r["duration_pred"] == r["y_duration"] for r in dur]):.4f} ({len(dur)} switches)')
    print(f'coverage: {len(preds)} of {n_valid_total} valid labels scored, '
          f'{meta["n_dropped_by_truncation"]} dropped by max_len {cfg["max_len"]}')
    for pair, items in sorted(by_pair.items()):
        print(f'  {pair:8s} F1 {switch_f1(items, threshold):.4f}  ({len(items)} positions)')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
