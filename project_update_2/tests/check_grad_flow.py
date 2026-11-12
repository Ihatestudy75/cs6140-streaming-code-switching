"""Gradient-flow and loss checks.

1. Backward on L_duration alone: the backbone and the duration head get gradients,
   the switch head gets none. Same for L_switch alone with the heads swapped.
   This shows that both tasks train the shared backbone.
2. A batch with no valid duration label gives L_duration = 0, not NaN.
3. Positions labeled -1 do not change the loss (ignore_index=-1 works).

Run from repo root:
  .venv-train/Scripts/python.exe project_update_2/tests/check_grad_flow.py \
      --data <combined>/labeled_tokens.jsonl --splits <combined>/splits.jsonl
"""
import argparse
import math
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Collator, load_rows  # noqa: E402
from losses import multitask_loss  # noqa: E402
from model import BACKBONES, DualHeadModel  # noqa: E402


def grad_norm(module):
    grads = [p.grad.norm() for p in module.parameters() if p.grad is not None]
    return torch.stack(grads).norm().item() if grads else 0.0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--splits', required=True)
    a = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(6140)
    rows = load_rows(a.data, a.splits, 'train')[:8]
    batch = Collator.from_tokenizer(AutoTokenizer.from_pretrained(BACKBONES['mbert']))(rows)
    ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
    y_sw, y_dur = batch['switch_labels'].to(device), batch['duration_labels'].to(device)
    model = DualHeadModel('mbert', dropout=0.0).to(device)
    ok = True

    print('\n[1] which parts receive gradients')
    for name, pick in (('L_duration only', 2), ('L_switch only', 1)):
        model.zero_grad(set_to_none=True)
        sw, dur = model(ids, mask)
        multitask_loss(sw, dur, y_sw, y_dur)[pick].backward()
        g = {part: grad_norm(getattr(model, part)) for part in ('backbone', 'switch_head', 'duration_head')}
        expect_head = 'duration_head' if pick == 2 else 'switch_head'
        other_head = 'switch_head' if pick == 2 else 'duration_head'
        good = g['backbone'] > 0 and g[expect_head] > 0 and g[other_head] == 0
        ok &= good
        print(f'  {name:16s} backbone {g["backbone"]:.3e} | switch_head {g["switch_head"]:.3e} | '
              f'duration_head {g["duration_head"]:.3e} -> {"PASS" if good else "FAIL"}')

    print('\n[2] batch without any switch')
    with torch.no_grad():
        sw, dur = model(ids, mask)
        _, _, l_dur = multitask_loss(sw, dur, y_sw, torch.full_like(y_dur, -1))
    good = l_dur.item() == 0.0 and not math.isnan(l_dur.item())
    ok &= good
    print(f'  L_duration = {l_dur.item()} -> {"PASS" if good else "FAIL"}')

    print('\n[3] ignore_index=-1')
    with torch.no_grad():
        _, l_sw1, l_dur1 = multitask_loss(sw, dur, y_sw, y_dur)
        sw2, dur2 = sw.clone(), dur.clone()
        sw2[y_sw == -1] += 100.0 * torch.randn_like(sw2[y_sw == -1])   # scramble ignored positions only
        dur2[y_dur == -1] += 100.0 * torch.randn_like(dur2[y_dur == -1])
        _, l_sw2, l_dur2 = multitask_loss(sw2, dur2, y_sw, y_dur)
    good = abs(l_sw1 - l_sw2) < 1e-5 and abs(l_dur1 - l_dur2) < 1e-5
    ok &= bool(good)
    print(f'  scrambling logits at -1 positions: L_sw {l_sw1:.5f} -> {l_sw2:.5f}, '
          f'L_dur {l_dur1:.5f} -> {l_dur2:.5f} -> {"PASS" if good else "FAIL"}')
    n_valid_dur = int((y_dur != -1).sum())
    print(f'  (this batch: {int((y_sw != -1).sum())} valid switch labels, {n_valid_dur} valid duration labels)')

    print('\nALL CHECKS OK' if ok else '\nSOMETHING IS WRONG')
    sys.exit(0 if ok else 1)
