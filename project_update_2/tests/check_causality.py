"""Causality test: the output at position t must not depend on any token after t.

Method: run a batch once, replace every token after position t with random tokens,
run again, and compare the outputs at positions 0..t. They must be equal.
Negative control: the same test with causal=False must FAIL, which shows the test
can detect look-ahead.

Run from repo root:
  .venv-train/Scripts/python.exe project_update_2/tests/check_causality.py
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DualHeadModel  # noqa: E402

TOL = 1e-4   # fp32 GPU kernels can differ by ~1e-6; real look-ahead gives differences near 1e-1


@torch.no_grad()
def outputs(model, ids, mask):
    sw, dur = model(ids, mask)
    return torch.cat([sw, dur], dim=-1)    # (B, T, 5)


def check(backbone, causal, device, batch=4, length=64, cut_points=(0, 10, 31, 62)):
    torch.manual_seed(6140)
    # eval(): turns dropout off, so two runs on the same input give the same output.
    model = DualHeadModel(backbone, causal=causal).to(device).eval()
    vocab = model.backbone.config.vocab_size
    # Token IDs >= 1000 avoid special tokens ([PAD], [CLS], <s>, ...) in both vocabularies.
    ids = torch.randint(1000, vocab, (batch, length), device=device)
    mask = torch.ones_like(ids)
    base = outputs(model, ids, mask)
    worst_prefix, results = 0.0, []
    for t in cut_points:
        changed = ids.clone()
        changed[:, t + 1:] = torch.randint(1000, vocab, (batch, length - t - 1), device=device)
        new = outputs(model, changed, mask)
        prefix_diff = (new[:, :t + 1] - base[:, :t + 1]).abs().max().item()   # must be ~0
        suffix_diff = (new[:, t + 1:] - base[:, t + 1:]).abs().max().item()   # must be > 0
        worst_prefix = max(worst_prefix, prefix_diff)
        results.append((t, prefix_diff, suffix_diff))
    passed = worst_prefix <= TOL
    attn = getattr(model.backbone.config, '_attn_implementation', '?')
    print(f'\n[{backbone} | causal={causal} | attention={attn}]')
    for t, p, s in results:
        print(f'  t={t:2d}  max diff at 0..t = {p:.2e}   max diff after t = {s:.2e}')
    print(f'  -> {"PASS" if passed else "FAIL"} (tolerance {TOL:.0e})')
    del model
    torch.cuda.empty_cache()
    return passed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbones', nargs='+', default=['mbert', 'xlmr'])
    a = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ok = True
    for b in a.backbones:
        ok &= check(b, causal=True, device=device)          # must PASS
        ok &= not check(b, causal=False, device=device)     # negative control: must FAIL
    print('\nALL CHECKS OK' if ok else '\nSOMETHING IS WRONG')
    sys.exit(0 if ok else 1)
