"""Collator checks.

1. Alignment: after collation, subword i of a row sits at model position i+1 and
   carries its own labels; CLS, SEP and PAD positions carry -1.
2. Padding: a sequence gives the same outputs at its real positions whether it runs
   alone or inside a padded batch (causal mask + padding mask work together).
   Runs on real data with mBERT, and on random IDs with XLM-R.

Run from repo root:
  .venv-train/Scripts/python.exe project_update_2/tests/check_collator.py \
      --data <combined>/labeled_tokens.jsonl --splits <combined>/splits.jsonl
The data must be tokenized with the mBERT tokenizer.
"""
import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Collator, load_rows  # noqa: E402
from model import BACKBONES, DualHeadModel  # noqa: E402

TOL = 1e-4


def check_alignment(rows, collator):
    batch = collator(rows)
    for b, r in enumerate(rows):
        n = batch['n_subwords'][b]
        assert batch['input_ids'][b, 0] == collator.cls_id
        assert batch['input_ids'][b, n + 1] == collator.sep_id
        assert batch['input_ids'][b, 1:n + 1].tolist() == r['input_ids'][:n]
        assert batch['switch_labels'][b, 1:n + 1].tolist() == r['switch_labels'][:n]
        assert batch['duration_labels'][b, 1:n + 1].tolist() == r['duration_labels'][:n]
        for key in ('switch_labels', 'duration_labels'):
            assert batch[key][b, 0] == -1 and (batch[key][b, n + 1:] == -1).all()
        assert batch['attention_mask'][b].sum() == n + 2
    print(f'  alignment OK on {len(rows)} rows, batch shape {tuple(batch["input_ids"].shape)}')


@torch.no_grad()
def check_padding(model, collator, rows, device):
    batch = collator(rows)
    sw, dur = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
    worst = 0.0
    for b, r in enumerate(rows):
        alone = collator([r])
        sw1, dur1 = model(alone['input_ids'].to(device), alone['attention_mask'].to(device))
        L = alone['input_ids'].shape[1]
        diff = max((sw[b, :L] - sw1[0]).abs().max().item(), (dur[b, :L] - dur1[0]).abs().max().item())
        worst = max(worst, diff)
    ok = worst <= TOL
    print(f'  padding: max diff alone vs. in padded batch = {worst:.2e} -> {"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--splits', required=True)
    a = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(6140)

    rows = load_rows(a.data, a.splits, 'validation')
    rows = sorted(rows, key=lambda r: len(r['input_ids']))
    sample = [rows[0], rows[len(rows) // 2], rows[-1]]          # short, medium and long rows
    ok = True

    print('\n[mBERT, real data]')
    tok = AutoTokenizer.from_pretrained(BACKBONES['mbert'])
    col = Collator.from_tokenizer(tok, max_len=256)
    check_alignment(sample, col)
    check_alignment(sample, Collator.from_tokenizer(tok, max_len=16))    # truncation path
    model = DualHeadModel('mbert').to(device).eval()
    ok &= check_padding(model, col, sample, device)
    del model

    print('\n[XLM-R, random IDs]')
    tok = AutoTokenizer.from_pretrained(BACKBONES['xlmr'])
    col = Collator.from_tokenizer(tok, max_len=256)
    fake = [{'sequence_id': f'fake:{n}', 'language_pair': 'fake', 'input_ids': torch.randint(1000, tok.vocab_size, (n,)).tolist(),
             'switch_labels': [0] * n, 'duration_labels': [-1] * n} for n in (5, 40, 120)]
    check_alignment(fake, col)
    model = DualHeadModel('xlmr').to(device).eval()
    ok &= check_padding(model, col, fake, device)

    print('\nALL CHECKS OK' if ok else '\nSOMETHING IS WRONG')
    sys.exit(0 if ok else 1)
