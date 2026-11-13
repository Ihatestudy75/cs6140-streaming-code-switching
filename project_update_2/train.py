"""Train the dual-head causal model and keep the checkpoint with the best validation switch F1.

Outputs in --out:
  config.json      every setting of this run (reproducibility)
  train_log.jsonl  one line per training step (losses, lr) and one line per validation pass
  best.pt          model weights of the best epoch (git-ignored: *.pt)

Examples (from repo root):
  # Sanity check: overfit 32 training sequences
  .venv-train/Scripts/python.exe project_update_2/train.py --backbone mbert \
      --data D/labeled_tokens.jsonl --splits D/splits.jsonl --out project_update_2/runs/overfit_mbert \
      --data-tag data-v1.0 --overfit 32 --lr 5e-5 --max-steps 300
  # Normal training
  .venv-train/Scripts/python.exe project_update_2/train.py --backbone mbert \
      --data D/labeled_tokens.jsonl --splits D/splits.jsonl --out project_update_2/runs/mbert \
      --data-tag data-v1.0
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import transformers
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Collator, SequenceDataset, load_rows  # noqa: E402
from losses import multitask_loss  # noqa: E402
from model import BACKBONES, DualHeadModel  # noqa: E402
from provenance import check_tokenizer, file_sha256, git_commit  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(device, amp):
    """Mixed precision context. bf16 needs no loss scaling, so it is the default on this GPU."""
    if amp == 'none' or device != 'cuda':
        return torch.autocast(device_type='cpu', enabled=False)
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16 if amp == 'bf16' else torch.float16)


@torch.no_grad()
def evaluate(model, loader, args, device):
    """Validation losses, switch P/R/F1 at argmax (= threshold 0.5), duration accuracy at true switches."""
    model.eval()
    totals, n_batches = [0.0, 0.0, 0.0], 0
    sw_true, sw_pred, dur_true, dur_pred = [], [], [], []
    for batch in loader:
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        y_sw, y_dur = batch['switch_labels'].to(device), batch['duration_labels'].to(device)
        with autocast(device, args.amp):
            sw, dur = model(ids, mask)
        losses = multitask_loss(sw, dur, y_sw, y_dur, args.lambda_sw, args.lambda_dur, args.sw_pos_weight)
        totals = [t + l.item() for t, l in zip(totals, losses)]
        n_batches += 1
        valid = y_sw != -1
        sw_true.append(y_sw[valid].cpu())
        sw_pred.append(sw.argmax(-1)[valid].cpu())
        has_dur = y_dur != -1
        dur_true.append(y_dur[has_dur].cpu())
        dur_pred.append(dur.argmax(-1)[has_dur].cpu())
    model.train()
    yt, yp = torch.cat(sw_true).numpy(), torch.cat(sw_pred).numpy()
    dt, dp = torch.cat(dur_true).numpy(), torch.cat(dur_pred).numpy()
    return {'loss_total': totals[0] / n_batches, 'loss_switch': totals[1] / n_batches,
            'loss_duration': totals[2] / n_batches,
            'switch_f1': f1_score(yt, yp, zero_division=0),
            'switch_precision': precision_score(yt, yp, zero_division=0),
            'switch_recall': recall_score(yt, yp, zero_division=0),
            'duration_acc': float((dt == dp).mean()) if len(dt) else float('nan')}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backbone', choices=sorted(BACKBONES), required=True)
    p.add_argument('--data', required=True, help='combined labeled_tokens.jsonl from pipeline.py split')
    p.add_argument('--splits', required=True, help='splits.jsonl from pipeline.py split')
    p.add_argument('--out', required=True, help='run directory')
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--max-steps', type=int, default=0, help='stop after this many steps (0 = no limit)')
    p.add_argument('--warmup-ratio', type=float, default=0.1)
    p.add_argument('--max-len', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--lambda-sw', type=float, default=1.0)
    p.add_argument('--lambda-dur', type=float, default=1.0)
    p.add_argument('--sw-pos-weight', type=float, default=None, help='weight of the SWITCH class; choose on validation')
    p.add_argument('--amp', choices=['bf16', 'fp16', 'none'], default='bf16')
    p.add_argument('--seed', type=int, default=6140)
    p.add_argument('--overfit', type=int, default=0,
                   help='sanity check: train AND evaluate on the first N training sequences, dropout 0')
    p.add_argument('--log-every', type=int, default=1)
    p.add_argument('--data-tag', required=True, help='e.g. data-v1.1')
    args = p.parse_args()

    if args.amp == 'fp16':
        raise SystemExit('fp16 needs a GradScaler, which this script does not use; use --amp bf16')
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)

    # Data
    train_rows = load_rows(args.data, args.splits, 'train')
    val_rows = load_rows(args.data, args.splits, 'validation')
    if args.overfit:
        train_rows = train_rows[:args.overfit]
        val_rows = train_rows
        args.dropout = 0.0
    tokenizer = AutoTokenizer.from_pretrained(BACKBONES[args.backbone])
    tokenizer_sha256 = check_tokenizer(args.data, tokenizer, args.backbone)
    max_id = max(max(r['input_ids']) for r in train_rows + val_rows)
    if max_id >= len(tokenizer):
        raise SystemExit(f'Data has token ID {max_id} but the {args.backbone} vocabulary has {len(tokenizer)} entries: '
                         'the data was tokenized with a different tokenizer.')
    collate = Collator.from_tokenizer(tokenizer, args.max_len)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(SequenceDataset(train_rows), batch_size=args.batch_size, shuffle=True,
                              generator=gen, collate_fn=collate)
    val_loader = DataLoader(SequenceDataset(val_rows), batch_size=args.batch_size * 2, collate_fn=collate)

    # Model, optimizer, schedule
    model = DualHeadModel(args.backbone, dropout=args.dropout).to(device)
    no_decay = ('bias', 'LayerNorm.weight')
    groups = [
        {'params': [p for n, p in model.named_parameters() if not n.endswith(no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if n.endswith(no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    epochs = math.ceil(total_steps / steps_per_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    config = dict(vars(args), device=device, total_steps=total_steps, epochs_run=epochs,
                  n_train=len(train_rows), n_validation=len(val_rows),
                  model_id=BACKBONES[args.backbone], torch=torch.__version__,
                  tokenizer_sha256=tokenizer_sha256, data_sha256=file_sha256(args.data),
                  git_commit=git_commit(), transformers=transformers.__version__)
    with open(os.path.join(args.out, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f'{args.backbone}: {len(train_rows)} train / {len(val_rows)} val sequences, '
          f'{steps_per_epoch} steps/epoch, {total_steps} steps total')

    log = open(os.path.join(args.out, 'train_log.jsonl'), 'w', encoding='utf-8')
    best_f1, step, start = -1.0, 0, time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        for batch in train_loader:
            ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            y_sw, y_dur = batch['switch_labels'].to(device), batch['duration_labels'].to(device)
            with autocast(device, args.amp):
                sw, dur = model(ids, mask)
            total, l_sw, l_dur = multitask_loss(sw, dur, y_sw, y_dur,
                                                args.lambda_sw, args.lambda_dur, args.sw_pos_weight)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % args.log_every == 0:
                log.write(json.dumps({'split': 'train', 'step': step, 'epoch': epoch,
                                      'lr': scheduler.get_last_lr()[0], 'loss_total': total.item(),
                                      'loss_switch': l_sw.item(), 'loss_duration': l_dur.item()}) + '\n')
            if step >= total_steps:
                break
        metrics = evaluate(model, val_loader, args, device)
        log.write(json.dumps({'split': 'validation', 'step': step, 'epoch': epoch, **metrics}) + '\n')
        log.flush()
        improved = metrics['switch_f1'] > best_f1
        if improved:
            best_f1 = metrics['switch_f1']
        if improved and not args.overfit:          # overfit runs are sanity checks; no checkpoint
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'step': step, 'config': config},
                       os.path.join(args.out, 'best.pt'))
        if not args.overfit or epoch % 10 == 0 or step >= total_steps:
            print(f'epoch {epoch:3d} step {step:5d} | val L_sw {metrics["loss_switch"]:.4f} '
                  f'L_dur {metrics["loss_duration"]:.4f} | F1 {metrics["switch_f1"]:.3f} '
                  f'(P {metrics["switch_precision"]:.3f} R {metrics["switch_recall"]:.3f}) '
                  f'dur acc {metrics["duration_acc"]:.3f} {"*best*" if improved else ""} '
                  f'| {time.time() - start:.0f}s')
        if step >= total_steps:
            break
    log.close()
    print(f'done: best validation switch F1 = {best_f1:.4f}, {time.time() - start:.0f}s')


if __name__ == '__main__':
    main()
