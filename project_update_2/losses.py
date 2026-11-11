"""Multitask loss: L_total = lambda_sw * L_switch + lambda_dur * L_duration.

Both terms are cross-entropy with ignore_index=-1:
  - switch labels are -1 at CLS/SEP/PAD and at positions pipeline.py marked invalid;
  - duration labels are -1 at every position without a switch, so the duration head
    learns only at the moments where a switch is about to happen.
"""
import torch
import torch.nn.functional as F


def masked_ce(logits, labels, weight=None):
    """Cross-entropy over positions whose label is not -1.

    If a batch has no valid label (for example no switch at all, so every duration
    label is -1), plain cross_entropy divides by zero and returns NaN. In that case
    this returns 0 with a gradient path, so the backward pass still works.
    """
    logits = logits.reshape(-1, logits.shape[-1]).float()   # (B*T, C); float32 for a stable softmax
    labels = labels.reshape(-1)
    if not (labels != -1).any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits, labels, weight=weight, ignore_index=-1)


def multitask_loss(sw_logits, dur_logits, sw_labels, dur_labels,
                   lambda_sw=1.0, lambda_dur=1.0, sw_pos_weight=None):
    """Return (L_total, L_switch, L_duration). Only L_total is used for backward.

    sw_pos_weight: optional weight for the rare SWITCH class (class 1). None = no weighting.
    """
    weight = None
    if sw_pos_weight is not None:
        weight = torch.tensor([1.0, float(sw_pos_weight)], device=sw_logits.device)
    l_sw = masked_ce(sw_logits, sw_labels, weight)
    l_dur = masked_ce(dur_logits, dur_labels)
    return lambda_sw * l_sw + lambda_dur * l_dur, l_sw, l_dur
