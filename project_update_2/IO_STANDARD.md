# Experiment IO Standard (v0.2)

Owner: Yufan · Applies to: `project_update_2/` and the Final deliverables (Performance JSONs)
Status: **DRAFT for group review.** Items marked `[CONFIRM]` need a group decision; items marked
`[GAP]` are missing from the current code and have an owner.

v0.2 documents what PR #9 (`train.py`, `predict.py`, `data.py`) **already writes** and adds only
what is missing. Where this document and the code disagree on a name, the code wins and this
document is updated.

---

## 0. Ground rules

1. **No original text anywhere in `results/`.** Only ids, positions, probabilities and labels
   (SwitchLingua license).
2. **`position` is always in pipeline coordinates**: the 0-based index of subword x_t in the
   pipeline's `input_ids` list, without CLS/SEP. `predict.py` already converts
   (model position `i + 1` = subword `i`). Nobody else converts anything.
3. **Metrics are fractions in [0, 1]**, not percentages. Plots convert to %.
4. **Threshold and hyperparameters are chosen on `validation` only.** `predict.py` refuses to
   tune on `test`. Test predictions are produced once, by the final runs (Sat 11/14).
5. Split names are `train`, `validation`, `test` (as in `data.py`), never `val`.
6. One run = one backbone + one seed + one config.

---

## 1. Two folders: `runs/` (local) and `results/` (committed)

```
project_update_2/runs/<run_id>/          # written by train.py / predict.py — git-ignored
  config.json                            # train.py
  train_log.jsonl                        # train.py
  best.pt                                # train.py — NEVER committed (*.pt is git-ignored)
  threshold.json                         # predict.py --split validation --tune-threshold
  predictions_validation.jsonl           # predict.py
  predictions_validation_meta.json       # predict.py
  predictions_test.jsonl                 # predict.py (final runs only)
  predictions_test_meta.json             # predict.py (final runs only)

project_update_2/results/<run_id>/       # committed to GitHub — text-free copies + metrics
  config.json
  train_log.jsonl
  threshold.json
  predictions_<split>.jsonl[.gz]         # gzip if > ~20 MB
  predictions_<split>_meta.json
  metrics_<split>.json                   # written by the team evaluation code (Yulun)
  NOTES.md                               # optional free text
```

Copy rule: everything in `runs/<run_id>/` except `best.pt` goes into `results/<run_id>/`.
Checkpoints stay on Google Drive; their path goes in `NOTES.md`.

### `run_id` format

```
<backbone>_<data_tag>_s<seed>_<YYYYMMDD-HHMM>[_<tag>]
```

Examples: `xlmr_v1.1_s6140_20261113-1430`, `mbert_v1.1_s6140_20261113-1600_lam1-0.5`,
`logreg_v1.1_s6140_20261113-2000`. Pass it as `train.py --out project_update_2/runs/<run_id>`.
Allowed characters: letters, digits, `.`, `-`, `_`.

---

## 2. `config.json` (written by `train.py` before training)

`train.py` writes every command-line argument plus a few extras. Current fields:

| Field | Source | Note |
|---|---|---|
| `backbone`, `model_id` | args / `BACKBONES` | `mbert` or `xlmr` |
| `data`, `splits`, `out` | args | local paths |
| `lr`, `weight_decay`, `batch_size`, `epochs`, `max_steps`, `warmup_ratio`, `dropout`, `grad_clip`, `amp`, `seed` | args | |
| `max_len` | args | **default 256 — every Update 2 run passes `--max-len` explicitly** (see §6) |
| `lambda_sw`, `lambda_dur`, `sw_pos_weight` | args | the multitask loss settings |
| `overfit`, `log_every` | args | |
| `device`, `torch`, `total_steps`, `epochs_run`, `n_train`, `n_validation` | computed | |

`[GAP]` (owner: Yufan, one small PR to `train.py`) — add:

| Field | Why |
|---|---|
| `tokenizer_sha256` | the value from the data rows; proves which tokenizer made the labels |
| `tokenizer_model` | `model` from `tokenizer_manifest.json`; must equal `model_id` |
| `data_tag` | e.g. `data-v1.1` |
| `data_sha256` | SHA-256 of the `--data` file, so two runs can be checked for the same data |
| `git_commit` | code version |
| `transformers` | library version (torch is already recorded) |

The same PR adds a **tokenizer/backbone guard** (see §7).

---

## 3. `train_log.jsonl` (written by `train.py`)

One JSON object per line. Two kinds, told apart by `split`:

```json
{"split": "train", "step": 50, "epoch": 1, "lr": 1.8e-05, "loss_total": 0.41, "loss_switch": 0.23, "loss_duration": 0.18}
{"split": "validation", "step": 1106, "epoch": 1, "loss_total": 0.39, "loss_switch": 0.22, "loss_duration": 0.17,
 "switch_f1": 0.47, "switch_precision": 0.51, "switch_recall": 0.44, "duration_acc": 0.66}
```

- `loss_total = lambda_sw * loss_switch + lambda_dur * loss_duration`.
- Train lines are written every `log_every` steps (default every step); validation lines once per epoch.
- Validation `switch_*` uses argmax (= threshold 0.5) — monitoring only; `best.pt` is chosen by it.
- A batch with no valid duration label logs `loss_duration = 0.0` (not NaN, see `losses.masked_ce`).
  With batch size 16 this is rare, but the convergence plot should smooth over steps
  (e.g. moving average over 50 steps) rather than show raw per-step values.

---

## 4. Predictions (written by `predict.py`)

### `predictions_<split>.jsonl` — one line per position with a valid switch label

| Field | Type | Meaning |
|---|---|---|
| `sequence_id` | str | same id as the pipeline output |
| `position` | int | pipeline coordinate t (see §0.2) |
| `language_pair` | str | e.g. `fr_en` |
| `p_switch` | float | P(language of x_{t+1} ≠ language of x_t) |
| `switch_pred` | int | `p_switch >= threshold` |
| `p_duration` | [float, float, float] | P(Small 1–2), P(Medium 3–6), P(Large 7+) |
| `duration_pred` | int | argmax of `p_duration` |
| `y_switch` | int | 0 / 1 |
| `y_duration` | int | 0 / 1 / 2, or -1 when there is no duration label |

Punctuation boundary and `cs_type` are **not** stored here. The evaluation code joins them
locally by (`sequence_id`, `position`), so prediction files stay small and text-free.

### `predictions_<split>_meta.json`

Current fields: `run`, `backbone`, `model_id`, `split`, `threshold`, `n_sequences`,
`n_positions`, `n_true_switches`, `checkpoint_epoch`.

`[GAP]` (owner: Shaoyu or Yufan) — add `n_valid_labels_total` (valid switch labels before
truncation) and `n_dropped_by_truncation`. `predict.py` skips positions cut by `max_len`
without counting them; without these two numbers we cannot say how much of the test set was scored.

### `threshold.json`

`{"threshold": 0.31, "validation_switch_f1": ..., "validation_switch_f1_at_0.5": ..., "grid": [0.05, 0.95, 0.01]}`
Criterion: maximize **pooled** switch F1 on validation (see §6).

---

## 5. `metrics_<split>.json` (written by the team evaluation code — Yulun)

Computed only from `predictions_<split>.jsonl` + `threshold.json` + local join data.
The logistic regression baseline writes the same file (`backbone = "logreg"`).

```json
{
  "io_version": "0.2",
  "run_id": "xlmr_v1.1_s6140_20261113-1430",
  "split": "validation",
  "threshold": 0.31,
  "coverage": {"n_valid_labels_total": 0, "n_positions": 0, "n_dropped_by_truncation": 0},

  "pooled": {
    "switch":   {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pos": 0, "n_positions": 0},
    "duration": {"accuracy": 0.0, "macro_f1": 0.0, "n": 0,
                 "per_class": {"0": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
                               "1": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
                               "2": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}},
                 "confusion_matrix": [[0,0,0],[0,0,0],[0,0,0]]},
    "joint_macro_f1": 0.0
  },
  "by_boundary": {
    "after_punct": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pos": 0},
    "non_punct":   {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pos": 0}
  },
  "per_pair": {
    "fr_en": {"lid_method": "lingua",
              "switch": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pos": 0},
              "duration": {"accuracy": 0.0, "macro_f1": 0.0, "n": 0}}
  },
  "by_cs_type": {
    "intrasentential": {"switch": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pos": 0}}
  },
  "universality": {
    "metric": "per_pair.switch.f1",
    "ddof": 0,
    "sigma_all": {"value": 0.0, "n_pairs": 14, "mean": 0.0},
    "sigma_by_lid_method": {
      "script_rule": {"value": 0.0, "n_pairs": 6, "mean": 0.0},
      "llm":         {"value": 0.0, "n_pairs": 4, "mean": 0.0},
      "lingua":      {"value": 0.0, "n_pairs": 4, "mean": 0.0}
    }
  }
}
```

### Metric definitions

- **Anticipatory (switch) F1** — positive class = switch at t+1; over rows of the prediction
  file (all have `y_switch != -1`); predicted positive when `p_switch >= threshold`.
- **Duration accuracy / macro F1** — over rows with `y_duration != -1` (gold switches), using
  `duration_pred`. This measures the duration head alone. Both are required: ~65% of bursts are
  Large, so accuracy alone flatters a model that always says "long".
- **joint_macro_f1** — same definition as the Update 1 baseline. `[CONFIRM]` Yulun / Qianfu paste it here.
- **by_boundary** — `after_punct`: subword x_t is punctuation; `non_punct`: all other positions.
  Same rule as the Update 1 baseline evaluation.
- **σ (universality)** — population std (`ddof = 0`) of per-pair switch F1 at the run's
  threshold; over all 14 pairs and within each LID group.
  `[CONFIRM]` that "three versions of σ" means the three LID groups.
- **by_cs_type** — optional for Update 2; needed for the Final (intra- vs inter-sentential).

### LID method groups

| Group | Pairs |
|---|---|
| `script_rule` | ar_en, zh_en, ja_en, yue_en, ko_en, ru_en |
| `llm` | es_en, vi_en, ms_en, hi_en* |
| `lingua` | fr_en, de_en, it_en, philippines_en |

\* hi_en is Rule + Claude; grouped as `llm` because most of its text is romanized.
`[CONFIRM]` pair names match `language_pair` values in the data.

---

## 6. Decisions for Update 2

| Topic | Decision | Reason |
|---|---|---|
| Collator | Shaoyu's `data.Collator` is the team collator | Already meets the alignment checklist; a second copy would drift |
| Context mode | whole sequence + causal mask (`is_decoder=True`) | Causality test passes for both backbones; prefix truncation dropped |
| `max_len` | 128, passed explicitly; raise only if `data.py --max-len 128 192 256` shows > 1% of valid labels lost on data-v1.1 | Colab memory; the code default (256) is not the team setting |
| Threshold criterion | maximize pooled switch F1 on validation (as `predict.py` does); the baseline rerun uses the same criterion | Fair comparison; differs from Update 1 (average over 42 groups) — state this in the report |
| Duration metric | report accuracy **and** macro F1 for every model, baseline included | Class imbalance |
| Labels per backbone | mBERT runs use mBERT-tokenized data, XLM-R runs use XLM-R-tokenized data | Duration counts subwords, so the two label sets differ; say so in the report |

---

## 7. Tokenizer/backbone guard `[GAP]` (owner: Yufan)

`train.py` stops when a token id is ≥ the backbone's vocabulary size. That catches XLM-R data
fed to mBERT (XLM-R ids reach ~250k, mBERT has ~120k), but **not** mBERT data fed to XLM-R:
every mBERT id is a valid XLM-R id, so training would run on meaningless inputs without any error.

Fix: `train.py` and `predict.py` take `--tokenizer-dir` (the frozen tokenizer folder from
`pipeline.py prepare-tokenizer`), read its `tokenizer_manifest.json`, and stop unless
(a) `manifest["model"]` equals `BACKBONES[backbone]` and (b) every data row's
`tokenizer_sha256` equals `manifest["tokenizer_sha256"]`. Both values go into `config.json`.

---

## 8. Checklist before pushing a run to `results/`

- [ ] `config.json`, `train_log.jsonl`, `threshold.json`, `predictions_validation.jsonl`, its meta, `metrics_validation.json`
- [ ] `max_len` in `config.json` is the team value, not the default
- [ ] no `best.pt`, no file with original text
- [ ] no `NaN` in any JSON
- [ ] test files only for the final runs

## Changelog

- v0.1 (2026-11-09) — first draft (Yufan)
- v0.1.1 (2026-11-10) — context mode fixed to whole sequence + causal mask
- v0.2 (2026-11-12) — rewritten to match PR #9 outputs: `runs/` vs `results/`, real field names,
  split name `validation`, threshold criterion, gaps with owners, tokenizer guard
