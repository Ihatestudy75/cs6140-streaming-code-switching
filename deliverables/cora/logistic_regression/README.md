# Logistic Regression streaming baseline

Qianfu Peng — CS6140, Update 1

This is the primary machine learning baseline for next-token language-switch prediction. A second classifier predicts the following language span length: 1–2, 3–6, or 7+ subwords. One shared Logistic Regression classifier per task is trained on all 14 language pairs.

## Source files

- `train.py`: fixed-split training, validation model/threshold selection, and evaluation.
- `model.py`: causal sparse token features and inference.
- `metrics.py`: original baseline scoring functions.
- `predict.py`: prediction from observed token IDs.
- `analyze_boundaries.py` and `boundary_rules.py`: validation threshold verification, per-pair precision/recall/F1, and punctuation analysis.
- `test_model.py` and `test_boundary_rules.py`: nine checks for causality, metrics, serialization, and boundary rules.

## Install and test

Use Python 3.9 or later in an isolated environment. The recorded experiment used Python 3.9.6. Run these commands from this directory:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest test_model test_boundary_rules -v
```

## Local data contract

SwitchLingua prohibits redistribution. Obtain authorized reference assets locally through the project's approved process. This directory contains code and documentation only. Do not commit dataset files, sample text, tokenized records, per-record predictions, trained models, or data archives.

The local reference folder, `cs6140-all-deliveries-20261018`, must contain:

- `runs/all-v3/combined/labeled_tokens.jsonl` and `splits.jsonl`;
- `runs/all-v3/run_manifest.json`, `all_pilot_record_ids.txt`, and `results.json`;
- earlier prediction files under `runs/all-v3/` for boundary comparisons;
- `source/deliverables/` with original sequence-level cs_type metadata;
- `run.py` from the original counting baseline.

The runner verifies data and split hashes against the reference manifest. Data acquisition and preprocessing are separate from this runner.

## Train and analyze

```bash
python3 train.py --reference-root /path/to/cs6140-all-deliveries-20261018
python3 analyze_boundaries.py --reference-root /path/to/cs6140-all-deliveries-20261018
python3 predict.py --prefix-ids '[101, 1234, 5678]'
```

The IDs illustrate syntax and are not dataset samples. Use IDs from the same frozen mBERT tokenizer, including only the current and past tokens. Using its tokenizer does not mean that mBERT was trained.

Alternatively, set `CS6140_REFERENCE_ROOT`. The fallback is `reference` relative to the working directory. No machine-specific absolute reference path is included.

Training defaults to `runs/logistic-v1/`, which must not already exist. `--out` accepts another new directory. Keep generated files local. The boundary analyzer expects the default run directory; prediction supports `--model` for other model locations.

The published `train.py` changes only the original machine-specific default reference-path lookup. The analyzer checks source hashes, so train a fresh local run with this version rather than mixing it with an older source manifest. No training was rerun for this publication. The original local verification/report-generation helper is omitted because it is tied to that saved experiment snapshot.

## Recorded experiment

The grid tests windows of 5 and 20 subwords, C values of 0.1 and 1.0, and balanced versus unweighted classes. Features include relative-position token identity, normalized window counts, and suffix bigrams/trigrams, hashed into 262,144 dimensions. Models use L2 regularization, liblinear, and seed 6140.

The selected switch classifier uses window 5, C=1.0, no class weighting, and validation threshold 0.22744425. The duration classifier uses window 5, C=0.1, and balanced weights. Switch selection averages F1 equally over 42 language-pair × cs_type groups; duration selection uses validation macro F1 at known gold switches.

| Method | Pooled test switch F1 | Joint switch-and-duration macro F1 |
|---|---:|---:|
| Token counts | 49.33% | 24.23% |
| Prefix counts | 52.14% | 27.68% |
| Logistic Regression | 53.23% | 30.42% |

Test precision is 51.77% and recall is 54.77%. These are not accuracy scores. The gain over prefix counts is 1.09 percentage points; statistical significance has not been established.

Recall is 90.49% after punctuation and 18.51% at non-punctuation boundaries. Non-punctuation is a within-clause proxy, not a verified syntactic label. The lowest-scoring dataset group is `philippines_en`, with F1 30.80%; this does not establish poor data quality or inherent language difficulty.

## Limits

Only current and past token IDs enter the predictors. Gold labels, pair IDs, cs_type, full text, and future tokens are excluded from features. Boundary groups are used for analysis only. Existing subword segmentation is reused; this is not a complete raw-character streaming system.

Labels are automatic. Prior test scores had already been viewed, so this is an exploratory comparison. The five-token window won only the limited validation search and is not an assignment requirement or a general optimum.

## Data-v1.1 validation runs

Use the official `data-v1.1_combined.zip` and its `mbert/combined` directory.
The runners preserve the supplied split and use `--max-len 128` (126 subwords,
reserving the same two special-token slots as the sequence collator).
Run from the repository root with the dependencies listed above:

```bash
python3 deliverables/cora/logistic_regression/train_v1_1.py \
  --combined /path/to/mbert/combined --max-len 128 --out /path/to/new-original-run
python3 deliverables/cora/logistic_regression/train_enhanced.py \
  --combined /path/to/mbert/combined --max-len 128 \
  --vocab /path/to/frozen-mbert/vocab.txt --out /path/to/new-enhanced-run
```

The enhanced runner adds causal subword, punctuation, and Unicode-script features
and compares 88 configurations. Both runners select switch parameters by pooled
validation F1 and duration parameters by validation macro F1 at known gold
switches. They export the standard validation predictions and retain models in
the local output directory. Neither runner evaluates test.

| Validation metric | Original LR | Enhanced LR |
|---|---:|---:|
| Switch F1 | 58.05% | 59.55% |
| Switch precision | 60.60% | 59.59% |
| Switch recall | 55.71% | 59.51% |
| Duration macro F1 | 55.14% | 57.16% |
| Joint F1 | 34.96% | 34.90% |

These are validation-selected development scores, not held-out test estimates.
The Update 1 test scores above are not directly comparable to this data version.

Results and ablations:
- [Original LR](../../../project_update_2/results/logreg_v1.1_s6140_20261113_mac/NOTES.md)
- [Enhanced LR](../../../project_update_2/results/logreg_enhanced_v1.1_s6140_20261113/NOTES.md)

Run the feature and baseline checks from this directory:

```bash
python3 -m unittest test_model test_boundary_rules test_enhanced -v
```
