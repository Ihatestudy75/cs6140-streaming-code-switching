# Three-model validation comparison

Data: data-v1.1; seed: 6140; max_len: 128; 2,180 identical validation sequences; 14 language pairs. The baseline is the enhanced logistic regression run from PR #16.

All pooled and per-pair metrics below were recomputed from predictions with the shared `deliverables/cora/logistic_regression/metrics.py` scorer and checked against the saved metrics. Baseline source IO version is 0.5; transformer source IO version is 0.6.

## Overall results

Scores are percentages. Each model uses its own validation-tuned threshold.

| Metric | LR baseline (enhanced) | mBERT | XLM-R |
| --- | --- | --- | --- |
| Switch precision | 59.59 | 64.23 | 62.91 |
| Switch recall | 59.51 | 64.76 | 65.01 |
| Switch F1 | 59.55 | 64.49 | 63.95 |
| Duration accuracy (oracle switch) | 57.60 | 58.83 | 58.30 |
| Duration macro F1 (oracle switch) | 57.16 | 57.59 | 57.43 |
| Joint duration macro F1 | 34.90 | 36.86 | 36.40 |
| Switch threshold | 0.25 | 0.34 | 0.35 |
| Switch F1 at fixed threshold 0.5 | 52.61 | 61.81 | 62.37 |
| Valid positions | 55912 | 55912 | 49590 |
| Gold switches | 3865 | 3865 | 3893 |
| Duration support | 3677 | 3677 | 3813 |

## Per-pair switch F1

| Language pair | LR baseline (enhanced) | mBERT | XLM-R |
| --- | --- | --- | --- |
| ar_en | 61.08 | 66.12 | 65.76 |
| de_en | 63.00 | 70.50 | 70.71 |
| es_en | 64.61 | 72.07 | 69.83 |
| fr_en | 73.68 | 75.96 | 74.18 |
| hi_en | 63.20 | 66.72 | 65.87 |
| it_en | 59.81 | 64.39 | 66.02 |
| ja_en | 71.43 | 76.30 | 76.89 |
| ko_en | 74.23 | 79.72 | 80.29 |
| ms_en | 53.71 | 58.14 | 58.24 |
| philippines_en | 44.95 | 51.02 | 53.23 |
| ru_en | 59.46 | 64.36 | 61.67 |
| vi_en | 60.14 | 69.41 | 66.06 |
| yue_en | 47.46 | 51.01 | 49.72 |
| zh_en | 60.95 | 64.60 | 62.04 |
| Unweighted pair mean | 61.26 | 66.45 | 65.75 |
| Pair standard deviation (pp, ddof=0) | 8.30 | 8.36 | 8.23 |

## Interpretation and scope

- mBERT and XLM-R improve pooled switch F1 over the enhanced LR baseline by 4.95 and 4.40 percentage points, respectively.
- LR and mBERT use mBERT tokenization. XLM-R uses its own tokenization: valid positions, switch support, and duration labels differ. Duration bins are S=1–2, M=3–6, L=7+ subwords of each tokenizer.
- Oracle-switch duration scores use gold switches with known duration. Joint duration F1 also requires correct switch detection. Pooled switch F1 and the unweighted mean of pair F1 are different aggregates.
- These are validation results used for model/threshold selection, not held-out test performance. The runs do not have matched hyperparameter search budgets; small differences are descriptive, not evidence of statistical significance.
