# Expanded LR baseline — validation development experiment

Same official mbert/combined data-v1.1 split, seed 6140, max_len=128 and 55,912 validation positions as the original LR run. Test was not evaluated. Original results are preserved separately. Predictions follow the team IO contract and were evaluated by the same team script.

## Features and search

Original hashed token features are kept unchanged in a separate 262,144-dimensional block. Extra features use a 16,384-dimensional block and the frozen mBERT vocabulary (verified against every data row). No encoder is trained. Only current/past token IDs and their vocabulary strings enter features; no gold language labels, full text, language pair, future tokens or cs_type.

Morphology features: continuation (##), punctuation, digit, unknown token, capitalization, capped subword length, recent positions, window rates, and distance to the last punctuation within the context window (capped at 8, with a distinct no-punctuation category).

Script features: Unicode script categories at recent positions, script bigram/trigram suffixes, window proportions, distinct-script count and script-transition rate. These are character-script categories, not predicted or gold language IDs. Scripts shared by multiple languages cannot distinguish those languages on their own.

88 configurations, each fitting independent switch and duration classifiers:
- Original features and all features: identical 40-setting grid each; windows {3,5,10,20}, C {0.03,0.1,0.3,1,3}, class weights {None,balanced}.
- Morphology-only addition and script-only addition: four settings each, window=5, C={0.1,1}, weights={None,balanced}.

`ablation.csv` first compares all four feature families under the same four-setting window-5 grid. It then compares original versus all features under the same expanded 40-setting grid. Best switch and duration scores are selected independently and may come from different configurations; these columns do not describe one shared fitted model. Individual feature contributions cannot be inferred from unequal search budgets.

Switch selection: pooled validation F1, same 0.05..0.95 threshold grid and six-decimal probabilities as the original run. Duration selection: macro F1 on gold switches with known duration. No selection by joint F1. Ties keep the first candidate. Environment and exact source hashes are in config.json.

## Selected classifiers

Switch: `{"mode": "all", "window": 5, "C": 0.3, "class_weight": null, "threshold": 0.25}`

Duration: `{"mode": "all", "window": 10, "C": 0.03, "class_weight": null}`

## Validation results

| Metric | Original | Selected | Change (pp) |
|---|---:|---:|---:|
| switch_f1 | 58.05% | 59.55% | +1.50 |
| precision | 60.60% | 59.59% | -1.01 |
| recall | 55.71% | 59.51% | +3.80 |
| duration_accuracy_oracle_switch | 55.40% | 57.60% | +2.20 |
| duration_macro_f1_oracle_switch | 55.14% | 57.16% | +2.01 |
| joint_duration_macro_f1 | 34.96% | 34.90% | -0.06 |

These are validation-selected development results. Repeated searches increase selection optimism; no improvement is claimed to generalize to test, and no statistical significance claim is made. Freeze the selected configuration before any final test evaluation. Per-pair regressions are included in per_pair_comparison.csv.

Checks: 13 feature/causality/serialization/metric unit tests; complete validation schema, labels and position coverage; threshold-grid optimality; loaded-model prediction reproduction; source-hash verification. Predictions contain no original sentences or input token sequences. Model and vocabulary stay in the local run folder. Nothing has been pushed to GitHub.

## Reproduce from repository root

```bash
python3 deliverables/cora/logistic_regression/train_enhanced.py \
  --combined /path/to/data-v1.1/mbert/combined --max-len 128 \
  --vocab /path/to/frozen-mbert/vocab.txt --out /path/to/new-local-run
python3 project_update_2/evaluate.py --run /path/to/new-local-run \
  --split validation --data /path/to/data-v1.1/mbert/combined/labeled_tokens.jsonl
```

Use the recorded Python/scikit-learn environment: the baseline uses liblinear multiclass OvR support available in scikit-learn 1.6.1.
