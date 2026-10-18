# Baseline and Next-Step Results

We used 21,859 sequences from 14 language pairs. Read [REPORT.md](REPORT.md) for methods and limits.

| Test switch Macro F1 | Baseline | Prefix |
|---|---:|---:|
| Full-record samples: 10 pairs | 56.01% | 58.84% |
| Screened samples: Yufan's 4 pairs | 36.11% | 37.96% |
| All 14 pairs: mixed samples | 50.32% | 52.87% |

The baseline uses the current token's switch frequency. The prefix model uses up to five tokens. Settings were chosen on validation data only.

These scores measure agreement with automatic labels. Sampling and label methods differ across data owners, so lower scores do not prove lower data quality. All 29 tests passed. Source labels and shared code are unchanged.

Detailed scores are in metrics.csv and results.json. Source commits and hashes are in run_manifest.json.

The complete experiment is in `cs6140-all-deliveries-20261018.zip`, including input snapshots, fixed splits, model counts, and predictions.
