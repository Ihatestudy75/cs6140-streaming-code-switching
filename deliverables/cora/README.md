# Cora baseline + causal-prefix experiment

Data: PR #1 (`data/shaoyu`), commit `440cbb032b57b010f909ac30d6deb223f7e642dc`.

Read [REPORT.md](REPORT.md) for methods, data audit, limitations, metrics, and next steps. Scores measure agreement with pseudo-labels, not independent human ground truth.

| Test metric | Current-token baseline | Causal prefix |
|---|---:|---:|
| Switch F1 averaged over 12 pair/type cells | 52.81% | 54.61% |
| Pooled switch F1 | 51.84% | 53.72% |
| Duration macro F1 at gold switch positions | 48.49% | 44.11% |

`run.py` and `test_baseline.py` are exposed for code review. The complete offline experiment, including source snapshot, fixed splits, model counts and predictions, is in `cs6140-local-baseline.zip`. Extract it before running:

```bash
unzip cs6140-local-baseline.zip
cd cs6140-local
python3 -m unittest test_baseline -v
python3 run.py --out runs/v2
```

Only Python standard library is required (tested with Python 3.9.6). Keep `runs/v1` unchanged; outputs must be empty. Original 8 pipeline tests and 5 baseline tests passed.

The report describes the local run before this publication. Publishing this branch does not merge it or the data PR. No original data files or shared pipeline files are modified.
