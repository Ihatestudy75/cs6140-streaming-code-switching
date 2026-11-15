# xlmr_v1.1_s6140_20261113-1823

- Backbone `xlmr`, data-v1.1 (14 pairs), `--max-len 128`, all other settings default (see `config.json`).
- Training code: commit `a3efb49` (`config.json` → `git_commit`).
- Predictions: `predict.py --split validation --tune-threshold`, threshold 0.35. The coverage fields in the meta file come from PR #17 (`feat/predict-coverage`).
- Checkpoint: `xlmr_v1.1_s6140_20261113-1823/best.pt` inside `update2_checkpoints_shaoyu.zip` on the team's private Google Drive (not in this repo).
- `metrics_validation.json` and `metrics_test.json`: `evaluate.py --split <split> --data <local labeled_tokens.jsonl>`.
- Test predictions: run once on 2026-11-14 with `predict.py --split test` (code at main `9467cdc`). The threshold 0.35 is read from `threshold.json` (chosen on validation); nothing was tuned on test.
