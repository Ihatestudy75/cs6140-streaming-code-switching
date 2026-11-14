# mbert_v1.1_s6140_20261113-1805

- Backbone `mbert`, data-v1.1 (14 pairs), `--max-len 128`, all other settings default (see `config.json`).
- Training code: commit `a3efb49` (`config.json` → `git_commit`).
- Predictions: `predict.py --split validation --tune-threshold`, threshold 0.34. The coverage fields in the meta file come from PR #17 (`feat/predict-coverage`).
- Checkpoint: `mbert_v1.1_s6140_20261113-1805/best.pt` inside `update2_checkpoints_shaoyu.zip` on the team's private Google Drive (not in this repo).
- `metrics_validation.json` will be added after `evaluate.py` (PR #13) is merged.
- Test predictions: not run yet (run once for the final numbers).
