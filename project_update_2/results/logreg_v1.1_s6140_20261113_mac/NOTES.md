# LR baseline — data-v1.1 / mBERT

Validation run only; the held-out test split was not evaluated.

## Protocol

Two independent Logistic Regression classifiers, using the original position-token, normalized window token counts, and suffix bigram/trigram features (262,144 hash dimensions). No mBERT encoder is trained. Only current/past token IDs are inputs. Gold language IDs, language pair, full text, and future tokens are not features.

Data: official data-v1.1_combined.zip, mbert/combined. All 14 pairs; fr/de/it/ph use the new Claude annotation from unscreened sampling. LID groups: script_rule (6) and llm (8). Fixed split: train 18,601 / validation 2,180 / test 2,148; seed 6140. Record/group split isolation and pilot train-only assignments were checked.

Maximum sequence length is 128 including the two special-token slots reserved by the team collator. LR predicts the same first 126 original subword positions but does not add CLS/SEP to its features. This drops 14 / 480,671 valid training labels (0.0029%) and zero validation/test labels. Context window (5 or 20) is independent of max_len.

Grid: window {5,20}, C {0.1,1.0}, class_weight {None,balanced}, L2/liblinear, max_iter=500, tol=1e-4. Python 3.9.6 / scikit-learn 1.6.1 was used, as recorded in config.json; the original duration estimator uses liblinear multiclass OvR support. Do not substitute the neural-network environment's newer scikit-learn without adapting that estimator.

Switch model and threshold maximize pooled validation switch F1 on the team's 0.05–0.95 grid (step 0.01); ties keep the first candidate. Threshold selection uses six-decimal exported probabilities, matching predict.py. Duration model maximizes validation macro F1 at known gold switches. Duration probability order is S/M/L = 0/1/2. Duration labels of -1 are ignored.

## Selected settings and results

Switch: window 5, C=1.0, class_weight=None, threshold 0.27. Consult config.json for the selected duration settings.

Validation switch F1 58.05%, precision 60.60%, recall 55.71%; duration macro F1 at gold switches 55.14%, duration accuracy 55.40%; joint duration macro F1 34.96%. All 55,912 valid validation positions across 14 pairs are covered.

Scores come from the team's origin/feat/evaluate evaluation script; exact revision/hash in evaluation_provenance.json. The same validation set was used for selection, so these are development scores, not held-out test estimates. The old Update 1 test F1 of 53.23% is not directly comparable: sampling, annotations, split, and selection criterion changed.

## Reproduce from repository root

```bash
python3 deliverables/cora/logistic_regression/train_v1_1.py \
  --combined /path/to/data-v1.1/mbert/combined \
  --max-len 128 --out /path/to/new-local-run
python3 project_update_2/evaluate.py --run /path/to/new-local-run \
  --split validation --data /path/to/data-v1.1/mbert/combined/labeled_tokens.jsonl
```

Model weights remain in the local run directory, outside this results folder. This folder contains only configuration, aggregate metrics and the agreed ID/position/probability/label prediction contract; no original text or input token sequences. Nothing has been pushed to GitHub.
