# Update 1 Data Delivery: Arabic, Hindi, Chinese, Japanese, Spanish and Vietnamese × English

Author: Yu-Lun Juan (yulun). Six language pairs: `ar_en`, `hi_en`, `zh_en`, `ja_en`, `es_en`, `vi_en`.

> Status: all six pairs are complete and pass `pipeline.py validate`. Twenty records per pair were reviewed with model assistance (`reviewed`); everything else is `unreviewed`.

All results use the unchanged shared `pipeline.py` and the team settings: mBERT (tokenizer_sha256 `78b44c32…`), one sequence per item, `inherit_previous`, `observed`, `--allow-unreviewed`, 500 records per pair. Reasons and analysis are in [DECISIONS.md](DECISIONS.md).

**This folder contains no dataset text.** The SwitchLingua licence forbids redistribution of the dataset, in whole or in part, so Git keeps only statistics, manifests, IDs, `cs_type`, review records, and the Claude labels as character offsets. [`project_update_1/run_unicode_claude.py`](../../project_update_1/run_unicode_claude.py) rebuilds the full delivery, with text, in `deliverables/yulun_full/` (ignored by Git; do not commit it). See section 6.

## 1. Files for the baseline

After the rebuild (section 6), these files are in `deliverables/yulun_full/`:

| File | Content |
|---|---|
| `<pair>/tokenized/labeled_tokens_yulun.jsonl` | Labeled data (full build only) |
| `<pair>/cs_type_yulun.jsonl` | `sequence_id` → `cs_type`, only for the Macro-Average by switch type; **not a model input** |
| `<pair>/annotation/pilot_record_ids_yulun.txt` | The 20 reviewed records; they must be forced into train |

`<pair>` is `ar_en`, `hi_en`, `zh_en`, `ja_en`, `es_en` or `vi_en`.

Split command (run from the repository root). First concatenate the six `pilot_record_ids_yulun.txt` files into `all_pilot_record_ids.txt`:

```bash
python project_update_1/pipeline.py split --inputs deliverables/yulun_full/ar_en/tokenized/labeled_tokens_yulun.jsonl deliverables/yulun_full/hi_en/tokenized/labeled_tokens_yulun.jsonl deliverables/yulun_full/zh_en/tokenized/labeled_tokens_yulun.jsonl deliverables/yulun_full/ja_en/tokenized/labeled_tokens_yulun.jsonl deliverables/yulun_full/es_en/tokenized/labeled_tokens_yulun.jsonl deliverables/yulun_full/vi_en/tokenized/labeled_tokens_yulun.jsonl --seed 6140 --train-records all_pilot_record_ids.txt --out combined
```

Tested: train / validation / test = 7,918 / 912 / 900 sequences, no record crosses splits, and all 120 reviewed records are in train. ar, zh and ja also split together with Shaoyu's `ko_en` without errors (same tokenizer and label policies).

## 2. Layout

Scripts are in `project_update_1/`:

| Script | Role |
|---|---|
| `run_unicode_claude.py` | Complete processing entry point: extraction, sampling, LID, pilot review, tokenization, `cs_type`, comparison with this folder |
| `annotate_script.py` | Unicode script rule for ar, zh, ja and Devanagari (extends Shaoyu's `annotate_unicode.py`) |
| `annotate_claude.py` | Claude word-level labeling: `prepare`, `collect`, `apply`, `assemble` (`--script-native` for Hindi) |
| `claude_lid_instructions.md` | Labeling rules given to Claude (word-level version of `LID_PROMPT.md` v1.0) |
| `tests/test_annotate_script.py` | Tests for `annotate_script.py` |

```text
deliverables/yulun/              # In Git (text-free)
deliverables/yulun_full/         # Local build with text (ignored by Git)
  README.md, DECISIONS.md        # deliverables/yulun only
  <pair>/
    cs_type_yulun.jsonl
    extraction/
      texts_yulun.jsonl              [full only]
      lid.template_yulun.jsonl       [full only]
      sequence_index_yulun.jsonl     # texts without the text field
      extract_issues_yulun.jsonl
      extract_manifest_yulun.json    # includes the sampling rule
      sampled_record_ids_yulun.jsonl
    annotation/
      lid_yulun.jsonl                [full only]
      lid_offsets_yulun.jsonl        # hi, es, vi only: Claude labels as [start, end, lang]
      harmonize_changes_yulun.jsonl  # hi, es, vi only: sequence_id + offsets, before/after code
      harmonize_manifest_yulun.json  # hi, es, vi only
      lid_manifest_yulun.json
      build_manifest_yulun.json
      review_yulun.csv
      pilot_record_ids_yulun.txt
    tokenized/
      labeled_tokens_yulun.jsonl     [full only]
      token_issues_yulun.jsonl
      stats_yulun.json
      manifest_yulun.json
```

`[full only]` files contain dataset text and exist only in the local `deliverables/yulun_full/`. Every result file has the `_yulun` suffix. Scripts, shared code and the tokenizer do not. The manifests here come from the original 2026-10-18 run; a rebuild writes its own dates and checksums.

## 3. Data summary

| Pair | Method | Sequences | Valid positions | Switches | Switch rate | Small / Medium / Large | Switches without duration | UNKNOWN subwords |
|---|---|---:|---:|---:|---:|---|---:|---:|
| ar_en | Script rule | 1,594 | 40,784 | 1,936 | 4.7% | 351 / 605 / 935 | 45 | 64 |
| zh_en | Script rule | 1,747 | 54,032 | 2,402 | 4.4% | 379 / 668 / 1,283 | 72 | 163 |
| ja_en | Script rule | 1,662 | 52,918 | 2,300 | 4.3% | 374 / 567 / 1,254 | 105 | 194 |
| hi_en | Script rule + Claude | 1,571 | 42,575 | 6,198 | 14.6% | 2,661 / 2,031 / 1,325 | 181 | 676 |
| es_en | Claude | 1,561 | 37,355 | 2,319 | 6.2% | 417 / 748 / 1,016 | 138 | 642 |
| vi_en | Claude | 1,595 | 39,495 | 2,701 | 6.8% | 640 / 935 / 932 | 194 | 819 |

500 records per pair. Sequences by switch type, the sources of UNKNOWN and an analysis of how duration is measured are in DECISIONS.md sections 4 and 5.

## 4. Labeling and review

The method follows the writing system, so that every pair gets labels as close to the real language as possible:

| Pairs | Method |
|---|---|
| ar_en, zh_en, ja_en | `annotate_script.py` labels each character by Unicode script, as Shaoyu did for Cantonese, Korean and Russian; only Arabic, Devanagari and kana ranges were added |
| es_en, vi_en | `annotate_claude.py`: Claude returns one language code per word, never the text; the script rebuilds segments from the original, so they always match it exactly. One extra pass makes names and loanwords consistent |
| hi_en | Devanagari by the script rule, Latin-letter words by Claude (most Hindi in this file is written in Latin letters) |

| Character | Script-rule pairs | Claude-labeled pairs |
|---|---|---|
| Native words | Native code | Native code (by context) |
| English words | `en` | `en` (by context, including tags such as `you know?`, `right?`) |
| Proper nouns | Every Latin-letter name `en` | By written form: English form `en`, native form native, same in both or third language `UNKNOWN` |
| Digits, punctuation, symbols (including `،` `؟` `।`) | `NEUTRAL` | `NEUTRAL` |

The Latin-script pairs were first tried with Lingua (Yufan's tool): in 30 of 64 Spanish pilot items it mislabeled at least one word, almost always a short tag at a switch point, so word-level labeling with Claude was used instead (DECISIONS.md 3.3).

Review: 20 records per pair (416 items) were read by Claude; no errors were found. For es, vi and hi, Claude also produced the labels, so this review is not independent (DECISIONS.md 3.5).

## 5. Notes for users

1. **The review is model-assisted.** Reports must not call it human review; for es, vi and hi it is not independent either. **All other items are pseudo-labels** (`unreviewed`); F1 on this data is agreement with pseudo-labels, not accuracy against human ground truth.
2. **Labels for es, vi and hi are not reproducible**; the rebuild uses the delivered labels (`lid_offsets_yulun.jsonl`). Instructions, every harmonization change and file hashes are kept, so the labels are traceable.
3. **The two methods treat proper nouns differently** (table above). In hi, es and vi, proper nouns are often `UNKNOWN`, which invalidates nearby switch and duration labels.
4. **Chinese uses `zh`**, meaning SwitchLingua's Chinese file; about 1.2% of its records contain Cantonese characters, also labeled `zh`.
5. **Curly quotes and dashes (`’` `“` `—`) are `[UNK]`** in mBERT's vocabulary and are the main source of `[UNK]` in every pair.
6. **Duration depends on how bursts are measured.** Counting subwords adds 2–9 points of Large, and counting inherited punctuation moves 9–20% of bursts to another class. Differences between pairs remain after both corrections, so they come mainly from the data (DECISIONS.md 5).
7. **hi_en has the highest switch rate (14.6%)** because Latin-letter Hindi often inserts single English words.
8. See DECISIONS.md section 6 for the full list.

## 6. Reproducing the results

Run all commands from the repository root. Download `Arabic_eng.csv`, `Chinese_eng.csv`, `Japanese_eng.csv`, `Hindi_eng.csv`, `Spanish_eng.csv` and `Vietnamese_eng.csv` from the Hugging Face dataset [`Shelton1013/SwitchLingua_text`](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text) and put them in `datasets/` (create it; Git ignores it). The runner checks the SHA-256 of each file. Python 3.11 or newer is required.

```bash
# Create/reuse project_update_1/.venv, install requirements.lock.txt, and prepare the frozen tokenizer.
python project_update_1/run_unicode_claude.py --setup-only

# Rebuild the full delivery in deliverables/yulun_full/ and compare it with this folder.
python project_update_1/run_unicode_claude.py --skip-install
```

The runner prints `CHECK PASS` when every text-free file matches this folder. No API key, GPU or model weights are needed. ar, zh and ja are labeled again by the script rule; hi, es and vi are rebuilt from `lid_offsets_yulun.jsonl` and the CSV text, because Claude's output can differ between runs. A rebuild gives the same labels as the original delivery; only `annotation_source` differs for ar, zh and ja, because it contains the script name and the run date. Intermediate files go to `project_update_1/data/yulun_reproduction/` (ignored by Git).

**How the Claude labels were made** (Spanish shown; Vietnamese is the same; for Hindi add `--script-native` to `prepare` and `assemble`). `<texts.jsonl>` is `extraction/texts_yulun.jsonl` from the full build:

```bash
python project_update_1/annotate_claude.py prepare --texts <texts.jsonl> --out project_update_1/data/es_en/claude_work
# Claude writes batch_NNN.labels.jsonl for every batch_NNN.input.jsonl, following claude_lid_instructions.md
python project_update_1/annotate_claude.py collect --work project_update_1/data/es_en/claude_work --out project_update_1/data/es_en/harmonize_candidates.jsonl
# one labeler writes harmonize_decisions.jsonl ({"cid": n, "label": "n|e|u"}) for every candidate
python project_update_1/annotate_claude.py apply --work project_update_1/data/es_en/claude_work --candidates project_update_1/data/es_en/harmonize_candidates.jsonl --decisions project_update_1/data/es_en/harmonize_decisions.jsonl --out project_update_1/data/es_en/claude_work_v2
python project_update_1/annotate_claude.py assemble --texts <texts.jsonl> --work project_update_1/data/es_en/claude_work_v2 --out project_update_1/data/es_en/annotation_claude
```

The labeling steps (`#` lines) were done in Claude Code with subagents, so they cannot run unattended. A new run can give different labels; it is not needed to reproduce the delivery.

Tests:

```bash
cd project_update_1
python -m unittest discover -s tests -v
cd ..
```
