# Data v1.1 Standard

Owner: Yulun · Applies to: all 14 language pairs for Update 2 and the Final

data-v1.1 is the Update 1 data, re-tokenized with two frozen tokenizers (mBERT and XLM-R)
and split once for all 14 pairs together. The LID labels are **not** redone for 10 pairs:
they keep their Update 1 method. The exception is the four Update 1 Lingua pairs (`fr_en`,
`de_en`, `it_en`, `philippines_en`), which were re-sampled and relabeled with Claude for v1.1
because Lingua missed most short switches; see [`RELABEL_V1_1.md`](RELABEL_V1_1.md).
The Update 1 processing standard
([`project_update_1/README.md`](../project_update_1/README.md#shared-standard)) still applies;
this file lists only what is new or fixed for v1.1.

---

## 1. Frozen tokenizers

| Backbone | Model | Revision | `tokenizer_sha256` |
|---|---|---|---|
| `mbert` | `google-bert/bert-base-multilingual-cased` | `3f076fdb1ab68d5b2880cb87a0886f315b8146f8` | `78b44c32238e3b5e83d1deaa63c3fa1466ddf73547ce641a36f75180079ab45f` |
| `xlmr` | `FacebookAI/xlm-roberta-base` | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` | `088aef1538504d68661da8cf24bc0f98581c0e356cc7060ad724232d5f7545f2` |

The mBERT tokenizer is the Update 1 tokenizer, unchanged. Every `manifest.json` you deliver
must show one of these two hashes.

### Prepare them

Use the Update 1 environment (`project_update_1/requirements.lock.txt`, transformers 4.57.6).
From the repository root:

```bash
# Windows (Git Bash); on macOS/Linux use project_update_1/.venv/bin/python
PY=project_update_1/.venv/Scripts/python.exe
$PY project_update_2/prepare_tokenizer_v1_1.py --backbone mbert
$PY project_update_2/prepare_tokenizer_v1_1.py --backbone xlmr
```

Each command must end with `PASS`. The folders are `project_update_1/assets/mbert/` and
`project_update_1/assets/xlmr/` (ignored by Git). If a folder already exists, the script
only checks it.

**Do not** run `pipeline.py prepare-tokenizer` yourself, and do not apply the Update 1
runners' CRLF fix to the XLM-R folder. That fix rewrites every file, and XLM-R's binary
`sentencepiece.bpe.model` contains CRLF byte pairs that are data, so the file would be
corrupted. The script only converts JSON/TXT files.

## 2. Re-tokenize your pairs

Run both backbones for each of your pairs, from the **same** `texts` and `lid` files that
produced your Update 1 `labeled_tokens`. Example (zh_en, Yulun's file names):

```bash
for BB in mbert xlmr; do
  $PY project_update_1/pipeline.py tokenize \
    --texts deliverables/yulun_full/zh_en/extraction/texts_yulun.jsonl \
    --lid   deliverables/yulun_full/zh_en/annotation/lid_yulun.jsonl \
    --tokenizer-dir project_update_1/assets/$BB \
    --neutral-policy inherit_previous --end-policy observed --allow-unreviewed \
    --out project_update_1/data/v1.1/$BB/zh_en
  $PY project_update_1/pipeline.py validate --input project_update_1/data/v1.1/$BB/zh_en/labeled_tokens.jsonl
done
```

The policy flags are fixed for v1.1. Use exactly these values. `tokenize` refuses a
non-empty `--out` folder, so delete the pair's folder before a rerun.

### Checks before you deliver

1. `validate` prints `PASS` for both backbones.
2. `manifest.json` → `tokenizer.tokenizer_sha256` equals the table in §1.
3. **Same input as Update 1:** the new mBERT `labeled_tokens.jsonl` has the same
   `sequence_id`, `input_ids`, `language_ids`, `switch_labels` and `duration_labels` as your
   Update 1 file. The only expected exception is a pair whose LID was changed after Update 1
   (for example, newly reviewed pilot records). Say so when you deliver it.
4. Look at a few XLM-R rows: tokens and `language_ids` side by side. XLM-R often produces a
   lone `▁` subword before CJK characters and punctuation; it should carry the language of
   the next subword.

## 3. Deliver

| What | Where |
|---|---|
| `project_update_1/data/v1.1/{mbert,xlmr}/<pair>/` (all 4 files, zipped) | the team's **private** Google Drive folder (link in the group chat, never in this repo) |
| your `pilot_record_ids_<name>.txt` for each pair | the same Drive folder |

`labeled_tokens.jsonl` contains the original sentences. **Never commit it, and never put
it anywhere public** (SwitchLingua licence). Yulun commits the text-free files
(`stats.json`, `manifest.json`, `token_issues.jsonl`, `splits.jsonl`, `split_manifest.json`)
together with the split, in [`data_v1.1/`](data_v1.1/) (§4).

## 4. Pairs, LID groups and where to find them

`language_pair` values in the data:

| LID group | Pairs | Update 1 owner | v1.1 LID |
|---|---|---|---|
| `script_rule` | `ar_en`, `zh_en`, `ja_en` · `yue_en`, `ko_en`, `ru_en` | Yulun · Shaoyu | Update 1 labels |
| `llm` | `es_en`, `vi_en`, `hi_en` (script rule + Claude) · `ms_en` (DeepSeek) | Yulun · Shaoyu | Update 1 labels |
| `llm` | `fr_en`, `de_en`, `it_en`, `philippines_en` (Claude) | Yufan (Lingua) | relabeled by Yulun ([`RELABEL_V1_1.md`](RELABEL_V1_1.md)) |

The `lingua` group of Update 1 no longer exists in v1.1.

Text-free files (in this repository):

| What | Where |
|---|---|
| Per pair and backbone: `stats.json`, `manifest.json`, `token_issues.jsonl` (all 14 pairs) | [`data_v1.1/{mbert,xlmr}/<pair>/`](data_v1.1/) |
| The split: `splits.jsonl`, `split_manifest.json` | [`data_v1.1/{mbert,xlmr}/combined/`](data_v1.1/) |
| Statistics table (14 pairs × 2 backbones) | [`data_v1.1_stats.md`](data_v1.1_stats.md) · [`.csv`](data_v1.1_stats.csv) |
| fr/de/it/philippines: labels as character offsets, sample and pilot ids, `cs_type`, review and harmonization records | [`data_v1.1/relabel/<pair>/`](data_v1.1/relabel/) |
| ar/zh/ja/es/vi/hi: Update 1 records (labels, ids, `cs_type`, review) | [`deliverables/yulun/<pair>/`](../deliverables/yulun/) |
| yue/ko/ru/ms: Update 1 records | [`deliverables/shaoyu/<pair>/`](../deliverables/shaoyu/) |
| fr/de/it/philippines, Update 1 Lingua version (superseded, not used in v1.1) | branch `data/yufan`, `deliverables/yufan/` |

Files with text (private Drive only): `data-v1.1_combined.zip` holds
`{mbert,xlmr}/combined/{labeled_tokens,splits,split_manifest}`, the inputs of `train.py`.

## 5. Split (Yulun)

One split run per backbone over all 14 pairs, with the pilot records forced into train.
They were inspected during LID development and must not reach validation or test:

```bash
# <pilot-dir> holds every pair's pilot_record_ids .txt file
$PY project_update_2/data_v1_1.py split --pilot-dir <pilot-dir>
$PY project_update_2/data_v1_1.py stats-table --out project_update_2/data_v1.1_stats
$PY project_update_2/data_v1_1.py export --out project_update_2/data_v1.1
```

`split` refuses to run unless all 14 pairs exist for both backbones and every pair has pilot
ids. It writes `project_update_1/data/v1.1/{mbert,xlmr}/combined/` (`labeled_tokens.jsonl`,
`splits.jsonl`, `split_manifest.json`): the inputs of `train.py --data` / `--splits`.
The assignment depends only on `record_id`, the text and `language_pair`, not on the
tokenizer, so the mBERT and XLM-R `splits.jsonl` must give every `sequence_id` the same split;
`split` checks this at the end (`compare-splits`) and fails otherwise. `export` copies the
text-free files into the repository; it refuses any row with a text field and rewrites local
absolute paths as repository paths.

Result (seed 6140): 22,929 sequences, train 18,601 / validation 2,180 / test 2,148, identical
for both backbones; 280 pilot records (20 per pair) in train.

## 6. Fixed rules (v1.1)

| Rule | Value |
|---|---|
| Duration unit | subwords of the backbone's own tokenizer, so mBERT and XLM-R labels differ |
| Neutral tokens (punctuation, digits) | `inherit_previous` |
| End of sequence | `observed` |
| LID | Update 1 labels, not redone, except fr/de/it/philippines (relabeled with Claude); unreviewed except the pilot records, so `--allow-unreviewed` |
| Split seed | 6140; pilot records forced into train |
| Deferred to the Final | merging entries per record; ambiguous words such as "at" |

## Changelog

- v1.1 (2026-11-12): XLM-R tokenizer frozen; one split for all 14 pairs (Yulun)
- v1.1 (2026-11-13): fr/de/it/philippines re-sampled from the full CSVs and relabeled with
  Claude (LID group `lingua` → `llm`); 14-pair split done; text-free files in `data_v1.1/` (Yulun)
