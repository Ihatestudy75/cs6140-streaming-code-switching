# Methods and Decisions: Arabic, Hindi, Chinese, Japanese, Spanish and Vietnamese × English

Author: Yu-Lun Juan (yulun). Covers `ar_en`, `hi_en`, `zh_en`, `ja_en`, `es_en` and `vi_en`.

This document explains why each processing decision was made, what we found in the data, and what we suggest to the team (section 7). The shared code is unchanged: the delivery was produced with `update1_data_standard/` at `main` commit `367d6ec`, and `project_update_1/run_unicode_claude.py` rebuilds it with the renamed `project_update_1/` (same `pipeline.py`).

> Status: all six pairs are complete, pass `pipeline.py validate`, and combine without errors in `pipeline.py split`, both on their own and together with Shaoyu's `ko_en`. Twenty records per pair were reviewed with model assistance (section 3.5).

## 1. Team settings

| Item | Setting |
|---|---|
| Code | Shared `pipeline.py` unchanged (`update1_data_standard/` at commit `367d6ec`, now `project_update_1/`) |
| Tokenizer | mBERT `google-bert/bert-base-multilingual-cased`, revision `3f076fdb1ab68d5b2880cb87a0886f315b8146f8`, tokenizer_sha256 `78b44c32238e3b5e83d1deaa63c3fa1466ddf73547ce641a36f75180079ab45f` |
| Sequences | One sequence per extracted item, no concatenation |
| Label policies | `--neutral-policy inherit_previous`, `--end-policy observed` |
| Unreviewed data | `--allow-unreviewed` |
| Sampling | 500 original records per pair: `random.Random(f'6140:{pair}').sample(range(1, total+1), 500)`, the same rule as Shaoyu (`run_unicode_claude.py`) |

**Where the tokenizer comes from.** Running `prepare-tokenizer` on Windows (pinned revision) gives `28108c3b…`. The only difference is CRLF line endings in `special_tokens_map.json` and `tokenizer_config.json`; `vocab.txt` and `tokenizer.json` are already LF. After converting them to LF, `tree_hash` gives `78b44c32…`, the team value. Re-tokenizing Shaoyu's delivered `ko_en` (1,529 sequences) and `yue_en` (2,560 sequences) with this tokenizer reproduces his `labeled_tokens_shaoyu.jsonl` line for line.

## 2. Language codes

`ar`, `hi`, `zh`, `ja`, `es`, `vi`, `en`.

**Chinese uses `zh`, not `cmn`:**

- The `first_language` column of the source CSV is `Chinese` for all 8,458 records, not Mandarin. SwitchLingua treats Chinese and Cantonese as separate languages; Shaoyu handles Cantonese as `yue`.
- The text is written Chinese, not only Mandarin: about 7.9% of records are mostly in traditional characters, and about 1.2% contain Cantonese characters (`嘅`, `係`, `唔`). `cmn` would be less accurate for these records.
- The team uses two-letter codes where one exists (`ko`, `ru`, `ms`, `fr`, `de`, `it`). `yue` has three letters only because it has no two-letter code.

Reports should note that `zh` is an ISO macrolanguage that includes `yue`. In this project, `zh` means SwitchLingua's Chinese file and `yue` its Cantonese file.

## 3. Labeling methods

### 3.1 Overview

| Pairs | Method | Script | Why |
|---|---|---|---|
| ar_en, zh_en, ja_en | Unicode script rule | `annotate_script.py` | The two languages use different scripts, so the script decides the language almost deterministically |
| es_en, vi_en | Claude, word level | `annotate_claude.py` | Both languages use the Latin alphabet; a script rule cannot separate them |
| hi_en | Script rule for Devanagari, Claude for Latin-letter words | `annotate_claude.py --script-native` | Most Hindi in this file is written in Latin letters (section 3.4) |

The method follows the writing system. The goal is the same label quality for every pair, not the same tool: a script rule everywhere would label all Spanish and Vietnamese as English, and a language model everywhere would add cost and uncertainty to pairs that a rule already labels exactly.

### 3.2 Script rule (ar, zh, ja, and the Devanagari part of hi)

`project_update_1/annotate_script.py` extends Shaoyu's `annotate_unicode.py`. `char_label()`, `segments_for()`, `annotate()`, `main()`, whitespace inheritance, NEUTRAL handling, the manifest and the `validate_lid()` call are **identical to his**; only the script-range constants differ (Arabic, Devanagari and kana added; his Hangul and Cyrillic ranges are not needed here):

| Language | Native script |
|---|---|
| `ar` | Arabic (basic block, Supplement, Extended-A, Presentation Forms A/B) |
| `hi` | Devanagari (basic block, Extended) |
| `zh` | Han (same as Shaoyu's `yue`) |
| `ja` | Hiragana, Katakana, halfwidth Katakana, Han, plus `々` and `〆` |

Same conventions as Shaoyu: Latin letters, including proper nouns, are `en`; digits, punctuation and symbols are `NEUTRAL`; other scripts are `UNKNOWN`; whitespace follows the previous character. `HAN`, `HANGUL` and `CYRILLIC` are unchanged, so this script labels `yue`, `ko` and `ru` exactly as Shaoyu's does. `VERSION` stays `v1.0`; the two copies are distinguished by `script_sha256` in `lid_manifest`.

**Native punctuation and digits stay NEUTRAL.** This is not a new policy; it keeps the team rule that punctuation is NEUTRAL. The Arabic and Devanagari Unicode blocks contain their own punctuation and digits, so a plain range rule would label `؟` as `ar` and `।` as `hi`, and `validate_lid()` would accept it, because `ar` is a valid code. In `you know؟`, this would create a one-token switch after `know`. Counted in the full CSVs (previous non-space character is a Latin letter):

| Character | Total | Right after English |
|---|---:|---:|
| `،` Arabic comma | 14,889 | 754 |
| `؟` Arabic question mark | 1,236 | 886 |
| `؛` Arabic semicolon | 7 | 2 |
| `।` Devanagari danda | 5,886 | 245 |
| `・` Katakana middle dot | 729 | 13 |

That is about 1,900 fake switches. The ranges therefore exclude these characters, so `char_label()` labels them NEUTRAL by Unicode category, the same result Shaoyu's rule already gives for Chinese and Korean punctuation, which lies outside the Han and Hangul blocks:

- Arabic: `،` `؛` `؟` `۔`, Arabic-Indic digits and separators (U+0660–066D), extended Arabic-Indic digits (U+06F0–06F9)
- Devanagari: `।` `॥`, Devanagari digits, abbreviation sign (U+0964–0970)
- Japanese: `・`

**`々` and `〆` are `ja`.** `々` (`人々`, `時々`) is a letter but sits in the CJK Symbols and Punctuation block, outside the kana and Han blocks, so the plain rule would label it `UNKNOWN`. It appears 436 times in the Japanese file. Only U+3005–3006 were added, because the same block also contains `、` and `。`, which must stay NEUTRAL. The Han range is unchanged.

Both cases show the same point: Unicode block boundaries are not language boundaries, and each extension needs a manual check. `project_update_1/tests/test_annotate_script.py` covers these cases.

### 3.3 Claude word-level labeling (es, vi, and Latin-letter words in hi)

**Why not Lingua.** We first ran Lingua 2.2.0 (Yufan's tool; same span-to-character conversion as his `annotate_lingua.py`; the trial script is not part of the delivery) on the 20 es_en pilot records (64 items). In 30 of the 64 items (47%), at least one word was wrong, about 45 words in total. The errors cluster at switch points:

- English tags at the end of a Spanish sentence (`…la comunidad, right?`, `you know?`, `anyway`) were merged into the Spanish span.
- Spanish tags after an English sentence (`¿verdad?`, `¿no?`) were merged into the English span.
- One-letter words such as `y` were labeled English.

Lingua scores character n-grams against models that ship with the package and were trained on large monolingual texts; nothing is learned from our data or our sample. Short segments give few n-grams, words such as `no` and `a` exist in both languages, and a short word after a long sentence is not enough evidence to start a new segment. In our samples, 36–45% of records are tagged `Extra-sentential / Tag switching`, so these errors fall exactly on the positions we predict. On the same 64 items, a model-assisted review of Claude's word-level labels found no errors (section 3.5 explains what this does and does not show).

**Pipeline (`project_update_1/annotate_claude.py`).**

1. `prepare`: the script splits each item into words (runs of letters, including combining marks and word-internal apostrophes and hyphens) and writes batches of 100 items. Each item carries its full text and its word list.
2. Labeling: Claude returns exactly one code per word (`n` native, `e` English, `u` UNKNOWN). It never writes or copies the text. The rules are in `project_update_1/claude_lid_instructions.md`, a word-level version of the shared `LID_PROMPT.md` v1.0. Each pair has 16 batches, labeled by 8 Claude subagents in parallel.
3. `assemble`: the script maps the codes back onto the original text by position. Whitespace, punctuation, symbols and digits are `NEUTRAL`, the same rule as the Lingua conversion. An item whose code count does not match its word count is an error; nothing is patched. Because segments are cut from the original string, they always reproduce it exactly.
4. Harmonization (`annotate_claude.py collect` / `apply`): the parallel labelers applied the name and loanword rules differently (in es_en, `fans` was `en` 21 times and `es` 8 times; some batches labeled every person name `u`, others labeled English names `en`). The script collects every `u` word, every capitalized word that does not start the item, every English word between two native words, and every lowercase word that received both codes; one subagent relabels all of them in context under one rule. Each pair also received the decision table of the previous pair, so the same name gets the same code across pairs. Every change is recorded in `harmonize_changes_yulun.jsonl` (sequence ID and character offsets, no text).

| Pair | Harmonization candidates | Changed |
|---|---:|---:|
| es_en | 1,922 | 123 |
| vi_en | 2,519 | 75 |
| hi_en | 2,604 | 70 |

After harmonization, the es_en capitalized words with conflicting labels dropped from 22 types to 5; the remaining 5 (`X`, `S`, `A`, `No`, `I`) differ correctly by context, for example `generación X` versus the model name `X-37B`.

**Rules for names and loanwords** (rules 4 and 5 in `claude_lid_instructions.md`):

- Names are judged by their written form, not by the sentence or the person's nationality: English words or an English form → `en` (`Taylor Swift`, `Washington State`, `NASA`); a native form → native (`Estados Unidos`, `Hoa Kỳ`, `Narendra Modi`); the same spelling in both languages, or a third language → `UNKNOWN` (`Musk`, `Trump`, `Sony`, `Texas`, `China`).
- Loanwords: an English word kept in its English form → `en` (`fans`, `show`, `game`); an established native word or native spelling → native (`golf`, `tivi`, `crore`, `yoga`); words spelled the same in both languages that are not names follow their clause.

**How the labelers worked.** The context for each decision is the full text of one item (one or two sentences); in hi, Devanagari text is part of that context. Labelers did not use dictionaries, web search or other models. According to their reports:

- Most labelers wrote their decisions in a short form (for example `n6 e2`) and used a small helper script to expand it.
- Three Vietnamese labelers pre-filled codes with a heuristic (Vietnamese diacritics → `n`, plain ASCII → `e`) and then corrected every item by hand. They reported fixing unaccented Vietnamese syllables (`tham`, `phim`, `Sau khi`) and `AI` read as Vietnamese `ai`. This carries a higher risk of missed errors, so the automatic checks below were added.
- Several helper scripts in a shared scratch folder overwrote each other; the affected labelers rebuilt their files from their own decision files. The ID and count checks would reject any file mixed up between batches.

**Self-checks by each labeler.** Each labeler ran a check that every item has as many codes as words and that IDs match the input. Each also listed word/code pairs to find labels shifted by one position, which a count check cannot catch. Together, the labelers found and fixed more than ten shifted items.

**Automatic checks that do not depend on the model:**

| Check | Result |
|---|---|
| Label files have the same IDs, in the same order, as the inputs | Correct for all batches |
| es: Spanish tags (`¿verdad?`, `¿no?`) labeled `en`; English tags at the end of an item labeled `es` | 2 errors found (Spanish `en` before an English show title; Spanish `quién` inside an English clause), fixed during harmonization |
| vi: words with Vietnamese diacritics labeled `en` or `u`; common English words labeled `vi` | 1 error found (`nhật` in `Chủ nhật`), fixed; 0 English words. Diacritics are dense, so a shifted label would almost certainly show up here |
| hi: common Hindi function words (`hai`, `ki`, `mein`…) labeled `en`; common English words labeled `hi` inside English stretches | 0 real errors |

**Cost.** Claude Code subscription quota; no API charges. About 1.4 million tokens per pair for labeling and harmonization.

**Reproducibility.** Running Claude again on the same input may not give exactly the same labels: the model samples its output, there is no seed to fix, model versions change, and batch composition affects context. The eight labelers themselves disagreed on some names and loanwords. A script rule, or Lingua at a fixed version, gives the same output every time. The Claude labels are therefore **traceable but not reproducible**: the delivered labels, stored as character offsets in `lid_offsets_yulun.jsonl`, are the official version (as with Shaoyu's Malay), and the instructions, every harmonization change, `annotation_source` (model, method, prompt version, date) and the file hashes in each manifest are kept.

### 3.4 Hindi in two scripts

Much of the Hindi file is written in Latin letters, for example (constructed) `Ye plan sach mein accha hai, especially for the new team.` Of all 8,381 records, only 40.6% contain any Devanagari letter, and the two writing styles rarely mix within one record.

A script rule labels Latin-letter Hindi as `en`. The first delivery (commit `687072f` in git history) therefore sampled only records containing Devanagari. That dropped 59.4% of the data and represented only one writing style, so hi_en was redone:

- Sample 500 records from the whole file, as for every other pair (`run_unicode_claude.py`).
- `annotate_claude.py --script-native`: Devanagari characters keep the `annotate_script.py` labels (identical to the first delivery); only words made entirely of Latin letters go to Claude.
- On the same 500 records, the script rule alone labeled about 30K characters as Hindi and 118K as English; the final labels have 61K Hindi and 77K English characters.

Devanagari is never used for English, so the rule is exact there. Latin-letter words that exist in both languages are decided by context: `main` (मैं), `the` (थे), `is`/`us` (इस/उस), `hi` (ही), `to` (तो), `par`, `do` (दो).

### 3.5 Pilot review

Same procedure as Shaoyu:

- 20 records per pair: `random.Random(f'6140:{pair}:pilot').sample(sorted(sampled record_ids), 20)`. This rule reproduces Shaoyu's pilot records for all three of his checked pairs. All items of these records are marked `reviewed`.
- Record IDs are in `<pair>/annotation/pilot_record_ids_yulun.txt`. They must be forced into train with `--train-records`, because they were seen during development.
- Every segment of every item was read: switch boundaries, tags, homographs, punctuation and digits, proper nouns.
- Fixes would be applied by the review step of `run_unicode_claude.py` (same logic and output format as Shaoyu's runner) and logged in `review_yulun.csv` and `build_manifest_yulun.json`.

| Pair | Items reviewed | Errors found |
|---|---:|---:|
| ar_en | 72 | 0 |
| hi_en | 70 | 0 |
| zh_en | 81 | 0 |
| ja_en | 68 | 0 |
| es_en | 64 | 0 |
| vi_en | 61 | 0 |

**What this shows.** Zero errors in 60–80 items means the share of items with an error is probably below about 4–5% per pair (rule of three, 3/n). Items from the same record are correlated, so the real bound is somewhat higher.

**What it does not show.** The review was done by Claude, delegated by yulun (see the `reviewer` column). For es, vi and hi, Claude also labeled the data, so the review is **not independent**: a systematic model error could be repeated in the review. Reports must not call it human review. An independent check, by someone who speaks the language or by a second model with a human checking only the disagreements, is still needed. All other items are `unreviewed` pseudo-labels.

## 4. Statistics

| Pair | Method | Records | Sequences | Valid positions | Switches | Switch rate | Small / Medium / Large | Large | Switches without duration | UNKNOWN subwords |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| ar_en | Script rule | 500 | 1,594 | 40,784 | 1,936 | 4.7% | 351 / 605 / 935 | 49% | 45 | 64 |
| zh_en | Script rule | 500 | 1,747 | 54,032 | 2,402 | 4.4% | 379 / 668 / 1,283 | 55% | 72 | 163 |
| ja_en | Script rule | 500 | 1,662 | 52,918 | 2,300 | 4.3% | 374 / 567 / 1,254 | 57% | 105 | 194 |
| hi_en | Script rule + Claude | 500 | 1,571 | 42,575 | 6,198 | 14.6% | 2,661 / 2,031 / 1,325 | 22% | 181 | 676 |
| es_en | Claude | 500 | 1,561 | 37,355 | 2,319 | 6.2% | 417 / 748 / 1,016 | 47% | 138 | 642 |
| vi_en | Claude | 500 | 1,595 | 39,495 | 2,701 | 6.8% | 640 / 935 / 932 | 37% | 194 | 819 |

Every sampled record has items, and there are no extraction errors within the samples (hi_en has 3 and zh_en 1 `possible_wrapping_or_escape` warning; following the shared rule, they are logged and the text is unchanged).

Sequences by switch type (record-level `cs_type`):

| Pair | Intersentential | Intrasentential | Extra-sentential / Tag |
|---|---:|---:|---:|
| ar_en | 431 | 531 | 632 |
| zh_en | 554 | 473 | 720 |
| ja_en | 492 | 551 | 619 |
| hi_en | 519 | 489 | 563 |
| es_en | 453 | 488 | 620 |
| vi_en | 418 | 452 | 725 |

Single-language sequences: ar_en 250, zh_en 106, ja_en 69, hi_en 49, es_en 46, vi_en 65. Spot checks show these are standalone single-language sentences in the dialogue, not labeling errors.

**Where UNKNOWN subwords come from:**

| Pair | `[UNK]` token | Character labeled UNKNOWN | Punctuation after UNKNOWN |
|---|---:|---:|---:|
| ar_en | 64 | 0 | 0 |
| zh_en | 137 | 0 | 26 |
| ja_en | 141 | 0 | 53 |
| hi_en | 90 | 497 | 89 |
| es_en | 76 | 455 | 111 |
| vi_en | 89 | 633 | 97 |

In hi, es and vi, "character labeled UNKNOWN" is almost entirely proper nouns under the section 3.3 rule (`Trump`, `Musk`, `Google`, `Samsung`…). This is a labeling convention, not a tokenizer issue; the script-rule pairs label Latin-letter names `en`, so they have none.

## 5. How the duration labels are measured

Duration is the length of the next language burst in mBERT subwords, binned 1–2 / 3–6 / 7+ by the shared `labels()`. Two measurement choices push bursts toward Large: inherited punctuation counts toward the length, and one word can be several subwords. To separate them, we recounted the same bursts four ways from `labeled_tokens` (no relabeling or re-tokenizing; a "word" is a subword that does not start with `##`, and a token whose `alignment_notes` is `neutral_inherited_previous` is inherited punctuation):

| Pair | Subwords per native word | Large: subwords (current) | Large: words | Large: subwords, no punctuation | Large: words, no punctuation | Bursts changing class without punctuation |
|---|---:|---:|---:|---:|---:|---:|
| ar_en | 2.09 | 49% | 40% | 45% | 35% | 17% |
| hi_en | 1.62 | 22% | 16% | 19% | 13% | 9% |
| zh_en | 1.00 | 55% | 53% | 51% | 48% | 15% |
| ja_en | 1.48 | 57% | 49% | 54% | 45% | 14% |
| es_en | 1.29 | 47% | 41% | 41% | 36% | 18% |
| vi_en | 1.10 | 37% | 35% | 33% | 30% | 20% |

Findings:

1. **The tokenizer adds 2–9 points of Large**, more for languages that mBERT splits finely (Arabic, Japanese) and less for Vietnamese and Chinese.
2. **Punctuation moves 9–20% of bursts to another class.** Most bursts end with punctuation; Hindi is least affected because many of its bursts are single English words inside a Hindi sentence.
3. **The two effects add up and do not cancel**: together they lower Large by 7–14 points depending on the pair.
4. **Most bursts are English** (58–79%), and English splits into about 1.15–1.36 subwords per word in every pair. Native-language fertility therefore affects only a minority of bursts, and subwords per word alone does not explain the differences between pairs.
5. **Differences between pairs survive both corrections** (Large 13–48%), so they come mainly from the data: how switches are written in each pair (Hinglish often inserts single English words; median English burst 2 words), how SwitchLingua generated the text, and labeling conventions.

Limits: the word count is an approximation (punctuation tokens are single words in the "words" column), and Chinese and Japanese have no spaces, so their word counts are really character counts.

**Why we keep `inherit_previous`.** The alternative policy, `mask`, turns punctuation-only tokens into `UNKNOWN`. Simulated on ar, hi, es and vi, it removes 24–61% of switches (es_en 2,319 → 899) and 46–85% of switches with a valid duration, because switches happen next to punctuation (sentence boundaries, tags). Large nearly disappears (4–7%), because long bursts end with a period. That is survivorship, not better measurement. `inherit_previous` keeps every switch; its cost is the punctuation length inflation above.

## 6. Limitations

1. **Labels for es, vi and hi were produced by Claude and are not reproducible**; the delivered labels (`lid_offsets_yulun.jsonl`) are the official version. The review of these pairs was also done by Claude and is not independent.
2. **Proper nouns labeled `UNKNOWN` invalidate nearby positions.** hi, es and vi have far more UNKNOWN subwords (642–819) than the script-rule pairs (64–194), and more switches without a duration. This follows the shared prompt ("unjudgeable names are UNKNOWN"), as in Shaoyu's Malay.
3. **The same name can be labeled differently across pairs.** The rule judges a name's form relative to the pair: `Kamala` is `UNKNOWN` in es and vi but a Hindi name, `hi`, in hi.
4. **Place names are not perfectly consistent.** For example, in es_en `Washington` and `Pennsylvania` are `en` while `Michigan` and `Tennessee` are `UNKNOWN`; about a dozen words.
5. **The two methods differ in one convention.** Script-rule pairs label every Latin-letter name `en` (as Shaoyu does); Claude-labeled pairs follow section 3.3. Keep this in mind when comparing across methods.
6. **mBERT has no curly quotes or dashes.** `’` `“` `”` `‘` `–` `—` become `[UNK]`, so `It’s` is tokenized `It [UNK] s` (straight `It's` is fine). This is the main source of `[UNK]` in every pair and affects all team members' English text.
7. **Duration labels depend on measurement choices.** See section 5: subword counting and inherited punctuation both push toward Large, and part of the duration difference between pairs comes from them.
8. **hi_en has the highest switch rate (14.6%) and the most Small bursts (44%).** Latin-letter Hindi often inserts single English words (`match`, `plan`, `team`), and each insertion is two switches and a Small burst. This is a property of the data.
9. **Katakana loanwords are `ja`.** `コンピューター` is an English loanword but is labeled `ja` by script. Known limitation, not an error.
10. **Cantonese characters in the Chinese file are `zh`** (about 1.2% of records); `哋`, `咗`, `喺` become `[UNK]`, but rarely.
11. **Other scripts are `UNKNOWN`.** A few Greek letters appear in the Japanese and Chinese files, and some Telugu in the Hindi file. Some Devanagari words with nukta (`खिलाड़ी`, `बढ़`) are tokenized as a single `[UNK]`.
12. **Full-width Latin letters in Japanese become one `[UNK]` together with the following kana** (`ＥＭＡは`), rarely.
13. **Chinese and Japanese have no spaces.** A long run of Han characters is one LID segment; this does not affect labeling, because the tokenizer handles segmentation.
14. **Per-item processing loses switches between items**, as for Shaoyu and Yufan (see Shaoyu's DECISIONS.md, section 4).

## 7. Suggestions for the team (not applied)

The delivered data does **not** use any of these changes.

| Suggestion | How | Effect |
|---|---|---|
| Measure duration without inherited punctuation, and by words | Keep `language_ids`, switch labels and the 1–2 / 3–6 / 7+ bins; when counting a burst, skip tokens whose `alignment_notes` is `neutral_inherited_previous` and count word starts. This can first be computed from existing files; adopting it means adding a policy option to `labels()`, recording it like `end_policy`, and having `split` reject mixed settings | Removes the two measurement effects in section 5. Shifting the bin thresholds instead would be unfair across pairs, because punctuation adds a token to some bursts and none to others (57% of Hindi bursts have none). Chinese and Japanese would need a word segmenter |
| Pilot context-aware word-level LID on a sample of the Lingua pairs | Relabel about 20 records per pair with `annotate_claude.py` (add the language codes) and compare switch counts and duration distributions | Measures how much missed short switches and filtering change the Lingua pairs |
| Agree on one proper-noun rule | Options: all `UNKNOWN`; judged by form (current Claude rule); or treat names as NEUTRAL so they inherit the surrounding language | Today script-rule pairs label names `en` and prompt-based pairs often `UNKNOWN`, which changes switch counts and valid positions between pairs |
| Replace curly quotes and dashes only in the copy that is tokenized | Map `’`→`'`, `“”`→`"`, `–—`→`-` before calling the tokenizer; every mapping is one character to one character, so offsets still match the original text, which stays unchanged | Removes most `[UNK]` in section 6, item 6 |
| Compare a tokenizer with better coverage (XLM-R) | Separate experiment; all labels must be regenerated, and `split` does not allow mixing tokenizers | Addresses Cantonese `[UNK]` and fertility differences |
| Independent review of the pilot records | A speaker of the language, or a second model with a human checking disagreements | Turns the model-assisted review into a real accuracy estimate |
