# Relabeling fr/de/it/philippines for data-v1.1

Owner: Yulun (on behalf of Yufan, the Update 1 owner of these pairs) · Date: 2026-11-13 ·
Pairs: `fr_en`, `de_en`, `it_en`, `philippines_en` (Tagalog/Filipino)

In Update 1 these four pairs were labeled with Lingua 2.2.0, screened, and sampled from the
screened pool. For data-v1.1 they were rebuilt with the same procedure as the other LLM pairs
(es_en, vi_en, hi_en; [`deliverables/yulun/DECISIONS.md`](../deliverables/yulun/DECISIONS.md)
section 3.3): 500 records sampled from the **whole** CSV, word-level Claude labels, one
harmonization pass per pair, a model-assisted review of 20 pilot records, and tokenization with
both frozen tokenizers. The other ten pairs are unchanged.

## 1. Why

Lingua scores character n-grams and needs long segments, so it misses short insertions
(`you know`, `right?`, a single English noun). Its v1.1 data had almost no short switches:

| | Small (1–2 subwords), mBERT | Small, XLM-R |
|---|---:|---:|
| fr/de/it/philippines, Lingua | 3.0–7.3% | 2.4–7.2% |
| The other 10 pairs | 16.3–44.2% | 14.8–47.9% |

Small switches are exactly what the duration head must predict, so the four pairs were not
comparable with the rest. Screening made it worse: it removed the sequences where Lingua had
missed an English phrase, and the Update 1 sample was drawn from what was left. In the new
sample, 21–33% of the sequences would have been screened out (they are not in Yufan's retained
pool: fr 33%, de 29%, it 23%, philippines 21%).

## 2. Procedure

Tool: [`relabel_v1_1.py`](relabel_v1_1.py) (Update 1 environment, run from the repository root).
Outputs with text stay in `project_update_1/data/v1.1_relabel/` (ignored by Git).

| Step | What | Same as Update 1? |
|---|---|---|
| `extract` | `pipeline.py extract` on the full CSV; sample 500 records with `random.Random('6140:<pair>').sample(range(1, total+1), 500)`; 20 pilot records with seed `6140:<pair>:pilot`; `cs_type` per sequence | Yes (`run_unicode_claude.py` functions reused). CSV hashes equal Yufan's Update 1 files |
| `annotate_claude.py prepare` | batches of 100 items, one word list per item | Yes |
| Label | Claude subagents return one code per word (`n` native, `e` English, `u` UNKNOWN), never the text | Yes; **2 labelers per pair** (Update 1: 8) |
| Rules | [`claude_lid_instructions.md`](../project_update_1/claude_lid_instructions.md) rules 1–6 + [`claude_lid_instructions_v1_1.md`](claude_lid_instructions_v1_1.md) | Same rules; the addendum only adds French/German/Italian/Tagalog examples |
| `collect` | harmonization candidates: every `u`; capitalized non-initial words; an `e` word between two `n` words; lowercase words with both codes | Yes, with three changes (below) |
| Decide | one harmonizer per pair decides every candidate in context under [`claude_harmonize_policy_v1_1.md`](claude_harmonize_policy_v1_1.md) | Update 1 ran the pairs in sequence, each with the previous pair's decision table; here the four ran in parallel with one written policy built from the es/vi/hi decisions |
| `build-lid` | decisions + `extra_fixes.jsonl` → `annotate_claude.py apply` → `assemble` | Yes |
| Review | model-assisted review of every word of the 20 pilot records | Yes |
| `finalize` | review applied; `lid_offsets.jsonl`, `review.csv`; `pipeline.py tokenize` with both tokenizers (`inherit_previous`, `observed`) | Yes |

Changes to `collect`:
1. **Short homographs per language.** Update 1 used one Spanish list (`a`, `me`, `no`…) for es,
   vi and hi. Each pair now has its own list (French `en`, `on`; German `in`, `was`; Italian `e`,
   `come`; Tagalog `at`, `may`…), so these words are not sent to harmonization on every occurrence.
2. **German capitals.** Every German noun is capitalized, so a capitalized German word is a
   candidate only if it is coded `e`/`u`, or if the same capitalized word got different codes.
3. **Loanword audit (new).** A word coded `n` that is coded `e` at least 3 times anywhere in the
   four pairs is also a candidate. Without it, a labeler who always codes e.g. `interview` as
   native would never be checked.

### Names and loanwords (where labelers disagreed)

The policy keeps the Update 1 rules and decisions:
- **Names** are judged by form, relative to the pair. A name of native origin is native
  (`Macron` in fr_en, `Berlin` in de_en, `Duterte` in philippines_en), as `Kamala` was `hi` in
  hi_en. An English form is `en`. A name spelled the same in both languages, or from a third
  language, is `UNKNOWN`. Names that appear in the Update 1 decisions keep their code:
  `Trump`, `Musk`, `Google`, `Samsung` `UNKNOWN`; `Apple`, `Microsoft`, `NASA` `en`.
- **Loanwords:** an English word in English spelling with no native inflection is `en`, as
  `fans` and `show` were in es_en (`team`, `interview`, `design`, `weekend`, `match`). Only very
  old, fully native loans are native (`sport`, `film`, `hotel`, `internet`). Native inflection
  or spelling makes a word native (`nag-shopping`, `gecheckt`, French `matchs`). A compound of
  an English and a native part with no native inflection is `UNKNOWN` (`LID_PROMPT.md` rule 3).
- **Third languages** are `UNKNOWN` (three philippines_en items are mostly Hiligaynon).

### Corrections after harmonization (`extra_fixes.jsonl`)

| Source | fr | de | it | ph |
|---|---:|---:|---:|---:|
| Shifted word next to a fixed candidate, not itself a candidate (reported by the harmonizer) | 4 | 0 | 3 | 3 |
| Cross-pair check: `Usa` spelled the same in Italian and English → `UNKNOWN` (as `USA` in de_en) | – | – | 2 | – |
| Loanword sweep after the pilot review found `leader` coded native (policy C words in English form) | 5 | 2 | 7 | 0 |

Every fix is applied by `apply`, so it is recorded in `harmonize_changes.jsonl` like the
harmonizer's own changes.

## 3. Results

| Pair | Sequences | Harmonization candidates → changed | Extra fixes | Pilot items reviewed | Review errors |
|---|---:|---:|---:|---:|---:|
| fr_en | 1,524 | 4,343 → 189 | 9 | 64 | 0 |
| de_en | 1,437 | 1,844 → 102 | 2 | 64 | 0 |
| it_en | 1,492 | 3,604 → 146 | 12 | 57 | 1 |
| philippines_en | 1,525 | 5,423 → 146 | 3 | 57 | 0 |

Counts include the extra fixes. The one review error (`leader`, coded native) was fixed and
led to the loanword sweep above.

Lingua (Update 1 sample) vs Claude (v1.1):

| Pair | Version | Switch rate mBERT / XLM-R | Small / Medium / Large, mBERT | Small / Medium / Large, XLM-R |
|---|---|---|---|---|
| fr_en | Lingua | 4.1 / 4.2% | 3.0 / 25.4 / 71.7% | 2.4 / 26.4 / 71.2% |
| | Claude | 5.3 / 5.3% | 16.7 / 32.5 / 50.8% | 18.5 / 30.3 / 51.2% |
| de_en | Lingua | 4.0 / 4.3% | 3.2 / 23.9 / 72.9% | 3.1 / 25.0 / 71.9% |
| | Claude | 6.1 / 6.6% | 20.2 / 34.2 / 45.6% | 23.1 / 31.7 / 45.1% |
| it_en | Lingua | 4.3 / 4.5% | 5.5 / 30.0 / 64.4% | 5.5 / 30.8 / 63.8% |
| | Claude | 5.4 / 5.7% | 18.7 / 31.2 / 50.1% | 21.2 / 29.0 / 49.8% |
| philippines_en | Lingua | 5.3 / 5.9% | 7.3 / 35.2 / 57.5% | 7.2 / 38.4 / 54.5% |
| | Claude | 11.6 / 12.9% | 39.1 / 32.3 / 28.6% | 40.9 / 33.1 / 25.9% |

French, German and Italian now look like es_en, ru_en and ar_en. philippines_en mixes inside
sentences as densely as hi_en and ms_en. UNKNOWN subwords rise from 0 to 631–1,045 (mBERT):
Lingua never outputs UNKNOWN, while the Claude rules label names spelled the same in both
languages `UNKNOWN`, as in es_en (642) and vi_en (819). Full table:
[`data_v1.1_stats.md`](data_v1.1_stats.md).

## 4. Checks

| Check | Result |
|---|---|
| Label files: same ids and order as the inputs, one code per word, codes `n`/`e`/`u` only | All 62 batches |
| Harmonization decisions cover every candidate exactly | All 4 pairs |
| `validate_lid`: segments reproduce the text exactly | All 5,978 sequences |
| Function words (independent of the model): native function words coded `en`; English function words coded native | 0 errors. The hits are correct homographs: German `was` ("what"), French `but` ("goal"), names such as `De Zerbi`, `La Liga` |
| `pipeline.py validate`, tokenizer hashes | PASS, both backbones, all 4 pairs |
| XLM-R spot check | A lone `▁` takes the next subword's language; `nagwo-work` is Tagalog |
| 14-pair split | identical for both backbones (`compare-splits` PASS); pilot records in train |

## 5. Limitations

- **The review is not independent.** Claude labeled and reviewed. 0–1 errors in 57–64 items
  bound the item error rate at roughly 5% per pair (rule of three), but a systematic model error
  could repeat in the review. Reports must not call it a human review.
- **Not reproducible run to run.** Claude samples its output; the delivered character offsets
  (`lid_offsets.jsonl`) are the official labels, as for es/vi/hi and ms.
- **Loanword line.** Where an established loan ends and an English switch begins is a judgment
  call. The policy fixes one line for all four pairs (in line with es_en), but other lines are
  defensible and would move some Small switches.
- **Hiligaynon.** Three philippines_en items are mostly Hiligaynon; its words are `UNKNOWN`.

## 6. Files

Text-free, in [`data_v1.1/relabel/<pair>/`](data_v1.1/relabel/):

| File | Content |
|---|---|
| `lid_offsets.jsonl` | **official labels**: `[start, end, lang]` character spans per sequence, with `text_sha256` |
| `sampled_record_ids.jsonl`, `pilot_record_ids.txt` | the 500 sampled records, the 20 pilot records |
| `sequence_index.jsonl` | every sequence's ids, source row and `text_sha256` (no text) |
| `cs_type.jsonl` | `cs_type` of each sequence's record (evaluation metadata) |
| `harmonize_changes.jsonl`, `extra_fixes.jsonl` | every harmonization change and extra fix (ids, word index, offsets, codes) |
| `review.csv` | one row per pilot item (or per fix): reviewer, status, issue type, codes, offsets |
| `extract_issues.jsonl` | extraction warnings (ids only) |
| `*_manifest.json` | inputs, counts, hashes of every step |

To rebuild the labels without Claude: take the texts from `pipeline.py extract` on the CSV,
check `text_sha256`, and cut each text at the spans in `lid_offsets.jsonl` (as
`run_unicode_claude.py` does for es/vi/hi).
