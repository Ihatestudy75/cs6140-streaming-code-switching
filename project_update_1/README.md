# Update 1: Data Processing Standard and Run Instructions

Choose the section for the language pairs you want to process. Each contributor's entry point, method, dependencies, and run commands belong in that section. Common tools and conventions appear under the shared standard below.

- [French–English, German–English, Italian–English, and Filipino/Tagalog–English](#fr-de-it-ph-en)
- [Cantonese–English, Korean–English, Russian–English, and Malay–English](#yue-ko-ru-ms-en)
- [Additional language pairs — Contributor 3](#contributor-3)
- [Shared data-processing standard](#shared-standard)

<a id="fr-de-it-ph-en"></a>
## French–English, German–English, Italian–English, and Filipino/Tagalog–English

**Method:** Lingua language-span annotation with frozen mBERT subword alignment.
**Run all commands in this section from the repository root**, not from inside `project_update_1/`.
This section describes only these four configured language pairs.

| Script | Role for these language pairs |
|---|---|
| [`run_lingua.py`](run_lingua.py) | Complete processing entry point |
| [`annotate_lingua.py`](annotate_lingua.py) | Language-span annotation helper, invoked by the entry point |

The entry point reuses the shared `pipeline.py`; it does not train a baseline or implement the other contributors' annotation methods.

### Inspect the code without generating data

```bash
python3 project_update_1/run_lingua.py --help
cd project_update_1
python3 -m unittest discover -s tests -v
cd ..
```

An existing delivery is not required for these checks. Optional delivery verification below applies only after the specified output has been generated or otherwise obtained.

### Required repository files

```text
repository/
  datasets/                       # CSV files are not in Git; see datasets/README.md
    French_eng.csv
    German_Eng.csv
    Italian_eng.csv
    Philippines_eng.csv
  project_update_1/
    run_lingua.py
    pipeline.py
    annotate_lingua.py
    requirements.lock.txt
  deliverables/
    yufan/                        # Default final output; never overwritten
```

Keep these scripts together; `run_lingua.py` is an entry point that reuses the shared implementation, not an isolated replacement for it. The four CSV filenames and their paths relative to the repository root are fixed in the runner. No machine-specific absolute input paths are needed.

Use Python 3.13 when possible; the entry point requires Python 3.11 or newer. You need Python with `venv` and `pip`, internet access for the initial package/tokenizer downloads, and the four CSV files. The CSV files are not in Git (the SwitchLingua licence forbids redistribution); download them from Hugging Face as described in [`datasets/README.md`](../datasets/README.md). The processing workflow requires no API key, GPU, PyTorch, or Transformer model weights. Lingua's language models are supplied with its installed package.

### 1. Create the environment and prepare the frozen tokenizer

```bash
python3 project_update_1/run_lingua.py --setup-only
```

This creates `project_update_1/.venv/` if absent, installs `requirements.lock.txt` and `lingua-language-detector==2.2.0`, and downloads only the frozen mBERT tokenizer if missing. It reuses an existing matching tokenizer after checking its revision and checksum. The runner launches the environment's Python itself, so manual activation is unnecessary.

The frozen tokenizer configuration is:

- Model: `google-bert/bert-base-multilingual-cased`.
- Revision: `3f076fdb1ab68d5b2880cb87a0886f315b8146f8`.
- Tokenizer SHA-256: `78b44c32238e3b5e83d1deaa63c3fa1466ddf73547ce641a36f75180079ab45f`.
- Asset directory: `project_update_1/assets/mbert/`.
- Policies: `inherit_previous` for neutral tokens and `observed` for text endings.

Success is reported as `SETUP COMPLETE: pinned dependencies and frozen tokenizer ready.` An informational message that PyTorch, TensorFlow, or Flax is missing does not prevent tokenizer-only processing.

### 2. Generate the four-pair delivery

For a fresh checkout without an existing `deliverables/yufan/`:

```bash
python3 project_update_1/run_lingua.py --skip-install
```

If `deliverables/yufan/` already exists, preserve it and reproduce into a new directory:

```bash
python3 project_update_1/run_lingua.py \
  --skip-install \
  --out deliverables/yufan_reproduction
```

`--skip-install` reuses the prepared environment and local tokenizer. It does not skip extraction, annotation, tokenization, screening, sampling, or validation.

The entry point performs these steps:

1. Extract original text and source IDs from all four CSV files, logging extraction issues.
2. Defer entries matching extraction issues and remove exact duplicate texts before Lingua annotation.
3. Generate unreviewed Lingua language spans, restricted to the source language and English.
4. Align spans to frozen mBERT subwords with character offsets, then generate switch/duration targets and validity masks.
5. Isolate suspicious whole sequences without editing their text or labels. Rules cover suspected missed short-English phrases, uncertain alignment, unavailable duration targets, and context-review candidates for English-labeled Filipino `at`. The 58 historical French pilot-disagreement exclusions are embedded for reproducibility; they are not human-verified truth.
6. Sample 500 source records per pair from the screened retained pool with `random.Random(f"6140:{pair}").sample(sorted(record_ids), 500)`. Include all remaining sequences of selected records.
7. Retrieve `cs_type` from the original CSV metadata, check source correspondence and target consistency, calculate statistics and checksums, and publish the completed output.

For the Philippines source label, Lingua's `TAGALOG` option maps to the project code `philippines`, following SwitchLingua's use of "Filipino (Tagalog)." This documents the detector choice, not annotation accuracy.

Working outputs, complete retained/quarantined partitions, and stage checkpoints default to `project_update_1/data/yufan_reproduction/`. Only the sampled delivery is published under `deliverables/yufan/` or `--out`. Nothing is automatically staged, committed, pushed, deleted, or overwritten.

### 3. Verify a delivery without changing it

This step is optional and requires the specified generated delivery to exist.
If this checkout contains scripts and raw inputs only, skip this step. A missing
delivery directory does not mean that the code checks have failed.

```bash
python3 project_update_1/run_lingua.py \
  --verify deliverables/yufan
```

For a reproduction, substitute `deliverables/yufan_reproduction`. Verification uses the Python standard library and shared validation code; it does not install packages or access the network. It checks checksums, aligned IDs, source fields, LID reconstruction, target rules, statistics, and selected record IDs. A PASS confirms structure and provenance, **not linguistic accuracy**.

### 4. Optional small end-to-end check

```bash
python3 project_update_1/run_lingua.py \
  --skip-install \
  --extract-limit 50 \
  --records 5 \
  --work-dir project_update_1/data/yufan_smoke_new \
  --out project_update_1/data/yufan_smoke_delivery_new
```

This processes the first 50 source records per pair and selects five retained records per pair. It is a workflow check, not the formal 500-record delivery. Use unused paths if these directories already exist. The check fails explicitly if fewer than five retained source records remain.

### Output layout and expected interpretation

```text
deliverables/yufan/
  README.md
  requirements_yufan.txt
  runtime_yufan.json
  summary_yufan.json
  file_inventory_yufan.json
  scripts/
  <pair>/
    cs_type_yufan.jsonl
    extraction/
      texts_yufan.jsonl
      lid.template_yufan.jsonl
      extract_issues_yufan.jsonl
      extract_manifest_yufan.json
      sampling_manifest_yufan.json
    annotation/
      lid_yufan.jsonl
      lid_manifest_yufan.json
      screening_manifest_yufan.json
      quarantine_reasons_yufan.jsonl
      review_yufan.csv
      pilot_record_ids_yufan.txt
      development_record_manifest_yufan.json
    tokenized/
      labeled_tokens_yufan.jsonl
      token_issues_yufan.jsonl
      stats_yufan.json
      manifest_yufan.json
```

The existing sampled delivery contains 1,143 French--English, 1,145 German--English, 1,299 Italian--English, and 1,321 Filipino/Tagalog--English sequences: 4,908 sequences across 2,000 source records. The runner computes counts from the actual inputs and rules rather than forcing these totals. All LID remains `unreviewed`; screening can miss errors and bias the retained distribution.

Completed working stages are reused only when their recorded inputs, configuration, code, and output checksums match. After an incomplete run or a configuration/code change, choose a fresh `--work-dir` and a fresh `--out` rather than deleting existing data. `--records`, `--seed`, `--pairs`, `--venv`, and `--repo-root` provide explicit overrides. See all options with:

```bash
python3 project_update_1/run_lingua.py --help
```

<a id="yue-ko-ru-ms-en"></a>
## Cantonese–English, Korean–English, Russian–English, and Malay–English

**Method:** Unicode script-rule LID for Cantonese, Korean, and Russian. DeepSeek LLM LID with the shared `LID_PROMPT.md` v1.0 for Malay. Frozen mBERT subword alignment for all four pairs.
**Run all commands in this section from the repository root**, not from inside `project_update_1/`.

| Script | Role for these language pairs |
|---|---|
| [`run_unicode_deepseek.py`](run_unicode_deepseek.py) | Complete processing entry point |
| [`annotate_unicode.py`](annotate_unicode.py) | Script-rule LID for `yue_en`, `ko_en`, `ru_en`. The entry point calls it. |
| [`annotate_deepseek.py`](annotate_deepseek.py) | DeepSeek LID for `ms_en`. The entry point calls it only with `--ms-lid deepseek`. |

The entry point reuses the shared `pipeline.py` for extraction, tokenization, and target generation. It does not change `pipeline.py`.

### Inspect the code without generating data

```bash
python3 project_update_1/run_unicode_deepseek.py --help
cd project_update_1
python3 -m unittest discover -s tests -v
cd ..
```

### Required repository files

```text
repository/
  datasets/
    Can_Eng.csv                   # Not in Git; download it (see below)
    Korean_eng.csv                # Not in Git
    Russian_eng.csv               # Not in Git
    Malay_eng.csv                 # Not in Git
  project_update_1/
    run_unicode_deepseek.py
    annotate_unicode.py
    annotate_deepseek.py
    pipeline.py
    requirements.lock.txt
  deliverables/
    shaoyu/                       # Text-free delivery in Git
      ms_en/annotation/lid_offsets_shaoyu.jsonl   # Malay labels as offsets, reused by default
```

**Licence.** SwitchLingua is CC BY-NC 4.0 with an additional licensing agreement. The agreement forbids redistribution of the dataset, in whole or in part. So Git contains no dataset text: no CSV files, no texts, no LID segments, and no tokens. The committed `deliverables/shaoyu/` keeps only statistics, manifests, IDs, `cs_type`, review records, and the Malay labels as character offsets. Each user downloads the CSV files and rebuilds the full delivery locally.

The four CSV files are not in the repository. Download them from the Hugging Face dataset [`Shelton1013/SwitchLingua_text`](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text) and put them in `datasets/`. The runner checks the SHA-256 of each file. A different file version stops the run, because record IDs contain the file checksum.

Use Python 3.11 or newer. You need internet access for the first package and tokenizer download. Cantonese, Korean, and Russian need no API key, GPU, PyTorch, or model weights.

### 1. Create the environment and prepare the frozen tokenizer

```bash
python3 project_update_1/run_unicode_deepseek.py --setup-only
```

This command creates `project_update_1/.venv/` if it does not exist and installs `requirements.lock.txt`. It also downloads the frozen mBERT tokenizer to `project_update_1/assets/mbert/` if it is missing. These are the same environment and tokenizer as in the section above. On Windows, the runner changes CRLF line endings in the new tokenizer files to LF, so the checksum is the frozen value `78b44c32…`. Success prints `SETUP COMPLETE: pinned dependencies and frozen tokenizer ready.`

### 2. Rebuild the full delivery from the CSV files

```bash
python3 project_update_1/run_unicode_deepseek.py --skip-install
```

The full delivery, with text, goes to `deliverables/shaoyu_full/` (`--out`). Git ignores this folder. **Do not commit it.** After the build, the runner compares the text-free files with the committed `deliverables/shaoyu/` and prints `CHECK PASS`. A difference stops the run with an error that names the files.

The entry point does these steps for each pair:

1. Extract all text items of the full CSV with `pipeline.py extract`.
2. Sample 500 original records with `random.Random(f"6140:{pair}").sample(range(1, total + 1), 500)`. Keep all items of each sampled record. Each item is one sequence.
3. Label languages. `yue_en`, `ko_en`, `ru_en`: each character gets the label of its Unicode script. Latin letters are `en`, digits and punctuation are `NEUTRAL`. `ms_en`: rebuild the delivered DeepSeek labels from `lid_offsets_shaoyu.jsonl` and the CSV text (default), or label again with `--ms-lid deepseek`.
4. Select 20 pilot records with `random.Random(f"6140:{pair}:pilot")`. Mark their items `reviewed` and apply the one review fix (a romanized Cantonese `d`). All other items stay `unreviewed`.
5. Align labels to mBERT subwords and generate switch and duration targets with `pipeline.py tokenize` (`inherit_previous`, `observed`).
6. Write `cs_type` from the original CSV, publish the full delivery, and compare it with the committed files.

Intermediate files go to `project_update_1/data/shaoyu_reproduction/` (`--work-dir`). Git ignores this folder. A later run reuses a finished full extraction there. Nothing is staged, committed, deleted, or overwritten.

**Malay with DeepSeek.** LLM output can change between runs, so the default reuses the delivered Malay labels. To label Malay again, put a line `DEEPSEEK_API_KEY=...` in a file outside the repository, then run the command below. New labels can differ from the delivery, so this command skips the comparison.

```bash
python3 project_update_1/run_unicode_deepseek.py --skip-install --out deliverables/shaoyu_deepseek --check-against "" --ms-lid deepseek --key-file <key file>
```

The defaults are the delivered settings: `deepseek-flash`, thinking mode on, reasoning effort `high`, temperature 0, prompt v1.0. The full Malay run cost about USD 3.4.

### 3. Optional small end-to-end check

```bash
python3 project_update_1/run_unicode_deepseek.py --skip-install --pairs ko_en --records 5 --pilot-records 2 --work-dir project_update_1/data/shaoyu_smoke --out project_update_1/data/shaoyu_smoke_delivery --check-against ""
```

This check uses only `Korean_eng.csv` and needs no API key. It is a workflow check, not the formal delivery.

### Output layout

`[full only]` marks files that contain dataset text. They exist only in the local `deliverables/shaoyu_full/`, never in Git.

```text
deliverables/shaoyu/              # In Git (text-free)
deliverables/shaoyu_full/         # Local build with text (ignored by Git)
  README.md, DECISIONS.md         # deliverables/shaoyu only (Chinese)
  <pair>/                         # yue_en, ko_en, ru_en, ms_en
    cs_type_shaoyu.jsonl          # sequence_id -> cs_type; never a model input
    extraction/
      texts_shaoyu.jsonl             [full only]
      lid.template_shaoyu.jsonl      [full only]
      sequence_index_shaoyu.jsonl    # texts without the text field
      extract_issues_shaoyu.jsonl
      extract_manifest_shaoyu.json   # Includes the sampling rule
      sampled_record_ids_shaoyu.jsonl
    annotation/
      lid_shaoyu.jsonl               [full only]
      lid_offsets_shaoyu.jsonl       # ms_en only: [start, end, lang] spans
      lid_manifest_shaoyu.json
      build_manifest_shaoyu.json
      review_shaoyu.csv
      pilot_record_ids_shaoyu.txt    # Pass to split --train-records
      failures_shaoyu.jsonl          # ms_en only
    tokenized/
      labeled_tokens_shaoyu.jsonl    [full only] Main file for the baseline
      token_issues_shaoyu.jsonl
      stats_shaoyu.json
      manifest_shaoyu.json
```

The manifests in `deliverables/shaoyu/` come from the original 2026-10-15/16 run. A new run writes its own dates and checksums, so the comparison skips manifests (except `extract_manifest`).

The delivery contains 2,560 Cantonese–English, 1,529 Korean–English, 1,564 Russian–English, and 1,568 Malay–English sequences. One sampled Cantonese record cannot be parsed, so `yue_en` has 499 records with items.

### Validation scope and known limitations

- All four pairs pass `pipeline.py validate`. This checks structure and target rules, **not linguistic accuracy**.
- The review of the 20 pilot records per pair is model-assisted (Claude). It is not an independent human review. All other items are pseudo-labels.
- The script rule labels Latin-script names `en`, for example `NASA` in Korean text. The DeepSeek labels can mark names `UNKNOWN`.
- mBERT has no vocabulary entry for 55 common Cantonese characters. They become `[UNK]`, so nearby positions have no valid label.
- Each item is a separate sequence, so a switch between two items of one record gets no label.
- A rebuild gives the same labels as the original delivery. Only `annotation_source` differs for `yue_en`, `ko_en`, and `ru_en`, because it contains the script name and the run date.

More detail is in [`deliverables/shaoyu/README.md`](../deliverables/shaoyu/README.md) and [`deliverables/shaoyu/DECISIONS.md`](../deliverables/shaoyu/DECISIONS.md).

<a id="contributor-3"></a>
## Additional Language Pairs — Contributor 3

Reserved for the third data-processing contributor. Replace this heading with the assigned language pairs and add:

- Language pairs and source CSV paths.
- Annotation method, entry script, and required dependencies.
- Setup, processing, and code-check commands, including the working directory.
- Output location, validation scope, and known limitations.

No method or runnable command is specified here yet.

<a id="shared-standard"></a>
## Shared Data-Processing Standard

### Shared workflow overview

These sections document common tools and the annotation interface. LID recommendations are candidate methods, not claims of validated accuracy across all team language pairs. Use the relevant language-pair section above for each contributor's actual entry point, method, frozen assets, and output paths.

0. The coordinator places shared files on `main`. Contributors synchronize `main`, create personal branches, and prepare annotations and outputs there. Official result filenames include a contributor identifier.
1. Run extraction to obtain `texts.jsonl` and `lid.template.jsonl`.
2. Fill language spans manually or with the chosen annotation tool and save `lid.jsonl`. `pipeline.py` itself does not infer language; use the appropriate contributor's annotation tool.
3. Run tokenization and target generation to obtain `labeled_tokens.jsonl`.
4. Inspect examples and statistics and provide the language-pair directories for downstream use.
5. Use one coordinated train/validation/test split and supply samples through `loader.py`.

Originally, LLM preannotation with human review was suggested as a pilot approach, with local token-level LID as an alternative and dictionaries/Unicode rules as auxiliary evidence. This recommendation did not establish accuracy for every pair. The original generic examples used mBERT while tokenizer selection was pending. Each contributor's actual frozen choice belongs in their language-pair instructions. Each extracted entry remains an independent sequence, and punctuation/end policies must be explicit.

Validate shared rules first and use consistent commands and assets across contributors.

### Initial installation of the shared folder in a team clone

Copy the complete `project_update_1/` folder into the root of the cloned team repository, preserving its structure and hidden `.gitignore`. Do not scatter individual files across directories. If using a ZIP distribution, unpack it first. Compare any existing same-named directory before replacing anything.

```text
repository/
  other_existing_files...
  project_update_1/
    README.md
    LID_PROMPT.md
    pipeline.py
    loader.py
    requirements.txt
    requirements.lock.txt
    VERIFICATION.md
    .gitignore
    tests/
    examples/
```

The shared code package contains code, documentation, dependencies, and teaching examples. It does not automatically include raw CSV files, full personal outputs, environments, or model caches. `examples/worked_example/` is a fictional worked example that can accompany shared code; it is not a contributor's experimental result. Original CSV files are managed separately under the repository-root `datasets/`. They are not in Git; each member downloads them from Hugging Face.

Suggested initial coordination sequence:

1. Inspect `git status`, preserve existing work, and switch to an updated `main`.
2. Create a preparation branch such as `setup/data-pipeline`, or reuse it if it already exists.
3. Copy the shared folder into the repository root without overwriting unrelated work.
4. Install dependencies and run checks using the environment instructions below.
5. From the repository root, stage only the intended shared files, inspect them, commit, and push the preparation branch.
6. Open and review a pull request to `main`. After merging, contributors synchronize before creating their data branches.

The following commands illustrate step 5; preparing local files does not execute them automatically:

```bash
git add project_update_1/
git diff --cached --stat
git diff --cached
git commit -m "Add shared Update 1 data pipeline and guidelines"
git push -u origin setup/data-pipeline
```

Check that staging contains no unrelated files. Shared code keeps its common name; personal result files use contributor suffixes. A subdirectory `.gitignore` only covers its own scope and does not replace a repository-wide review for credentials or caches.

### Personal branches and contributor naming

#### Shared files on main

Complete shared-code preparation before the data contributors start their personal work. Preserve relative paths and submit the unpacked code rather than only a ZIP.

| Shared file | Purpose |
|---|---|
| `README.md` | Processing standard, run instructions, and delivery rules |
| `LID_PROMPT.md` | Shared assisted-annotation prompt |
| `pipeline.py` | Extraction, subword alignment, targets, validation, and splitting |
| `loader.py` | Batches for current-token and prefix experiments |
| `requirements.txt`, `requirements.lock.txt` | Dependency ranges and tested pinned versions |
| `tests/` | Shared rule checks |
| `examples/` | Teaching examples, outputs, and review templates |
| `VERIFICATION.md` | Completed checks and unverified scope |
| `.gitignore` | Exclusions for working data, assets, environments, and caches |

Use a reviewed pull request for shared changes. Placing code on `main` does not require bypassing branch protection. Branch names here are examples; do not create duplicates of existing branches.

Personal deliverables belong under repository-root `deliverables/<name>/` and may also merge into `main` through review. Keep full intermediates, environments, caches, and API credentials outside the submitted code package. Raw inputs have their own explicitly managed location; Record each workflow's input location in its language-pair section.

`assets/` is ignored by the local `.gitignore`; a downloaded tokenizer is not automatically uploaded with code. Record the selected model, fixed revision, file checksum, and policies, and define how matching assets are obtained. A workflow may download and verify frozen assets during setup instead of tracking them in Git. The earlier teaching manifest alone was not a team-wide selection decision.

#### Create a personal branch

Create a personal branch from the shared-code version of `main` before editing annotations or outputs. The branch already contains common files such as `pipeline.py`; do not create renamed personal copies of the shared implementation. Suggested branch names use `data/<name>`, for example `data/alice`. Use a unique lowercase contributor identifier without spaces.

From the repository root, after preserving existing work:

```bash
git status
git switch main
git pull --ff-only
git switch -c data/alice
git push -u origin data/alice
```

Substitute your identifier and actual default branch. If the personal branch exists, use `git switch data/alice` instead of creating it again. Do not discard local work to force a branch switch. Confirm your branch with `git branch --show-current`, then use a pull request for review and integration.

#### Synchronize shared-code updates

Propose changes to shared scripts or rules through a pull request explaining their effects on outputs. Do not maintain incompatible copies or rename shared code to `pipeline_alice.py`.

After saving local work, synchronize from the personal branch:

```bash
git switch data/alice
git fetch origin
git merge origin/main
```

Then run `python -m unittest discover -s tests -v` from `project_update_1/`. Inspect both sides when resolving conflicts. If alignment, tokenization, or target rules change, regenerate affected outputs in a new directory rather than mixing rule versions.

Git branches organize code collaboration. They do not create train/validation/test dataset splits.

#### Add contributor names to official result filenames

Use `<original_stem>_<name>.<extension>`. For example:

| Working filename | Official filename |
|---|---|
| texts.jsonl | texts_alice.jsonl |
| lid.template.jsonl | lid.template_alice.jsonl |
| extract_issues.jsonl | extract_issues_alice.jsonl |
| extract_manifest.json | extract_manifest_alice.json |
| lid.jsonl | lid_alice.jsonl |
| review.csv | review_alice.csv |
| labeled_tokens.jsonl | labeled_tokens_alice.jsonl |
| token_issues.jsonl | token_issues_alice.jsonl |
| stats.json | stats_alice.json |
| manifest.json | manifest_alice.json |
| pilot_record_ids.txt, if applicable | pilot_record_ids_alice.txt |

Keep pairs in separate directories, for example repository-root `deliverables/alice/fr_en/tokenized/labeled_tokens_alice.jsonl`. Generic shared commands generate fixed working filenames; complete validation before copying results into the formal `extraction/`, `annotation/`, and `tokenized/` layout with suffixes. A contributor's entry point may automate packaging; document that behavior in the corresponding language-pair section.

Rename files only: do not change `sequence_id`, `record_id`, `text_sha256`, or text. Point `--texts`, `--lid`, `--input`, and `--inputs` to the actual suffixed filenames when reading deliverables. From `project_update_1/`:

```bash
python pipeline.py validate --input ../deliverables/alice/fr_en/tokenized/labeled_tokens_alice.jsonl
```

Raw CSV filenames, shared scripts, dependency files, and tokenizer assets do not receive personal suffixes. Combined outputs use the producing contributor's suffix when formally delivered, such as `splits_bob.jsonl`.

`data/` is ignored, so adding working files there does not automatically stage them. Prepare intended outputs under `deliverables/` and inspect the staging area. Use the team's agreed LFS or storage approach for large files. Do not submit environments, caches, or credentials.

### Tools and environment for shared commands

**Unless stated otherwise, commands in the remaining shared-tool sections run inside `project_update_1/` with the prepared environment active.** This differs from the repository-root entry-point commands above.

| Tool | Purpose | Trains a model? |
|---|---|---|
| Python 3.13 | CSV/JSON processing, rule calculations, and sample iteration | No |
| Built-in `csv`, `json`, `ast` | Reading CSV and literal list containers | No |
| Transformers / Tokenizers | Existing tokenizer, subword IDs, and character offsets | No |
| Chosen LID method | Assigning language to words or spans | Separate from `pipeline.py` |
| `unittest` | Boundary, grouping, and future-input isolation checks | No |

The tokenizer workflow does not require PyTorch, a GPU, Excel, or Transformer weights. `loader.py` is a pure-Python iterator rather than a PyTorch DataLoader; downstream code can adapt it.

`requirements.txt` specifies bootstrap ranges, while `requirements.lock.txt` records tested pinned dependencies. Use a consistent Python minor version and pinned dependencies across the team. Resolve cross-platform installation failures explicitly rather than silently changing versions independently. Any additional method-specific dependencies must be documented and installed by the corresponding workflow.

From this directory on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m unittest discover -s tests -v
```

On Windows, activation is `.venv\Scripts\Activate.ps1`. Quote paths containing spaces.

### Working layout and common formats

```text
project_update_1/
  README.md
  LID_PROMPT.md
  pipeline.py
  loader.py
  requirements.txt
  requirements.lock.txt
  tests/
  examples/
  assets/
    chosen_tokenizer/
      tokenizer_manifest.json
      tokenizer/                 # Frozen tokenizer files
  data/
    fr_en/
      extraction/
        texts.jsonl
        lid.template.jsonl
        extract_issues.jsonl
        extract_manifest.json
      annotation/
        lid.jsonl
        review.csv
      tokenized/
        labeled_tokens.jsonl
        token_issues.jsonl
        stats.json
        manifest.json
    ...
  combined/
    labeled_tokens.jsonl
    splits.jsonl
    split_manifest.json
```

JSON/JSONL uses UTF-8 without BOM and standard JSON. Each JSONL line is a complete object without Markdown fences. Embedded newlines are escaped as `\n` and restored when parsed. Review CSV files use UTF-8 with headers. The layout above uses working filenames; formal outputs receive contributor suffixes.

Agree on language codes before processing. Do not infer `fil`, `tl`, or another language solely from the filename `Philippines`. Document the chosen detector-to-source mapping in the corresponding language-pair section.

Use `sequence_id` to join outputs, not row positions. `source_row` is the one-based data-record number after the CSV header, not necessarily a physical text-file line number. All extracted items from one source record share a `record_id`.

Output directories must be empty for shared generation commands. Use a new version directory when changing rules.

### Step 0: Extract text

Try the first three source records:

```bash
python pipeline.py extract --input "../datasets/French_eng.csv" --languages fr,en --limit 3 --out data/fr_en/extraction
```

For the full source, omit `--limit 3` and choose an unused output directory. The extractor reads only `data_generation_result`, supports string lists and nested `instances`, and excludes summaries or evaluation prose.

Original text is preserved without Unicode normalization, translation, quote/backslash removal, or punctuation edits. Possible wrapping quotes and escape remnants produce warnings. Unparsable records produce errors while processing continues. Explain the disposition of errors before delivery; completion alone is not sufficient. Warnings are review candidates, not evidence that quoted speech is invalid.

Parsing uses JSON or `ast.literal_eval`, never `eval`. Cell-size limits and exception handling reduce accidental failures, but this is intended for known course data rather than an arbitrary hostile-file sandbox.

| Field in texts.jsonl | Type | Meaning |
|---|---|---|
| schema_version | string | Currently 1.0 |
| sequence_id | string | Unique pair/file-hash/source-record/item ID |
| record_id | string | Basic source-record grouping unit |
| item_id | integer | One-based extracted item number within the record |
| language_pair / languages | string / list of strings | Pair and permitted language codes |
| source_file / source_row / source_path | string / integer / list | File and original nested location |
| text | string | Original mixed-language text |
| text_sha256 | string | Checksum for detecting text changes |

One CSV record may contain multiple texts. Current processing does not concatenate them, but splitting must keep their record group together. Adding discourse context requires explicit sequence construction and boundary handling rather than naive concatenation.

### Step 1: Prepare LID annotations

#### Candidate methods and their limits

The original shared recommendation was to pilot contextual LLM preannotation with human review. This is an implementation option, not a course requirement or proof of reliability. No agent is required: small batches can use a chat interface, while larger runs may use an API. The actual implemented method for each contributor is documented in their language-pair section.

| Role | Method | How it is used | Main limitations |
|---|---|---|---|
| Pilot candidate | LLM preannotation plus human review | Shared prompt, original text and pair, consecutive span output, reconstruction and language checks | Possible errors or rewriting; API cost; unequal reliability across languages |
| Alternative | Local pretrained LID | Load a supported model and convert its output to the shared format | Confirm mixed-language support; sentence-level LID is not token-level annotation |
| Auxiliary | Dictionaries | Retain UNKNOWN when word membership is ambiguous | Shared words, borrowing, inflection, and proper names |
| Auxiliary | Unicode script rules | Script clues and checks for obvious conflicts | Script is not language; Latin script cannot reliably distinguish French and English |

Manual annotation can create a small reference set without requiring contributors to label the full corpus by hand. Review requires language competence. Having the same LLM confirm its own output is not independent human verification.

#### Pilot procedure for an assisted method

1. Select about 20 examples per pair covering single/multiple switches, names, and ambiguous words. This helps detect workflow problems; it is not a statistical quality guarantee.
2. Use a consistent selected model/version and `LID_PROMPT.md` to produce `segments`. Record the actual model, prompt version, date, and available generation parameters.
3. Fill the matching `lid.jsonl` entries, retaining template IDs and checksums. Start as `unreviewed`. Check structure/reconstruction automatically and language decisions independently.
4. Record error types and corrections. If results are poor, revise the method and review again before scaling. Keep each method's provenance distinct.
5. Estimate bulk runtime, output size, and cost. If using an API, provide your own account/credentials without placing secrets in code or shared files.
6. Bulk outputs remain automatic pseudolabels until reviewed. Report the actual review scope and remaining uncertainty per language pair.

`pipeline.py` supplies a LID interface, structural checks, and downstream alignment; it does not call an LLM API. Local annotation adapters, when provided, are separate from this shared interface. The list of candidate methods does not imply that all alternatives were implemented or validated.

#### LID format

Copy `lid.template.jsonl` to `annotation/lid.jsonl`, preserving IDs and checksums. Annotate consecutive text spans; character positions are computed automatically.

The example below illustrates structure only. Replace placeholder IDs/checksums with real template values:

```json
{
  "sequence_id": "copy_from_template",
  "text_sha256": "copy_from_template",
  "annotation_source": "manual or actual tool/version/prompt version",
  "review_status": "unreviewed",
  "reviewer": "",
  "segments": [
    {"text": "Je pense que ", "lang": "fr"},
    {"text": "the referee is unfair", "lang": "en"},
    {"text": ".", "lang": "NEUTRAL"}
  ]
}
```

In JSONL, each object occupies one line. Concatenating `segments[].text` must reconstruct the original exactly, including whitespace, newlines, quotation marks, and accents. Validation checks this and calculates zero-based half-open `[start,end)` character spans.

Annotation conventions:

- Judge words/spans using context; do not guess the language of isolated subword fragments.
- Use `NEUTRAL` for standalone punctuation, numbers, emoji, and similar content. Keep internal apostrophes with words where appropriate.
- Use context for names and borrowing; retain `UNKNOWN` if uncertain. A person's country of origin does not determine the language label of their name in context.
- Preserve whitespace, optionally within adjacent spans; alignment ignores whitespace-only language evidence.
- Allowed review states are `unreviewed`, `reviewed`, and `needs_review`. Set `reviewed` and fill `reviewer` only after actual review.
- `LID_PROMPT.md` is a reusable request, not an annotation method proven accurate for all 12 pairs.
- Unicode rules are auxiliary evidence, not a replacement for French--English LID.

Choose a concrete service, budget, and language coverage before integrating an external API. Record the real `annotation_source`; shared code does not upload text or invoke paid APIs automatically.

A suggested initial review covers roughly 20 examples with short bursts, multiple switches, and uncertain words. This is neither a course-mandated count nor certification. Seek a competent reviewer for unfamiliar languages. Records used to develop annotation rules must stay on the training side or be excluded from evaluation. Write their `record_id` values one per line in `pilot_record_ids.txt`; the splitter's `--train-records` option assigns their complete connected duplicate groups to training.

Record reviews in `review.csv` using the example header. Metrics computed against automatic labels must be described as pseudolabel-based, not accuracy against human ground truth.

### Step 2: Prepare the shared tokenizer

The coordinator prepares one asset for consistent use. This is the original generic mBERT example; actual workflows must use their documented frozen revisions and asset directories.

```bash
python pipeline.py prepare-tokenizer --model google-bert/bert-base-multilingual-cased --out assets/chosen_tokenizer
```

This command needs the network but downloads only tokenizer files. It resolves `main` to a concrete revision and records a checksum. Do not independently download a moving `main` revision at different times and assume the assets match; share or reproduce the frozen version.

To explore XLM-R, use `FacebookAI/xlm-roberta-base` in a separate asset/output directory and validate it first. Shared verification covers only tested tokenization behavior; inspect offset mappings, overlaps, and cross-language tokens for alternatives.

Tokenization uses `use_fast=True`, `add_special_tokens=False`, `truncation=False`, and `padding=False`. Do not truncate the source before calculating durations. Add required special tokens and padding later when constructing model inputs, extending ignore masks accordingly.

Different tokenizers can produce different subword counts and duration classes. Generate separate labels/statistics and explain differences instead of mixing them into one baseline. mBERT token IDs cannot be supplied directly to an XLM-R model.

### Step 3: Align subwords and generate targets

The following example selects two explicit policies: neutral tokens inherit the preceding resolved language and count toward length, and the observed text ending terminates the final segment.

```bash
python pipeline.py tokenize --texts data/fr_en/extraction/texts.jsonl --lid data/fr_en/annotation/lid.jsonl --tokenizer-dir assets/chosen_tokenizer --neutral-policy inherit_previous --end-policy observed --out data/fr_en/tokenized
```

By default, LID must be reviewed. To deliberately generate research pseudolabels from unreviewed annotations, pass `--allow-unreviewed`. The choice and review status are recorded. This permits processing; it is not a quality certificate.

#### Shared policy choices

| Parameter | Behavior | Consequence |
|---|---|---|
| neutral-policy=inherit_previous | Standalone neutral tokens inherit only the preceding resolved token language; initial neutral tokens or those after UNKNOWN remain uncertain | Punctuation can precede a switch and contributes to duration |
| neutral-policy=mask | Standalone neutral tokens become UNKNOWN | Adjacent switch positions and some durations become unavailable |
| end-policy=observed | The extracted entry ending terminates the segment | Labels describe observed item-local length, not necessarily complete discourse length |
| end-policy=censored | Ignore duration if the next language boundary is not observed before the ending | A switch target may remain while its duration is ignored |

These are project conventions, not verified instructor-prescribed rules. The generic standard initially suggested `inherit_previous + observed` for a demonstration; Actual frozen choices are documented per workflow. If using `mask`, inspect whether it systematically removes intersentential boundaries rather than ignoring the resulting bias.

Special cases:

- A subword spanning characters from multiple labeled languages becomes UNKNOWN; do not assign by an arbitrary majority vote.
- `[UNK]` becomes UNKNOWN rather than a confidently labeled subword.
- Whitespace supplies no language evidence. A token containing one real language plus neutral characters can inherit that language.
- A known switch followed by a segment ending at UNKNOWN keeps `switch=1` but receives `duration=-1`.
- Non-switch positions have `duration=-1`. The final position has `switch=-1` and a false switch mask because no next token exists.
- Long original texts are not truncated here. Later inputs may crop historical context without shortening or recomputing the target segment.

#### Labeled-token fields

`labeled_tokens.jsonl` retains extraction fields and adds:

| Field | Type | Meaning |
|---|---|---|
| input_ids | integer list | Subword vocabulary IDs |
| tokens | string list | Readable tokens for inspection, not text reconstruction |
| offsets | pairs of integers | Zero-based `[start,end)` source character positions |
| language_spans | object list | Character spans calculated from LID |
| language_ids | string list | Subword language, or UNKNOWN when uncertain |
| switch_labels | integer list | 0/1 targets, or -1 when invalid |
| switch_valid_mask | boolean list | Positions eligible for switch training/scoring |
| duration_lengths | integer/null list | Upcoming segment length in subwords |
| duration_labels | integer list | 0=1--2, 1=3--6, 2=7+, or -1 when ignored |
| duration_valid_mask | boolean list | Positions eligible for duration training/scoring |
| alignment_notes / label_notes | string lists | Alignment decisions and invalid-target reasons |
| tokenizer_sha256 | string | Frozen asset identity |
| neutral_policy / end_policy | strings | Actual processing policies |
| annotation_source / review_status / reviewer | strings | Annotation provenance and review status |

All per-token arrays must have equal length. A target at position `t` predicts `t+1`. Each task has its own validity mask; whether a model predicts the switch correctly must not determine whether a ground-truth duration target exists.

`manifest.json` records input/script checksums, actual library versions, and policies. Duration proportions in `stats.json` use duration-valid switches as the denominator; switch rate uses switch-valid positions. An undefined proportion is `null`, not zero.

### Step 4: Validate and deliver

```bash
python pipeline.py validate --input data/fr_en/tokenized/labeled_tokens.jsonl
```

A PASS confirms structure and rule consistency, not language accuracy. Additional review should address:

1. Trace extracted items to source records and confirm no text rewriting.
2. Inspect language assignments and subword alignment on examples.
3. Manually check lengths 1, 2, 3, 6, and 7, plus no-switch, multiple-switch, punctuation, ending, and UNKNOWN cases.
4. Explain extraction errors and inspect warnings.
5. Report UNKNOWN counts and causes from `token_issues.jsonl`.
6. Record a real case with the original text, issue, handling decision, verification, and remaining limitations. A unique software bug is not required; annotation review or statistical differences can be informative.

Each contributor prepares their assigned language-pair delivery on their personal branch with suffixed result filenames, shared-code provenance, actual dependency versions, and branch information. Exclude environments, credentials, and caches. Keep original CSV inputs unchanged in their designated location.

### Step 5: Coordinated grouped splitting

After collection, use one coordinated splitting run. For example:

```bash
python pipeline.py split --inputs data/fr_en/tokenized/labeled_tokens.jsonl data/ar_en/tokenized/labeled_tokens.jsonl --seed 6140 --out combined
```

Supply the actual paths for all selected pairs. The program rejects mixed tokenizer identities or punctuation/end policies.

- Group by `record_id`, then connect records sharing normalized exact text. Normalization serves duplicate detection only and does not alter original text or offsets.
- The suggested grouped split is approximately 80/10/10. It is not a course-mandated ratio, and sequence/position proportions need not match it exactly.
- Fewer than three independent groups sends a group category entirely to training; the manifest records this. Do not claim meaningful test performance from such a case.
- Exact duplicate detection does not resolve near duplicates, common generation templates, or shared articles; inspect those separately.
- Freeze and reuse `splits.jsonl`. Do not add data, randomly resplit, and compare results as though evaluation sets were unchanged.
- Use `--train-records pilot_record_ids.txt` for development records. Missing listed IDs raise an error instead of being silently ignored.
- Descriptive statistics may cover all data. Choose thresholds, hyperparameters, and rules using training/validation data rather than test results.

`splits.jsonl` records `sequence_id`, `record_id`, `group_id`, and `split`. `split_manifest.json` records the seed, input hashes, and actual per-pair counts.

### Step 6: Reading samples for a baseline

Inspect one batch of current-token samples:

```bash
python loader.py --data combined/labeled_tokens.jsonl --splits combined/splits.jsonl --split train --context current --batch-size 4
```

Each sample separates these fields:

```json
{
  "inputs": {"input_ids": [1234]},
  "targets": {"switch": 1, "duration": 0},
  "provenance": {"sequence_id": "example_id", "position": 5, "language_pair": "fr_en"}
}
```

The numbers illustrate format only. Supply only `inputs` to the predictor. `targets` contains answers and `provenance` supports tracing and pair-level reporting. LID, full-text language ratios, complete future text, and future tokens are absent from `inputs`.

For a full-prefix experiment, use `--context prefix`; inputs stop at `t` and exclude `t+1`. `--max-context 128` crops historical tokens on the left while preserving full-sequence targets. Outputs are variable-length Python lists without model special tokens, padding, or tensors. Transformer integration needs a corresponding collator and masks.

Here, streaming means consuming a pretokenized subword sequence position by position. It does not implement live character input or incremental tokenization; word-internal tokenization can depend on the full word's characters.

Downstream baseline code can estimate next-switch frequencies from training tokens, select thresholds on validation, and calculate F1 on a fixed evaluation set. The shared data package does not itself train or score a baseline or provide final model comparisons.

### Fictional example without real data

`examples/make_demo.py` generates a fictional French--English text and unreviewed teaching LID. It is not course experimental data.

```bash
python examples/make_demo.py data/demo
python pipeline.py tokenize --texts data/demo/texts.jsonl --lid data/demo/lid.jsonl --tokenizer-dir assets/chosen_tokenizer --neutral-policy inherit_previous --end-policy observed --allow-unreviewed --out data/demo_tokenized
python pipeline.py validate --input data/demo_tokenized/labeled_tokens.jsonl
```

Inspect actual tokens, offsets, and labels before processing real CSV files. A single fictional record cannot yield a meaningful generalization F1.

### Common errors

| Message | What to check |
|---|---|
| Output directory must be empty | Use an unused version directory rather than replacing results |
| Segments must reproduce original text exactly | Check spaces, newlines, quotation marks, or rewriting by the annotator |
| Text hash mismatch | Check changed text or annotation pasted from another example |
| LID IDs must exactly match texts IDs | Check missing, duplicate, or partial annotations |
| not reviewed | Complete review, or explicitly accept pseudolabels with `--allow-unreviewed` |
| Frozen tokenizer files changed | Use identical assets and do not manually edit the vocabulary |
| Mixed tokenizers or label policies | Use consistent configuration before merging |

### Reference documentation

- [Hugging Face tokenizer API and offset mapping](https://huggingface.co/docs/transformers/main_classes/tokenizer)
- [mBERT model/tokenizer documentation](https://huggingface.co/google-bert/bert-base-multilingual-cased)
- [Python literal_eval](https://docs.python.org/3/library/ast.html#ast.literal_eval)

These references document tools. The annotation, punctuation, and ending conventions are explicit project choices, not conclusions validated by those references or the course.
