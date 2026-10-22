# cs6140-streaming-code-switching

Group 8's CS6140 project on streaming code-switching prediction.

For Project Update 1, choose the section matching the language pairs you want to process:

- [French–English, German–English, Italian–English, and Filipino/Tagalog–English](#fr-de-it-ph-en)
- [Cantonese–English, Korean–English, Russian–English, and Malay–English](#yue-ko-ru-ms-en)
- [Additional language pairs — Contributor 3](#contributor-3)
- [Shared tools and repository layout](#shared-tools)

<a id="fr-de-it-ph-en"></a>
## French–English, German–English, Italian–English, and Filipino/Tagalog–English

**Method:** Lingua language-span annotation and frozen mBERT subword alignment.
**Entry point:** [`project_update_1/run_lingua.py`](project_update_1/run_lingua.py).
This workflow is configured for these four pairs only; it does not train a baseline or implement the other contributors' annotation methods.

### Run from the repository root

Use Python 3.13 if available; Python 3.11 or newer is required.

```bash
# Create/reuse the environment, install dependencies, and prepare the tokenizer.
python3 project_update_1/run_lingua.py --setup-only

# Process these four pairs and generate deliverables/yufan/.
python3 project_update_1/run_lingua.py --skip-install
```

The runner reads `French_eng.csv`, `German_Eng.csv`, `Italian_eng.csv`, and
`Philippines_eng.csv` from [`datasets/`](datasets/). It launches its virtual
environment automatically and downloads only the frozen tokenizer. No manual
activation or LLM API key is required. Existing outputs are never overwritten;
if `deliverables/yufan/` exists, add `--out deliverables/lingua_reproduction`
to the processing command.

### Scripts for these pairs

| Script | Role |
|---|---|
| [`run_lingua.py`](project_update_1/run_lingua.py) | Complete workflow entry point |
| [`annotate_lingua.py`](project_update_1/annotate_lingua.py) | Language-span annotation helper called by the entry point |

### Code check without generating data

```bash
python3 project_update_1/run_lingua.py --help
cd project_update_1
python3 -m unittest discover -s tests -v
cd ..
```

These checks do not require an existing delivery. Full setup details, a small
end-to-end run, output definitions, and optional verification of generated data
are in the [matching language-pair section](project_update_1/README.md#fr-de-it-ph-en).
All LID for this workflow remains automatically generated and unreviewed.
Structural validation establishes consistency, not linguistic accuracy.

<a id="yue-ko-ru-ms-en"></a>
## Cantonese–English, Korean–English, Russian–English, and Malay–English

**Method:** Unicode script-rule LID for Cantonese, Korean, and Russian; DeepSeek
LLM LID (shared prompt v1.0) for Malay; frozen mBERT subword alignment.
**Entry point:** [`project_update_1/run_unicode_deepseek.py`](project_update_1/run_unicode_deepseek.py).
**Output:** [`deliverables/shaoyu/`](deliverables/shaoyu/) in Git has no dataset text (licence: no
redistribution). The runner rebuilds the full data locally in `deliverables/shaoyu_full/`.

### Run from the repository root

The CSV files are not in the repository. Download `Can_Eng.csv`,
`Korean_eng.csv`, `Russian_eng.csv`, and `Malay_eng.csv` from the Hugging Face
dataset [`Shelton1013/SwitchLingua_text`](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text)
and put them in [`datasets/`](datasets/). Python 3.11 or newer is required.

```bash
# Create/reuse the environment, install dependencies, and prepare the tokenizer.
python3 project_update_1/run_unicode_deepseek.py --setup-only

# Rebuild the full delivery in deliverables/shaoyu_full/ (ignored by Git) and check it.
python3 project_update_1/run_unicode_deepseek.py --skip-install
```

No API key is needed. By default, Malay reuses the delivered DeepSeek labels
(stored as character offsets), because LLM output can change between runs. Setup details, the optional
DeepSeek run, outputs, and limitations are in the
[matching language-pair section](project_update_1/README.md#yue-ko-ru-ms-en).

<a id="contributor-3"></a>
## Additional Language Pairs — Contributor 3

Reserved for the third data-processing contributor. Replace this heading with
the assigned language pairs, then add the method, entry script, setup/run
commands, and output location. Detailed instructions belong in the
[matching section](project_update_1/README.md#contributor-3).

<a id="shared-tools"></a>
## Shared Tools and Repository Layout

| Shared component | Purpose |
|---|---|
| [`pipeline.py`](project_update_1/pipeline.py) | Text extraction, subword alignment, target generation, validation, and grouped splitting |
| [`loader.py`](project_update_1/loader.py) | Current-token and prefix sample iteration from labeled data and split assignments |
| [`tests/`](project_update_1/tests/) | Shared processing-rule checks |

```text
datasets/               Original CSV inputs
project_update_1/       Method-specific entry points, shared tools, tests, and instructions
deliverables/           Generated outputs, when present; not required to inspect the code
```

See the [shared data-processing standard](project_update_1/README.md#shared-standard)
for common formats and labeling rules. Full text supports target construction;
prediction inputs must contain only the current token or observed prefix.
