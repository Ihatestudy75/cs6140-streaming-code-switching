# cs6140-streaming-code-switching

Group 8's CS6140 project on streaming code-switching prediction.

## Project Update 1: start here

**To run the Lingua data-processing workflow, use
[`project_update_1/run_lingua.py`](project_update_1/run_lingua.py).**
It currently processes French–English, German–English, Italian–English, and
Filipino/Tagalog–English. This entry point is for data preparation, not baseline
training or an implementation of every team member's annotation method.

Run these commands from the **repository root** using Python 3.13 if available
(Python 3.11 or newer is required):

```bash
# Create/reuse the environment, install dependencies, and prepare the tokenizer.
python3 project_update_1/run_lingua.py --setup-only

# Process the four pairs and generate deliverables/yufan/.
python3 project_update_1/run_lingua.py --skip-install
```

The runner reads the four original CSV files in [`datasets/`](datasets/),
downloads only the frozen mBERT tokenizer, and launches its virtual environment
automatically. No manual environment activation or LLM API key is required.
Existing outputs are never overwritten. If `deliverables/yufan/` already exists,
add `--out deliverables/lingua_reproduction` to the processing command.

### Which script should I use?

| Task | Script | How to use it |
|---|---|---|
| Run the configured four-pair Lingua workflow | [`run_lingua.py`](project_update_1/run_lingua.py) | Main entry point; use the commands above |
| Generate language-span annotations only | [`annotate_lingua.py`](project_update_1/annotate_lingua.py) | Called by the entry point; direct use requires text, issues, language, and output arguments |
| Extract text, align subwords, generate/validate targets, or split groups | [`pipeline.py`](project_update_1/pipeline.py) | Shared tools; see the detailed instructions |
| Read current-token or prefix samples | [`loader.py`](project_update_1/loader.py) | Requires labeled data and split assignments |

### Quick code check without generating datasets

```bash
python3 project_update_1/run_lingua.py --help
cd project_update_1
python3 -m unittest discover -s tests -v
cd ..
```

These checks do not require an existing `deliverables/yufan/` directory.
For a small end-to-end run after setup, full output descriptions, and optional
verification of generated data, see
[`project_update_1/README.md`](project_update_1/README.md).

### Repository layout

```text
datasets/               Original CSV inputs
project_update_1/       Data-processing entry point, shared tools, tests, and instructions
deliverables/          Generated outputs, when present; not required to inspect the code
```

Language labels are automatically generated and remain unreviewed. Structural
validation confirms consistency of the processing outputs, not linguistic
accuracy. Full text supports label construction; prediction inputs must contain
only the current token or observed prefix.
