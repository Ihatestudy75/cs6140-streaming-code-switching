# Completed verification

Historical verification date: 2026-10-08. Environment: Python 3.13.7; actual dependency versions are recorded in `requirements.lock.txt`. This file records the original shared-code checks, rather than the current completion status of every language pair. See `README.md` for Yufan's current entry point.

## Extraction from a real file

The extractor ran read-only on the original `French_eng.csv`: 8,070 source records and 23,563 text items, with no extraction errors and 44 warnings about possible quotation wrapping or escape remnants. Nested `instances` records were extracted successfully. Warnings were not automatically corrected as errors. Full LID was not part of this historical extraction check.

## Tokenizer check

The mBERT tokenizer performed actual tokenization and alignment on one fictional teaching example:

```text
Je / pense / que / the / referee / is / un / ##fair / .
```

The demonstration language labels are unreviewed teaching labels. With `inherit_previous + observed`, the period inherits English and counts toward length. The English segment following `que` therefore contains six subwords and receives duration class 1. Changing the punctuation policy can change the statistics, so the team must use a consistent policy.

Complete example inputs and actual outputs are in `examples/worked_example/`. They are fictional, are not SwitchLingua data, and must not support claims about model performance. The tokenizer revision is recorded in `tokenizer_manifest.json` and can be reproduced with `prepare-tokenizer --revision`. This check downloaded no model weights and trained no baseline.

## Rule checks

Eight unit tests passed in the original verification:

1. Duration classes for lengths 1, 2, 3, 6, 7, and 9, including adjacent-position boundaries.
2. `UNKNOWN` and segments censored at the end of a text.
3. Rejection of LID annotations that rewrite text or omit whitespace.
4. Unicode character offsets, neutral characters, tokens crossing languages, and unknown tokens.
5. Newlines inside CSV cells, nested `instances`, and unparsable records.
6. Grouping by record and duplicate text; both current-token and prefix inputs exclude the next token.
7. Detection of incorrect targets by the validator.
8. Assignment of pilot records and their connected exact-duplicate groups to training.

The historical checks cover only the scope listed above. They do not establish LID accuracy for the other 11 language pairs, validate XLM-R tokenization, or demonstrate real model training.

## Re-running the checks

From `project_update_1/`, after environment setup:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.
