# Raw SwitchLingua CSV files

This folder holds the original SwitchLingua CSV files. **The CSV files are not in Git.**
SwitchLingua is CC BY-NC 4.0 with an additional licensing agreement, and the agreement
forbids redistribution of the dataset, in whole or in part. `.gitignore` ignores
`datasets/*.csv`.

Each team member downloads the files from the Hugging Face dataset
[`Shelton1013/SwitchLingua_text`](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text)
(log in and accept the access conditions first) and puts them here without renaming them.

| Files | Used by |
|---|---|
| `French_eng.csv`, `German_Eng.csv`, `Italian_eng.csv`, `Philippines_eng.csv` | `project_update_1/run_lingua.py` |
| `Can_Eng.csv`, `Korean_eng.csv`, `Russian_eng.csv`, `Malay_eng.csv` | `project_update_1/run_unicode_deepseek.py` |
