# Git history rewrite (2026-11-17)

On 2026-11-17 the full Git history of this repository was rewritten with
`git-filter-repo` 2.47.0 to remove files that contained SwitchLingua sentences.
The dataset license does not allow us to redistribute its text.

Removed from every commit:

| Path | What it was |
|---|---|
| `update1_data_standard/deliverables/yufan/` | Update 1 Lingua delivery for fr/de/it/philippines (texts, LID, labeled tokens, case studies, quarantine reasons) |
| `deliverables/yufan/` | The same files on the former `data/yufan` branch (branch deleted) |
| `deliverables/cora/error_examples.json` | Update 1 baseline error examples with text fields |

None of these paths existed on `main` at the time, so no current file changed.
Every commit hash changed.

## Old to new commit hashes

[`history_rewrite_commit_map.txt`](history_rewrite_commit_map.txt) lists one `old new` pair per line.
A new hash of all zeros means the commit only touched removed files and was dropped.

Run records store the old hashes (`git_commit` in `config.json`, `source_commit` in
`evaluation_provenance.json`). Look one up by its prefix, for example:

```bash
grep '^7fe3727' project_update_2/history_rewrite_commit_map.txt
```

GitHub pull-request pages may keep showing old commits until GitHub Support purges them.
