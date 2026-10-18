"""Write the English report and rebuild the local review package."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/all-v3'


def main():
    results = json.loads((RUN / 'results.json').read_text())
    split = json.loads((RUN / 'combined/split_manifest.json').read_text())
    pct = lambda x: f'{100*x:.2f}%'
    lines = [
        '# Baseline and Next-Step Report', '',
        '18 October 2026', '',
        '## Data and checks', '',
        'We used 21,859 sequences from 14 language pairs: four from Shaoyu, six from Yulun, and four from Yufan.',
        'We checked file hashes, IDs, text, language spans, token alignment, labels, and masks. These checks passed.',
        'Original data and shared pipeline files were not changed. All 29 tests passed.', '',
        'Yufan sampled from a screened set. The other two members sampled from full source records.',
        'Yufan removed sequences with unknown labels, missing duration, or some flagged phrases.',
        'These choices affect the data distribution. Results across the two groups are not a fair test of data quality.', '',
        f"Train / validation / test: **{split['counts']['train']:,} / {split['counts']['validation']:,} / {split['counts']['test']:,} sequences** (seed 6140).",
        'All 202 pilot or development records stay in training. The previous 16,951 sequences keep their splits.',
        'Records and exact duplicate text stay in one split. Similar text and shared templates may still cross splits.', '',
        '## Methods', '',
        '- **Baseline:** use the current token\'s switch frequency in training. For unseen tokens, use the overall training rate.',
        '- **Prefix model:** use up to five tokens ending at the current position. For rare histories, use a shorter history.',
        '- **Duration:** predict the most common length class for that context: 1-2, 3-6, or 7+ subwords.', '',
        'Only current and past token IDs enter the model. Labels, language IDs, and future tokens are excluded.',
        'We chose settings on validation data only. Both switch thresholds are 0.29047908.',
        'The selected prefix uses five tokens and a minimum count of five.', '',
        '## Test results', '',
        'Switch Macro F1 gives equal weight to each language-pair and switch-type group.',
        'The full experiment has 42 groups. This scoring choice still needs to be checked against the course rules.', '',
        '| Data group | Baseline switch Macro F1 | Prefix switch Macro F1 |',
        '|---|---:|---:|',
    ]
    for label, key in [('Full-record samples: 10 pairs', 'full_record_sample'),
                       ('Screened samples: Yufan\'s 4 pairs', 'screened_retained_sample')]:
        scores = [results[n]['test']['by_sampling_cohort'][key]['macro_pair_type_switch_f1']
                  for n in ['current_token', 'causal_prefix']]
        lines.append(f'| {label} | {pct(scores[0])} | {pct(scores[1])} |')
    scores = [results[n]['test']['macro_pair_type_switch_f1'] for n in ['current_token', 'causal_prefix']]
    lines.append(f'| All 14 pairs: mixed samples | {pct(scores[0])} | {pct(scores[1])} |')
    lines += ['', '| All test data | Baseline | Prefix |', '|---|---:|---:|']
    for label, key in [('Pooled switch F1', 'switch_f1'),
                       ('Duration Macro F1 at gold switches', 'duration_macro_f1_oracle_switch'),
                       ('Joint switch-and-duration Macro F1', 'joint_duration_macro_f1')]:
        values = [results[n]['test']['pooled'][key] for n in ['current_token', 'causal_prefix']]
        lines.append(f'| {label} | {pct(values[0])} | {pct(values[1])} |')
    lines += [
        '', 'The prefix model improves overall switch F1, but duration classification drops.',
        'Philippines-English switch F1 also drops. Lower scores do not prove that the data is worse.',
        'Joint duration scoring counts missed switches, false switches, and wrong length classes.',
        'Gold switches with unknown duration are excluded from duration scoring.', '',
        '## Limits and next steps', '',
        'These scores measure agreement with automatic labels, not human-verified accuracy.',
        'Earlier test samples have already been viewed. This is not a fully new blind test.',
        'The mixed data, label methods, and number of score groups differ from earlier runs.', '',
        'Next: confirm the course metrics and screening rules, then ask language speakers to check an independent sample.',
        'Keep the current results fixed. Choose any new model settings on training and validation data.', '',
        '## Reproduce', '',
        'Python 3.9.6; standard library only. From this folder:', '',
        '```bash', 'python3 -m unittest test_baseline -v',
        'python3 run.py --out runs/reproduction', '```', '',
        'Use an empty output folder. See `runs/all-v3/metrics.csv` for all language-pair scores,',
        '`audit.json` for checks, and `run_manifest.json` for source commits and hashes.',
        'Counts, predictions, fixed splits, and input snapshots are included in the package.',
        'The data owners\' original notes are kept in `source/` in their original language.',
    ]
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (ROOT / 'README.md').write_text(
        '# CS6140 Baseline\n\nRead [REPORT.md](REPORT.md).\n\n'
        'Run: `python3 run.py --out runs/reproduction`. Use an empty output folder.\n')
    files = [p for p in sorted(ROOT.rglob('*')) if p.is_file()
             and '__pycache__' not in p.parts and p.name != 'SHA256SUMS.txt']
    (ROOT / 'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' +
                                      p.relative_to(ROOT).as_posix() + '\n' for p in files))
    archive = ROOT.parent / 'cs6140-all-deliveries-20261018.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files + [ROOT / 'SHA256SUMS.txt']:
            z.write(p, p.relative_to(ROOT.parent))
    print('Saved English report and local package.')


if __name__ == '__main__':
    main()
