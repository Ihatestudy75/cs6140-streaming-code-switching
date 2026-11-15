"""Build validation comparison tables from pinned run artifacts; verify with shared scorer."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'deliverables/cora/logistic_regression'))
from metrics import score_group

RUNS = [
    ('LR baseline (enhanced)', 'ee4ba3b776d10abb16af6ee9c1367ff341030fff', 'logreg_enhanced_v1.1_s6140_20261113'),
    ('mBERT', '99a9ea36699dceb3173bb6e62f291eb4567a3bc4', 'mbert_v1.1_s6140_20261113-1805'),
    ('XLM-R', '99a9ea36699dceb3173bb6e62f291eb4567a3bc4', 'xlmr_v1.1_s6140_20261113-1823'),
]
FIELDS = {
    'Switch precision': 'precision', 'Switch recall': 'recall', 'Switch F1': 'switch_f1',
    'Duration accuracy (oracle switch)': 'duration_accuracy_oracle_switch',
    'Duration macro F1 (oracle switch)': 'duration_macro_f1_oracle_switch',
    'Joint duration macro F1': 'joint_duration_macro_f1',
}


def read(revision, run, name):
    path = f'project_update_2/results/{run}/{name}'
    raw = subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=REPO)
    return raw, hashlib.sha256(raw).hexdigest()


def items(rows):
    return [({'targets': {'switch': r['y_switch'], 'duration': r['y_duration']}},
             r['p_switch'], r['duration_pred']) for r in rows]


def verify(actual, expected):
    for key in list(FIELDS.values()) + ['positions', 'gold_switches', 'duration_support']:
        assert abs(actual[key] - expected[key]) < 1e-12, (key, actual[key], expected[key])


def main():
    results = []
    sequence_sets = []
    for label, revision, run in RUNS:
        sources, hashes = {}, {}
        for name in ['config.json', 'threshold.json', 'metrics_validation.json', 'predictions_validation.jsonl']:
            raw, hashes[name] = read(revision, run, name)
            sources[name] = [json.loads(line) for line in raw.splitlines()] if name.endswith('jsonl') else json.loads(raw)
        config, threshold, metrics = (sources[n] for n in ['config.json', 'threshold.json', 'metrics_validation.json'])
        rows = sources['predictions_validation.jsonl']
        assert config['max_len'] == 128 and config['seed'] == 6140 and config['data_tag'] == 'data-v1.1'
        assert metrics['split'] == 'validation'
        sequence_sets.append({r['sequence_id'] for r in rows})
        tuned = score_group(items(rows), threshold['threshold'])
        verify(tuned, metrics['pooled'])
        for pair, expected in metrics['per_pair'].items():
            verify(score_group(items([r for r in rows if r['language_pair'] == pair]), threshold['threshold']), expected)
        fixed = score_group(items(rows), 0.5)
        results.append(dict(model=label, revision=revision, run_id=run, source_sha256=hashes,
                            threshold=threshold['threshold'], pooled=tuned,
                            switch_f1_at_0_5=fixed['switch_f1'], per_pair=metrics['per_pair'],
                            universality=metrics['universality'], source_io_version=metrics['io_version']))
    assert sequence_sets[0] == sequence_sets[1] == sequence_sets[2]
    assert len(sequence_sets[0]) == 2180
    assert all(set(r['per_pair']) == set(results[0]['per_pair']) for r in results)
    lines = ['# Three-model validation comparison', '',
             'Data: data-v1.1; seed: 6140; max_len: 128; 2,180 identical validation sequences; 14 language pairs. '
             'The baseline is the enhanced logistic regression run from PR #16.', '',
             'All pooled and per-pair metrics below were recomputed from predictions with the shared '
             '`deliverables/cora/logistic_regression/metrics.py` scorer and checked against the saved metrics. '
             'Baseline source IO version is 0.5; transformer source IO version is 0.6.', '',
             '## Overall results', '', 'Scores are percentages. Each model uses its own validation-tuned threshold.', '']
    def table(headers, rows):
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
        lines.append('')
    table(['Metric'] + [r['model'] for r in results],
          [[label] + [f"{r['pooled'][key] * 100:.2f}" for r in results] for label, key in FIELDS.items()] +
          [['Switch threshold'] + [str(r['threshold']) for r in results],
           ['Switch F1 at fixed threshold 0.5'] + [f"{r['switch_f1_at_0_5'] * 100:.2f}" for r in results],
           ['Valid positions'] + [str(r['pooled']['positions']) for r in results],
           ['Gold switches'] + [str(r['pooled']['gold_switches']) for r in results],
           ['Duration support'] + [str(r['pooled']['duration_support']) for r in results]])
    lines += ['## Per-pair switch F1', '']
    table(['Language pair'] + [r['model'] for r in results],
          [[pair] + [f"{r['per_pair'][pair]['switch_f1'] * 100:.2f}" for r in results]
           for pair in sorted(results[0]['per_pair'])] +
          [['Unweighted pair mean'] + [f"{r['universality']['sigma_all']['mean'] * 100:.2f}" for r in results],
           ['Pair standard deviation (pp, ddof=0)'] + [f"{r['universality']['sigma_all']['value'] * 100:.2f}" for r in results]])
    lines += ['## Interpretation and scope', '',
              '- mBERT and XLM-R improve pooled switch F1 over the enhanced LR baseline by '
              f"{100 * (results[1]['pooled']['switch_f1'] - results[0]['pooled']['switch_f1']):.2f} and "
              f"{100 * (results[2]['pooled']['switch_f1'] - results[0]['pooled']['switch_f1']):.2f} percentage points, respectively.",
              '- LR and mBERT use mBERT tokenization. XLM-R uses its own tokenization: valid positions, switch support, '
              'and duration labels differ. Duration bins are S=1–2, M=3–6, L=7+ subwords of each tokenizer.',
              '- Oracle-switch duration scores use gold switches with known duration. Joint duration F1 also requires '
              'correct switch detection. Pooled switch F1 and the unweighted mean of pair F1 are different aggregates.',
              '- These are validation results used for model/threshold selection, not held-out test performance. '
              'The runs do not have matched hyperparameter search budgets; small differences are descriptive, '
              'not evidence of statistical significance.', '']
    (OUT / 'README.md').write_text('\n'.join(lines).rstrip() + '\n')
    scorer = REPO / 'deliverables/cora/logistic_regression/metrics.py'
    (OUT / 'comparison.json').write_text(json.dumps(dict(data_tag='data-v1.1', split='validation',
        n_sequences=2180, max_len=128, seed=6140, verification='pooled and all 14 pairs match shared scorer',
        scorer_sha256=hashlib.sha256(scorer.read_bytes()).hexdigest(), models=results), indent=2) + '\n')
    print('Verified all three models: pooled metrics, 14 pairs, identical validation sequence IDs. Wrote README.md and comparison.json.')


if __name__ == '__main__':
    main()
