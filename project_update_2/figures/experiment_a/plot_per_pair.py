"""Plot validation or test per-pair F1 from saved results, without inference."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/codeswitch-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--split', choices=['validation', 'test'], default='validation')
args = parser.parse_args()
if args.split == 'validation':
    data = json.loads((ROOT / 'comparison/comparison.json').read_text())
    models = data['models']
    n_sequences = 2180
else:
    results = ROOT.parents[1] / 'results'
    runs = ['logreg_enhanced_v1.1_s6140_20261113',
            'mbert_v1.1_s6140_20261113-1805', 'xlmr_v1.1_s6140_20261113-1823']
    models = []
    sequence_sets = []
    for run in runs:
        folder = results / run
        metrics = json.loads((folder / 'metrics_test.json').read_text())
        threshold = json.loads((folder / 'threshold.json').read_text())['threshold']
        meta = json.loads((folder / 'predictions_test_meta.json').read_text())
        assert metrics['split'] == 'test' and metrics['threshold'] == threshold == meta['threshold']
        assert meta['n_sequences'] == 2148 and meta['max_len'] == 128
        with (folder / 'predictions_test.jsonl').open() as f:
            sequence_sets.append({json.loads(line)['sequence_id'] for line in f})
        models.append(metrics)
    assert sequence_sets[0] == sequence_sets[1] == sequence_sets[2]
    n_sequences = 2148
pairs = sorted(models[0]['per_pair'])
assert len(models) == 3 and len(pairs) == 14
assert all(set(m['per_pair']) == set(pairs) for m in models)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'text.color': '#1C2938', 'axes.labelcolor': '#424B57',
    'xtick.color': '#596574', 'ytick.color': '#596574',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#D3D9E0', 'pdf.fonttype': 42,
})
fig, ax = plt.subplots(figsize=(17, 8))
fig.subplots_adjust(left=.065, right=.985, bottom=.22, top=.76)
fig.text(.065, .94, 'Experiment A | Switch F1 by language pair', fontsize=23, weight='bold')
fig.text(.065, .89, f'{args.split.title()}  ·  data-v1.1  ·  {n_sequences:,} sequences  ·  seed 6140  ·  max_len 128',
         fontsize=12, color='#596574')
colors = ['#708174', '#2269B0', '#D46C23']
labels = ['LR baseline (enhanced)', 'mBERT', 'XLM-R']
x = np.arange(len(pairs))
width = .25
for i, (model, color, label) in enumerate(zip(models, colors, labels)):
    scores = np.array([model['per_pair'][p]['switch_f1'] * 100 for p in pairs])
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 100)).all()
    bars = ax.bar(x + (i - 1) * width, scores, width=width, color=color,
                  label=f"{label}  (threshold {model['threshold']:.2f})", zorder=3)
    ax.bar_label(bars, labels=[f'{v:.1f}' for v in scores], padding=4,
                 fontsize=8, rotation=90, color='#424B57')
ax.set_xticks(x, ['ph–en' if p == 'philippines_en' else p.replace('_', '–') for p in pairs])
ax.set_ylabel('Switch F1 (%)', labelpad=12)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_xlim(-.65, len(pairs) - .35)
ax.grid(axis='y', color='#E4E9EF', linewidth=.8, zorder=0)
ax.tick_params(axis='x', length=0, pad=10)
fig.legend(*ax.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.058, .85),
           ncol=3, frameon=False, columnspacing=3)
pooled = '   |   '.join(f"{label}: {model['pooled']['switch_f1'] * 100:.2f}%"
                      for label, model in zip(labels, models))
fig.text(.065, .125, f'Pooled switch F1 — {pooled}', fontsize=11, weight='semibold')
fig.text(.065, .078, 'Each model uses its validation-tuned threshold. Shared scorer; LR uses mBERT tokenization.',
         fontsize=10, color='#596574')
fig.text(.065, .044, 'XLM-R uses different subword positions. ph–en denotes philippines_en. ' + ('Validation results; test reported separately.' if args.split == 'validation' else 'Test results; thresholds fixed from validation.'),
         fontsize=10, color='#596574')
stem = 'experiment_a_per_pair_f1' + ('_test' if args.split == 'test' else '')
for extension in ['png', 'pdf', 'svg']:
    fig.savefig(ROOT / f'{stem}.{extension}', dpi=200, facecolor='white')
svg = ROOT / f'{stem}.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
plt.close(fig)
print('Created PNG, PDF and SVG: 42 bars from verified per-pair metrics.')
