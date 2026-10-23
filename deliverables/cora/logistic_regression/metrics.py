"""Frozen scoring functions from the existing baseline, unchanged for comparability."""
import math
from collections import Counter, defaultdict
SCREENED_PAIRS = {'fr_en', 'de_en', 'it_en', 'philippines_en'}

def group(s, types):
    return (s['provenance']['language_pair'], types[s['provenance']['sequence_id']])

def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

def choose_threshold(samples, scores, types):
    """Choose the threshold with the best validation macro F1.

    On a tie, keep the higher threshold. Group equal scores together.
    """
    totals = Counter(group(s, types) for s in samples if s['targets']['switch'])
    groups = sorted({group(s, types) for s in samples})
    counts = {g: [0, 0, totals[g]] for g in groups}
    best = (0.0, math.nextafter(max(scores), math.inf))
    ranked = sorted(zip(scores, samples), key=lambda x: x[0], reverse=True)
    i = 0
    while i < len(ranked):
        score = ranked[i][0]
        while i < len(ranked) and ranked[i][0] == score:
            s = ranked[i][1]
            c = counts[group(s, types)]
            if s['targets']['switch']:
                c[0] += 1
                c[2] -= 1
            else:
                c[1] += 1
            i += 1
        value = sum(f1(*c) for c in counts.values()) / len(groups)
        if value > best[0]:
            best = value, score
    return {'threshold': best[1], 'validation_macro_pair_type_switch_f1': best[0]}

def score_group(items, threshold):
    tp = fp = fn = tn = correct_duration = duration_n = 0
    cm = [[0] * 3 for _ in range(3)]
    # Joint scoring counts missed and false switches and wrong length classes.
    # Skip gold switches whose duration is unknown.
    joint_tp, joint_fp, joint_fn = [0]*3, [0]*3, [0]*3
    for s, score, pred_d in items:
        y, d = s['targets']['switch'], s['targets']['duration']
        pred = score >= threshold
        tp += int(pred and y == 1)
        fp += int(pred and y == 0)
        fn += int(not pred and y == 1)
        tn += int(not pred and y == 0)
        if d >= 0:
            duration_n += 1
            correct_duration += int(d == pred_d)
            cm[d][pred_d] += 1
        if y == 1 and d < 0:
            continue
        if pred:
            if d == pred_d:
                joint_tp[pred_d] += 1
            else:
                joint_fp[pred_d] += 1
        if d >= 0 and (not pred or d != pred_d):
            joint_fn[d] += 1
    duration_f1 = [f1(cm[k][k], sum(cm[j][k] for j in range(3) if j != k), sum(cm[k][j] for j in range(3) if j != k)) for k in range(3)]
    return {'positions': len(items), 'gold_switches': tp+fn, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp/(tp+fp) if tp+fp else 0.0, 'recall': tp/(tp+fn) if tp+fn else 0.0,
            'switch_f1': f1(tp, fp, fn), 'duration_support': duration_n,
            'duration_accuracy_oracle_switch': correct_duration/duration_n if duration_n else None,
            'duration_macro_f1_oracle_switch': sum(duration_f1)/3,
            'duration_confusion_matrix': cm,
            'joint_duration_macro_f1': sum(f1(joint_tp[k], joint_fp[k], joint_fn[k]) for k in range(3))/3}

def evaluate(samples, predictions, threshold, types):
    cells, pairs, cstypes = defaultdict(list), defaultdict(list), defaultdict(list)
    items = []
    for s, (prob, d) in zip(samples, predictions):
        item = (s, prob, d)
        pair, typ = group(s, types)
        items.append(item)
        cells[pair + ' | ' + typ].append(item)
        pairs[pair].append(item)
        cstypes[typ].append(item)
    by_cell = {k: score_group(v, threshold) for k, v in sorted(cells.items())}
    cohorts = {}
    for name, screened in [('full_record_sample', False), ('screened_retained_sample', True)]:
        chosen = [item for item in items if (item[0]['provenance']['language_pair'] in SCREENED_PAIRS) == screened]
        chosen_cells = [v for k, v in by_cell.items() if (k.split(' | ')[0] in SCREENED_PAIRS) == screened]
        if chosen:
            cohorts[name] = {'pooled': score_group(chosen, threshold), 'macro_pair_type_switch_f1': sum(v['switch_f1'] for v in chosen_cells)/len(chosen_cells)}
    return {'pooled': score_group(items, threshold), 'by_pair_type': by_cell, 'by_sampling_cohort': cohorts,
            'macro_pair_type_switch_f1': sum(v['switch_f1'] for v in by_cell.values()) / len(by_cell),
            'by_pair': {k: score_group(v, threshold) for k, v in sorted(pairs.items())},
            'by_cs_type': {k: score_group(v, threshold) for k, v in sorted(cstypes.items())}}
