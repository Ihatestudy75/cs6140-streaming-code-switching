"""Validate the threshold, report per-pair PRF, and analyze boundary groups.

Read-only model analysis: no refitting and no test-based parameter selection.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/cs6140-lr-matplotlib')
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from metrics import choose_threshold
from model import predict_prefixes
from train import DEFAULT_REFERENCE, load_data, sha
from boundary_rules import boundary_kind, boundary_group

ROOT=Path(__file__).resolve().parent


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def write_csv(path,rows):
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(y,pred,mask=None):
    if mask is not None:y,pred=y[mask],pred[mask]
    tp=int(np.count_nonzero((y==1)&(pred==1)))
    fp=int(np.count_nonzero((y==0)&(pred==1)))
    fn=int(np.count_nonzero((y==1)&(pred==0)))
    tn=int(np.count_nonzero((y==0)&(pred==0)))
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    return {'positions':len(y),'gold_switches':tp+fn,'predicted_switches':tp+fp,
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}


def macro_group_f1(y,pred,codes):
    length=int(codes.max())+1
    tp=np.bincount(codes,weights=(y==1)&(pred==1),minlength=length)
    fp=np.bincount(codes,weights=(y==0)&(pred==1),minlength=length)
    fn=np.bincount(codes,weights=(y==1)&(pred==0),minlength=length)
    denom=2*tp+fp+fn
    return float(np.divide(2*tp,denom,out=np.zeros_like(tp),where=denom>0).mean())


def prf_table(rows,include_support=True):
    header='| Language pair | Precision | Recall | F1 | Gold switches |\n|---|---:|---:|---:|---:|'
    return header+'\n'+'\n'.join(f"| {r['pair']} | {100*r['precision']:.2f}% | {100*r['recall']:.2f}% | {100*r['f1']:.2f}% | {r['gold_switches']} |" for r in rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument('--out',type=Path,default=ROOT/'analysis-boundaries-v1')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Use a new output directory.')
    args.out.mkdir(parents=True)
    run=ROOT/'runs/logistic-v1'
    manifest=json.loads((run/'run_manifest.json').read_text())
    reference=args.reference_root
    for name,digest in manifest['source_sha256'].items():assert sha(ROOT/name)==digest
    data,types,audit=load_data(reference,20)
    del data['train']
    bundle=joblib.load(run/'logistic_model.joblib')
    valid=data['validation'];test=data['test']
    val_scores,_=predict_prefixes(bundle,[s['inputs']['input_ids'] for s in valid])
    choice=choose_threshold(valid,val_scores.tolist(),types)
    threshold=choice['threshold']
    assert threshold==bundle['switch_config']['threshold'],'Selected validation threshold changed.'
    val_y=np.array([s['targets']['switch'] for s in valid])
    val_groups=[(s['provenance']['language_pair'],types[s['provenance']['sequence_id']]) for s in valid]
    code_map={g:i for i,g in enumerate(sorted(set(val_groups)))}
    codes=np.array([code_map[g] for g in val_groups])
    assert len(code_map)==42
    threshold_rows=[]
    for t in sorted(set(np.linspace(0,1,101).tolist()+[threshold])):
        p=(val_scores>=t).astype(int)
        threshold_rows.append(dict(threshold=t,macro_pair_type_f1=macro_group_f1(val_y,p,codes),**summarize(val_y,p)))
    write_csv(args.out/'validation_threshold_curve.csv',threshold_rows)
    chosen_row=next(r for r in threshold_rows if r['threshold']==threshold)
    assert abs(chosen_row['macro_pair_type_f1']-choice['validation_macro_pair_type_switch_f1'])<1e-12
    default_row=next(r for r in threshold_rows if r['threshold']==0.5)
    write_json(args.out/'threshold_selection.json',{
        'selected_on':'validation only','objective':'Positive switch F1 averaged equally over 42 pair/type cells',
        'comparison_operator':'>=','selected':chosen_row,'default_0_5':default_row,
        'method':'Exact sweep of every unique validation score, unchanged from the prior run.',
        'curve_note':'The chart/CSV uses a 0.01 display grid plus the exact selected threshold; selection uses all unique scores.',
        'model_retrained':False,'test_used_for_selection':False})
    print('Validation threshold independently confirmed:',threshold,flush=True)

    saved=[json.loads(line) for line in (run/'test_predictions.jsonl').open()]
    assert len(saved)==len(test)
    for p,s in zip(saved,test):
        assert all(p[k]==s['provenance'][k] for k in s['provenance'])
        assert p['gold_switch']==s['targets']['switch']
    scores=np.array([p['probability'] for p in saved])
    y=np.array([p['gold_switch'] for p in saved])
    pred=(scores>=threshold).astype(int)
    np.testing.assert_array_equal(pred,[p['predicted_switch'] for p in saved])
    pairs=np.array([s['provenance']['language_pair'] for s in test])
    pair_names=sorted(set(pairs))
    pair_rows=[dict(pair=pair,**summarize(y,pred,pairs==pair)) for pair in pair_names]
    write_csv(args.out/'language_pair_prf.csv',pair_rows)
    totals=summarize(y,pred)
    for metric,fn in [('precision',precision_score),('recall',recall_score),('f1',f1_score)]:
        assert abs(totals[metric]-fn(y,pred))<1e-12
    original_results=json.loads((run/'results.json').read_text())
    assert abs(totals['f1']-original_results['test']['pooled']['switch_f1'])<1e-12

    raw_rows={}
    wanted={s['provenance']['sequence_id'] for s in valid+test}
    with (reference/'runs/all-v3/combined/labeled_tokens.jsonl').open() as file:
        for line in file:
            r=json.loads(line)
            if r['sequence_id'] in wanted:raw_rows[r['sequence_id']]=r
    def groups_for(samples):
        kinds=[]
        for s in samples:
            info=s['provenance'];r=raw_rows[info['sequence_id']]
            t=info['position']
            assert r['switch_valid_mask'][t] and r['switch_labels'][t]==s['targets']['switch']
            kinds.append(boundary_kind(r['text'],r['offsets'],t))
        return np.array(kinds),np.array([boundary_group(k) for k in kinds])
    kinds,groups=groups_for(test)
    val_kinds,val_boundary_groups=groups_for(valid)
    group_names=['punctuation','within_clause_proxy','unclassified']
    group_rows=[];detail_rows=[];pair_boundary_rows=[]
    for split,cy,cp,cgroups,ckinds in [
        ('validation',val_y,(val_scores>=threshold).astype(int),val_boundary_groups,val_kinds),
        ('test',y,pred,groups,kinds)]:
        for group in group_names:
            if np.any(cgroups==group):
                row=dict(split=split,group=group,**summarize(cy,cp,cgroups==group))
                row['share_of_gold_switches']=row['gold_switches']/int(cy.sum())
                group_rows.append(row)
        for kind in sorted(set(ckinds)):
            detail_rows.append(dict(split=split,kind=kind,**summarize(cy,cp,ckinds==kind)))
    for pair in pair_names:
        total_gold=int(y[pairs==pair].sum())
        for group in group_names:
            mask=(pairs==pair)&(groups==group)
            if not mask.any():continue
            row=dict(pair=pair,group=group,**summarize(y,pred,mask))
            row['share_of_pair_gold_switches']=row['gold_switches']/total_gold if total_gold else None
            pair_boundary_rows.append(row)
    write_csv(args.out/'boundary_prf.csv',group_rows)
    write_csv(args.out/'punctuation_details.csv',detail_rows)
    write_csv(args.out/'language_pair_boundary_prf.csv',pair_boundary_rows)
    assert sum(r['positions'] for r in group_rows if r['split']=='test')==len(test)
    assert sum(r['gold_switches'] for r in group_rows if r['split']=='test')==int(y.sum())

    # Frozen earlier models: same test positions and boundary definitions.
    earlier=json.loads((reference/'runs/all-v3/results.json').read_text())
    model_preds={'logistic_regression':pred}
    for name in ('current_token','causal_prefix'):
        before=[json.loads(line) for line in (reference/f'runs/all-v3/{name}_test_predictions.jsonl').open()]
        assert len(before)==len(test)
        for p,s in zip(before,test):
            assert all(p[k]==s['provenance'][k] for k in s['provenance'])
            assert p['gold_switch']==s['targets']['switch']
        model_preds[name]=np.array([p['predicted_switch'] for p in before])
    comparison=[]
    for name,mpred in model_preds.items():
        for group in group_names:
            if np.any(groups==group):
                comparison.append(dict(model=name,group=group,**summarize(y,mpred,groups==group)))
    write_csv(args.out/'model_boundary_comparison.csv',comparison)

    plt.rcParams.update({'figure.dpi':120,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    ts=[r['threshold'] for r in threshold_rows]
    for key,label in [('macro_pair_type_f1','42-group macro F1'),('f1','Pooled F1')]:
        axes[0].plot(ts,[100*r[key] for r in threshold_rows],label=label)
    for key,label in [('precision','Precision'),('recall','Recall')]:
        axes[1].plot(ts,[100*r[key] for r in threshold_rows],label=label)
    for ax in axes:
        ax.axvline(threshold,color='black',linestyle='--',label=f'Selected {threshold:.3f}')
        ax.set(xlabel='Threshold',ylabel='Validation score (%)',ylim=(0,102));ax.legend(fontsize=8)
    axes[0].set_title('Validation-only threshold selection');axes[1].set_title('Validation precision / recall trade-off')
    fig.tight_layout();fig.savefig(args.out/'threshold_selection.png',bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5.5));pos=np.arange(len(pair_rows));width=.25
    for i,(key,label) in enumerate([('precision','Precision'),('recall','Recall'),('f1','F1')]):
        ax.bar(pos+(i-1)*width,[100*r[key] for r in pair_rows],width,label=label)
    ax.set(xticks=pos,xticklabels=pair_names,ylabel='Test score (%)',ylim=(0,100),title=f'Per-pair scores at validation-selected threshold {threshold:.3f}')
    ax.tick_params(axis='x',rotation=45);ax.legend();fig.tight_layout();fig.savefig(args.out/'language_pair_prf.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5));pos=np.arange(2);width=.25
    for i,(name,label) in enumerate([('current_token','Token counts'),('causal_prefix','Prefix counts'),('logistic_regression','Logistic Regression')]):
        selected=[next(r for r in comparison if r['model']==name and r['group']==g) for g in group_names[:2]]
        for ax,key in zip(axes,['recall','f1']):
            bars=ax.bar(pos+(i-1)*width,[100*r[key] for r in selected],width,label=label)
            ax.bar_label(bars,fmt='%.1f',padding=2,fontsize=8)
    for ax,key in zip(axes,['Recall','F1']):
        ax.set(xticks=pos,xticklabels=['After punctuation','Non-punctuation\n(within-clause proxy)'],ylabel=f'Test {key} (%)',ylim=(0,100),title=f'{key} by boundary group')
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(args.out/'boundary_comparison.png',bbox_inches='tight');plt.close(fig)

    pooled_group=[r for r in group_rows if r['split']=='test']
    punct=next(r for r in pooled_group if r['group']=='punctuation')
    within=next(r for r in pooled_group if r['group']=='within_clause_proxy')
    pair_details=[]
    for pair in pair_names:
        a=next(r for r in pair_boundary_rows if r['pair']==pair and r['group']=='punctuation')
        b=next(r for r in pair_boundary_rows if r['pair']==pair and r['group']=='within_clause_proxy')
        pair_details.append(f"| {pair} | {100*a['share_of_pair_gold_switches']:.2f}% | {100*a['recall']:.2f}% ({a['gold_switches']}) | {100*b['recall']:.2f}% ({b['gold_switches']}) |")
    comparison_lines=[]
    for name in ['current_token','causal_prefix','logistic_regression']:
        a=next(r for r in comparison if r['model']==name and r['group']=='punctuation')
        b=next(r for r in comparison if r['model']==name and r['group']=='within_clause_proxy')
        comparison_lines.append(f"| {name} | {100*a['recall']:.2f}% | {100*b['recall']:.2f}% | {100*a['f1']:.2f}% | {100*b['f1']:.2f}% |")
    report=f'''# Threshold, language pairs, and punctuation analysis

21 October 2026. Frozen Logistic Regression model; same 14 language pairs and original sequence splits. No model was retrained, no test-driven threshold or feature adjustment was made, and the existing results are preserved.

## 1. Choose threshold on validation only

The exact validation sweep reconfirmed **threshold = {threshold:.8f}**. Predict a switch when score >= threshold. The selection metric is positive-switch F1 averaged equally over the same 42 language-pair × CS-type cells, preserving the original comparison protocol. It is not pooled F1, accuracy, or recall alone. Equal-score predictions are grouped in the exact sweep; ties keep the higher threshold.

| Validation threshold | Precision | Recall | Pooled F1 | 42-group macro F1 |
|---|---:|---:|---:|---:|
| Default 0.5 | {100*default_row['precision']:.2f}% | {100*default_row['recall']:.2f}% | {100*default_row['f1']:.2f}% | {100*default_row['macro_pair_type_f1']:.2f}% |
| Selected {threshold:.8f} | {100*chosen_row['precision']:.2f}% | {100*chosen_row['recall']:.2f}% | {100*chosen_row['f1']:.2f}% | {100*chosen_row['macro_pair_type_f1']:.2f}% |

Lowering the default threshold changes the precision/recall trade-off. The selected threshold is global: no separate pair-specific or punctuation-specific threshold was fitted. The plot uses a 0.01 display grid plus the exact chosen point; actual selection checks all unique validation scores. The frozen threshold is then applied to the test set.

![Validation threshold selection](threshold_selection.png)

## 2. Test precision, recall, and F1 by language pair

Across 54,166 test positions and 3,091 actual switches, precision is **{100*totals['precision']:.2f}%**, recall **{100*totals['recall']:.2f}%**, and F1 **{100*totals['f1']:.2f}%**. These are pooled metrics, not accuracy. TP={totals['tp']:,}, FP={totals['fp']:,}, FN={totals['fn']:,}, TN={totals['tn']:,}. Each pair below uses the same threshold.

{prf_table(pair_rows)}

![Per-pair scores](language_pair_prf.png)

## 3. Punctuation vs within-clause proxy

**Definition fixed before computing the groups:** at an eligible boundary after token t, inspect only observed text through that token's end offset and ignore trailing whitespace. If the last observed character has a Unicode punctuation category, classify it as `punctuation`; otherwise classify it as `within_clause_proxy`. Empty observed prefixes are `unclassified`. This supports Chinese, Latin-script, Arabic, and other punctuation without language-specific token decoding. Labels still refer to a change between t and t+1 under the existing neutral-character inheritance policy.

**Important limit:** non-punctuation is a proxy for within-clause switching, not a syntactic annotation. No clause parser or human clause labels are available. A punctuation boundary is not necessarily a sentence/clause boundary: quotes, apostrophes, decimal points, and abbreviations can be included. Sentence-ending, clause-separating, and other punctuation are therefore also reported separately in `punctuation_details.csv`. These group labels are used only for analysis, never as new model inputs, and the three-class cs_type label is not used to determine a boundary's category.

| Test boundary group | Eligible positions | Gold switches | Share of all gold switches | Precision | Recall | F1 | Missed switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| After punctuation | {punct['positions']:,} | {punct['gold_switches']:,} | {100*punct['share_of_gold_switches']:.2f}% | {100*punct['precision']:.2f}% | {100*punct['recall']:.2f}% | {100*punct['f1']:.2f}% | {punct['fn']:,} |
| Non-punctuation / within-clause proxy | {within['positions']:,} | {within['gold_switches']:,} | {100*within['share_of_gold_switches']:.2f}% | {100*within['precision']:.2f}% | {100*within['recall']:.2f}% | {100*within['f1']:.2f}% | {within['fn']:,} |

Group precision uses **all eligible positions** in that group, including non-switches, so false positives are counted. Recall uses actual switches in that group. It would be invalid to compute precision using only actual switches. Unscorable switch positions remain excluded by the original mask; the shown punctuation share describes the scorable labeled subset only.

### Compare all three frozen methods

| Model | Punctuation recall | Within-clause proxy recall | Punctuation F1 | Within-clause proxy F1 |
|---|---:|---:|---:|---:|
{chr(10).join(comparison_lines)}

![Punctuation comparison](boundary_comparison.png)

### Per-language-pair boundary analysis

Numbers in parentheses are actual-switch support; small groups have less stable recall estimates. See `language_pair_boundary_prf.csv` for each group's precision, recall, F1, and confusion counts.

| Pair | Gold switches after punctuation | Punctuation recall (support) | Within-clause proxy recall (support) |
|---|---:|---:|---:|
{chr(10).join(pair_details)}

## Interpretation

Logistic Regression's punctuation recall is {100*punct['recall']:.2f}%, versus {100*within['recall']:.2f}% for the within-clause proxy: a gap of {100*(punct['recall']-within['recall']):.2f} percentage points. The latter group contains {within['gold_switches']:,} switches and {within['fn']:,} missed switches. These measured counts show where errors occur; they do not by themselves prove that punctuation causes the model's decisions or that a longer neural model will fix them.

The prior presentation screenshot cannot be compared as an identical experiment: it covers six pairs, reports recall under its own threshold, and does not establish identical split/annotation/grouping rules. Our matched comparison above uses identical test positions, labels, and punctuation definitions for all three methods. Scores measure agreement with automatic labels; independent human checking is still needed. The old test results were already viewed earlier, so this remains exploratory analysis rather than a new blind evaluation.

## Files and reproduction

`threshold_selection.json` records the exact decision; `validation_threshold_curve.csv` records the display curve. `language_pair_prf.csv`, `boundary_prf.csv`, `punctuation_details.csv`, `language_pair_boundary_prf.csv`, and `model_boundary_comparison.csv` contain aggregate results. No sample text or data files are copied into this analysis folder.

From the parent model folder:

```bash
python3 -m unittest test_model test_boundary_rules -v
python3 analyze_boundaries.py --out analysis-boundaries-reproduction
```
'''
    (args.out/'REPORT.md').write_text(report)
    write_json(args.out/'analysis_manifest.json',{
        'data_sha256':audit['data_sha256'],'split_sha256':audit['split_sha256'],
        'model_sha256':sha(run/'logistic_model.joblib'),'test_predictions_sha256':sha(run/'test_predictions.jsonl'),
        'source_sha256':{name:sha(ROOT/name) for name in ['analyze_boundaries.py','boundary_rules.py']},
        'threshold':threshold,'same_threshold_as_training_run':True,'model_retrained':False,
        'test_positions':len(test),'validation_positions':len(valid),'validation_groups':len(code_map),
        'boundary_definition':'Last observed non-whitespace character through token t end offset; Unicode P* versus non-P*; non-P* is a within-clause proxy only.',
        'test_boundary_counts':dict(Counter(groups)),
        'checks':{'pair_counts_sum_to_total':sum(r['positions'] for r in pair_rows)==len(test),
                  'gold_and_position_alignment':True,'saved_predictions_unchanged':True,
                  'sklearn_prf_match':True,'validation_threshold_reproduced':True}})
    write_json(args.out/'summary.json',{'threshold':threshold,'validation_default':default_row,
        'validation_selected':chosen_row,'test_pooled':totals,'test_by_pair':pair_rows,
        'test_by_boundary':pooled_group,'test_model_boundary_comparison':comparison})
    print(json.dumps({'threshold':threshold,'test_pooled':totals,'test_by_boundary':pooled_group,
                      'model_boundary_comparison':comparison},indent=2),flush=True)

if __name__=='__main__':main()
