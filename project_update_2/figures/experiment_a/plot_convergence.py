from pathlib import Path
import os,json,csv,hashlib,subprocess
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codeswitch-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
ROOT=Path(__file__).resolve().parent

REPO=ROOT.parents[2]
SOURCE_COMMIT='7fe37273bdfe6fd0e90e5f6788072dafe63161f4'
RUNS={'mbert':'mbert_v1.1_s6140_20261113-1805','xlmr':'xlmr_v1.1_s6140_20261113-1823'}
def read_source(model, filename):
    return subprocess.check_output(['git','show',f'{SOURCE_COMMIT}:project_update_2/results/{RUNS[model]}/{filename}'],cwd=REPO,text=True)

WINDOW=100
COLORS={'mbert':'#2269B0','xlmr':'#D46C23'}
LABELS={'mbert':'mBERT','xlmr':'XLM-R'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'semibold','axes.labelcolor':'#424B57','text.color':'#1C2938','axes.edgecolor':'#D3D9E0','xtick.color':'#596574','ytick.color':'#596574','axes.spines.top':False,'axes.spines.right':False,'grid.color':'#E4E9EF','grid.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(2,3,figsize=(15,8.7))
fig.subplots_adjust(top=.81,bottom=.17,left=.065,right=.97,hspace=.43,wspace=.26)
fig.text(.065,.95,'Experiment A | Training convergence',fontsize=23,weight='bold')
fig.text(.065,.907,'data-v1.1  ·  seed 6140  ·  max_len 128  ·  batch size 16  ·  4 epochs',fontsize=11,color='#596574')
legend=[Line2D([0],[0],color=COLORS[k],lw=2.6,label=LABELS[k]) for k in COLORS]
legend += [Line2D([0],[0],color='#697482',lw=2,label='Training: 100-step moving average'),Line2D([0],[0],color='#697482',lw=1.8,ls='--',marker='o',markersize=4,label='Validation: epoch-end observations')]
fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.058,.885),ncol=4,frameon=False,fontsize=10,handlelength=2.5,columnspacing=2.4)
losses=[('loss_total','Total loss'),('loss_switch','Switch loss'),('loss_duration','Duration loss')]
metrics=[('switch_f1','Validation switch F1'),('switch_recall','Validation switch recall'),('duration_acc','Validation duration accuracy')]
export=[];summary=[]
for model in COLORS:
 rows=[json.loads(l) for l in read_source(model,'train_log.jsonl').splitlines()]
 train=[r for r in rows if r['split']=='train'];valid=[r for r in rows if r['split']=='validation'];cfg=json.loads(read_source(model,'config.json'))
 steps=np.array([r['step'] for r in train]);assert np.array_equal(steps,np.arange(1,cfg['total_steps']+1));assert len(valid)==4
 per_epoch=cfg['total_steps']/cfg['epochs_run'];x=steps/per_epoch;vx=np.array([r['step'] for r in valid])/per_epoch
 for ax,(key,title) in zip(axs[0],losses):
  y=np.array([r[key] for r in train]);assert np.isfinite(y).all()
  cs=np.r_[0,np.cumsum(y)];starts=np.maximum(0,np.arange(len(y))+1-WINDOW);ends=np.arange(len(y))+1;ma=(cs[ends]-cs[starts])/(ends-starts)
  ax.plot(x,y,color=COLORS[model],lw=.45,alpha=.08,rasterized=True)
  ax.plot(x,ma,color=COLORS[model],lw=1.8)
  ax.plot(vx,[r[key] for r in valid],color=COLORS[model],lw=1.6,ls='--',marker='o',markersize=4)
  ax.set_title(title,loc='left',pad=10);ax.set_ylabel('Cross-entropy loss');ax.set_ylim(bottom=0)
  for step,epoch,raw,smooth in zip(steps,x,y,ma):export.append({'model':model,'split':'train','step':int(step),'epoch':float(epoch),'metric':key,'raw':float(raw),'moving_average':float(smooth)})
 for ax,(key,title) in zip(axs[1],metrics):
  vals=[r[key] for r in valid];ax.plot(vx,vals,color=COLORS[model],lw=1.8,ls='--',marker='o',markersize=5)
  ax.set_title(title,loc='left',pad=10);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
  for r in valid:export.append({'model':model,'split':'validation','step':r['step'],'epoch':r['epoch'],'metric':key,'raw':r[key],'moving_average':''})
 for r in valid:
  for key,_ in losses:export.append({'model':model,'split':'validation','step':r['step'],'epoch':r['epoch'],'metric':key,'raw':r[key],'moving_average':''})
 threshold=json.loads(read_source(model,'threshold.json'));summary.append(f"{LABELS[model]}: {threshold['validation_switch_f1']:.3f} at threshold {threshold['threshold']:.2f}")
for ax in axs.flat:
 ax.set_xlim(0,4.12);ax.set_xticks([0,1,2,3,4]);ax.set_xlabel('Epoch');ax.grid(True,axis='y');ax.set_axisbelow(True)
axs[1,0].set_ylim(.50,.66);axs[1,1].set_ylim(.40,.60);axs[1,2].set_ylim(.52,.62)
fig.text(.065,.105,'Training curves: trailing mean of up to 100 steps; faint lines show raw batch losses. Validation points are not smoothed.',fontsize=9,color='#596574')
fig.text(.065,.077,'Switch curves use the logged 0.5 threshold. Final threshold-tuned F1 (separate evaluation): '+'; '.join(summary)+'.',fontsize=9,color='#596574')
fig.text(.065,.049,'Duration accuracy is measured at gold switches with known duration. Duration labels use each tokenizer\'s own subword counts.',fontsize=9,color='#596574')
for ext in ('png','pdf'):fig.savefig(ROOT/f'experiment_a_convergence.{ext}',dpi=200,facecolor='white')
manifest={'repository':'https://github.com/Ihatestudy75/cs6140-streaming-code-switching','commit':SOURCE_COMMIT,'runs':RUNS,'smoothing':'Trailing arithmetic mean, 100 optimizer steps; partial windows at beginning; training losses only','validation':'Unsmooth epoch-end values; switch metrics at argmax threshold 0.5','source_sha256':{model+'_train_log.jsonl':hashlib.sha256(read_source(model,'train_log.jsonl').encode()).hexdigest() for model in RUNS},'outputs':['experiment_a_convergence.png','experiment_a_convergence.pdf']}
(ROOT/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Generated PNG, PDF and provenance manifest; verified 4652 training steps and 4 validation observations per model.')
