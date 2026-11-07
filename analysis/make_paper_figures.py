"""Re-creates the two figures that were added for the paper (pipeline overview and per-quadrat cover-error statistics).
Usage: python analysis/make_paper_figures.py   -> paper/figures/"""
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'paper', 'figures'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
# ---------- Fig 1: pipeline ----------
fig,ax=plt.subplots(figsize=(12,3.9)); ax.set_xlim(0,12); ax.set_ylim(0,3.9); ax.axis('off')
def box(x,y,w,h,title,body,fc):
    ax.add_patch(mp.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.08',fc=fc,ec='#2b2b2b',lw=0.9))
    ax.text(x+w/2,y+h-0.17,title,ha='center',va='top',fontweight='bold',fontsize=8.6)
    ax.text(x+w/2,y+h-0.47,body,ha='center',va='top',fontsize=7.4,linespacing=1.35)
def arr(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=1.1,color='#333'))
C1,C2,C3,C4='#e8f1fb','#eaf6ee','#fdf3e4','#f3ecf8'
box(0.1,2.05,2.2,1.75,'1  Data','2,455 CoralNet quadrats\n404,255 expert points\n40 raw labels → 3 classes\nGrouped 70/15/15 split\nby quadrat (seed 42)',C1)
box(2.6,2.05,2.2,1.75,'2  Pre-processing','Gray-world white balance\nLAB-CLAHE (clip 1.5)\n224×224 point-centred crops\nreflect padding\nTrain: 8,000 crops / class',C1)
box(5.1,2.05,3.4,1.75,'3  Six models, one recipe','Transfer learning (ImageNet):\nMobileNetV2 · ResNet18\nEfficientNet-B0 · ShuffleNetV2\nFrom scratch: CoralCNN-SE\nSelf-supervised ViT: DINOv2 ViT-S/14',C2)
box(8.8,2.05,3.1,1.75,'4  Training','AdamW + OneCycle, AMP\nlabel smoothing 0.1\nGPU augmentation (D4, affine, colour)\nimbalance recipe R0 (ablated)\ncheckpoint = best val macro-F1',C2)
box(0.1,0.1,2.9,1.55,'5  Prior-shift correction','p ∝ p_model · (π_nat / π_train)^τ\nτ ∈ [0, 1.5] tuned on validation\nobjective ½(macro-F1 + bal. acc)\noptional 4-fold TTA',C3)
box(3.25,0.1,2.75,1.55,'6  Protocol A: point crops','6,000 test points (natural prior)\nAcc, bal. acc, macro-F1, κ, MCC\nROC-AUC, PR-AP, ECE\nbootstrap 95% CIs',C3)
box(6.25,0.1,2.75,1.55,'7  Protocol B: whole quadrats','150 held-out quadrats\n224 px windows, stride 112\nGaussian blending (σ = 0.35)\n25,082 points · cover MAE / r\nFriedman + Wilcoxon tests',C3)
box(9.25,0.1,2.65,1.55,'8  Cost & explainability','params, GFLOPs, GPU/CPU latency\nthroughput, Pareto front\nGrad-CAM · IG · Occlusion\ncentre-mass, pointing game\ninsertion / deletion AUC',C4)
arr(2.3,2.93,2.6,2.93); arr(4.8,2.93,5.1,2.93); arr(8.5,2.93,8.8,2.93)
arr(10.35,2.05,10.35,1.65); arr(9.25,0.88,9.0,0.88)
ax.annotate('',xy=(1.55,1.65),xytext=(9.9,1.85),arrowprops=dict(arrowstyle='-|>',lw=1.0,color='#555',connectionstyle='angle,angleA=180,angleB=90,rad=0'))
arr(3.0,0.88,3.25,0.88); arr(6.0,0.88,6.25,0.88)
plt.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_pipeline.png'),dpi=220,facecolor='white'); plt.close()
# ---------- per-quadrat error figure ----------
d=pd.read_csv(os.path.join(ROOT, 'results', 'tables', 'sw_per_image_composition_long.csv'))
d['ae']=d['Err Healthy (pp)'].abs(); d['src']=np.where(d['Quadrat'].str.startswith('mcr'),'Primary source (MCR, n = 128)','Secondary sources (n = 22)')
order=['MobileNetV2','ResNet18','EfficientNet-B0','ShuffleNetV2','CoralCNN-SE','DINOv2-ViT-S/14']
col={'MobileNetV2':'#e08a2e','ResNet18':'#c45a14','EfficientNet-B0':'#f2b84b','ShuffleNetV2':'#a8470f','CoralCNN-SE':'#7b4ea3','DINOv2-ViT-S/14':'#169c80'}
fig,ax=plt.subplots(1,2,figsize=(12,4.3),gridspec_kw=dict(width_ratios=[1.25,1]))
data=[d[d.Model==m]['ae'].values for m in order]
bp=ax[0].boxplot(data,patch_artist=True,showfliers=True,flierprops=dict(marker='.',ms=3,alpha=.5),medianprops=dict(color='k'))
for b,m in zip(bp['boxes'],order): b.set_facecolor(col[m]); b.set_alpha(.85)
ax[0].scatter(range(1,7),[x.mean() for x in data],marker='D',color='white',edgecolor='k',zorder=3,s=28,label='mean')
ax[0].set_xticks(range(1,7)); ax[0].set_xticklabels([m.replace('-ViT-S/14','\nViT-S/14') for m in order],fontsize=8)
ax[0].set_ylabel('|predicted − expert| live-coral cover (pp)'); ax[0].set_title('(a) Per-quadrat absolute cover error (150 quadrats)',fontweight='bold',fontsize=9.5)
ax[0].set_ylim(-0.5,30); ax[0].legend(loc='upper right',fontsize=8); ax[0].grid(axis='y',alpha=.3)
ax[0].text(0.02,0.96,'Friedman χ²(5) = 41.8, p < 0.001',transform=ax[0].transAxes,fontsize=8,va='top')
g=d.groupby(['src','Model'])['ae'].mean().unstack()[order]
x=np.arange(2); w=0.13
for i,m in enumerate(order): ax[1].bar(x+(i-2.5)*w,g[m].values,w,color=col[m],edgecolor='k',lw=.5,label=m)
ax[1].set_xticks(x); ax[1].set_xticklabels(g.index,fontsize=8); ax[1].set_ylabel('Mean absolute cover error (pp)')
ax[1].set_title('(b) Error by image source',fontweight='bold',fontsize=9.5); ax[1].legend(fontsize=7,ncol=2); ax[1].grid(axis='y',alpha=.3)
for s in ('top','right'):
    ax[0].spines[s].set_visible(False); ax[1].spines[s].set_visible(False)
plt.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_cover_error_stats.png'),dpi=220,facecolor='white')
print('Saved figures to', OUT)
