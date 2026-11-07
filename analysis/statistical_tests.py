"""
Statistical analysis reported in the paper (Section 3.7, 3.8 and 4.4, 4.6).

  * 95% bootstrap CIs for point-crop accuracy and macro-F1 (5,000 multinomial resamples of each confusion matrix)
  * Friedman test across the six models on per-quadrat absolute live-coral / dead-cover error (150 held-out quadrats)
  * paired Wilcoxon signed-rank tests vs the best model with Holm correction, paired bootstrap CIs of the mean difference
  * error split by image source (primary MCR-LTER quadrats vs secondary sources)

Usage:  python analysis/statistical_tests.py            (reads results/, writes results/statistical_tests.json)
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
rng = np.random.default_rng(42)
master = json.load(open(os.path.join(RES, "results_master.json")))


def f1_acc(c):
    c = np.asarray(c, float); tp = np.diag(c)
    p = tp / np.maximum(c.sum(0), 1); r = tp / np.maximum(c.sum(1), 1)
    return float(np.mean(np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0))), float(tp.sum() / c.sum())


# 1 ─ bootstrap CIs on the point-crop test set
boot = {}
for m, c in master["point_crop_cm"].items():
    c = np.array(c); n = c.sum()
    B = rng.multinomial(n, (c / n).ravel(), size=5000).reshape(-1, 3, 3)
    fs = np.array([f1_acc(b) for b in B]); f1, acc = f1_acc(c)
    boot[m] = dict(acc=acc, acc_ci=np.percentile(fs[:, 1], [2.5, 97.5]).tolist(), macro_f1=f1, macro_f1_ci=np.percentile(fs[:, 0], [2.5, 97.5]).tolist())
print("Point-crop bootstrap 95% CIs"); print(pd.DataFrame(boot).T.round(4).to_string())

# 2 ─ per-quadrat cover-error tests
d = pd.read_csv(os.path.join(RES, "tables", "sw_per_image_composition_long.csv"))
d["ae_h"] = d["Err Healthy (pp)"].abs(); d["ae_d"] = d["Err Dead (pp)"].abs()
wh = d.pivot(index="Quadrat", columns="Model", values="ae_h"); wd = d.pivot(index="Quadrat", columns="Model", values="ae_d")
fr_h = stats.friedmanchisquare(*[wh[c] for c in wh.columns]); fr_d = stats.friedmanchisquare(*[wd[c] for c in wd.columns])
ref = wh.mean().idxmin()
rows = []
for c in wh.columns:
    if c == ref: continue
    diff = (wh[c] - wh[ref]).values
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    rows.append(dict(model=c, mae=wh[c].mean(), median_ae=wh[c].median(), delta_mae=diff.mean(),
                     delta_ci=np.percentile(bs, [2.5, 97.5]).tolist(), wilcoxon_p=stats.wilcoxon(wh[ref], wh[c]).pvalue))
rows.sort(key=lambda r: r["wilcoxon_p"]); k = len(rows); running = 0
for i, r in enumerate(rows):                                   # Holm step-down adjustment
    running = max(running, min(1.0, r["wilcoxon_p"] * (k - i))); r["holm_p"] = running
print(f"\nFriedman (live-coral cover error): chi2={fr_h.statistic:.2f}, p={fr_h.pvalue:.2e}; dead cover: chi2={fr_d.statistic:.2f}, p={fr_d.pvalue:.2e}")
print(f"Pairwise vs best model ({ref}):"); print(pd.DataFrame(rows).round(4).to_string(index=False))

# 3 ─ error by image source
d["source"] = np.where(d["Quadrat"].str.startswith("mcr"), "MCR-LTER (primary)", "secondary")
by_src = d.groupby(["source", "Model"])["ae_h"].mean().unstack().round(3)
print("\nMean absolute live-coral cover error by source (pp)"); print(by_src.to_string())

out = dict(point_crop_bootstrap=boot, friedman_live=dict(chi2=fr_h.statistic, p=fr_h.pvalue), friedman_dead=dict(chi2=fr_d.statistic, p=fr_d.pvalue),
           best_model_cover=ref, pairwise_vs_best=rows, mean_rank=wh.rank(axis=1).mean().to_dict(), mae_by_source=by_src.to_dict())
json.dump(out, open(os.path.join(RES, "statistical_tests.json"), "w"), indent=2, default=float)
print("\nSaved results/statistical_tests.json")
