"""
Tahap 6b: uji statistik lintas cluster (Alibaba) — Friedman + post-hoc Nemenyi (critical difference), skill score,
dan jumlah cluster di mana tiap metode menang. Satu "subjek" = satu cluster (rata-rata 2 fold), per horizon.
Pemakaian: python 12_stats_maas.py [tag]
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results", "maas")
TAG = sys.argv[1] if len(sys.argv) > 1 else "full"
METHODS = ["naive", "seasonal_naive", "best_single", "avg", "nnls", "meta_ridge_regime", "meta_hgb_noregime", "meta_select"]


def nemenyi_cd(k, n, alpha=0.05):
    """Critical difference Nemenyi: q_alpha * sqrt(k(k+1)/(6n)), q dari studentized range / sqrt(2)."""
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6 * n))


runs = pd.read_csv(os.path.join(RES, f"metrics_runs_{TAG}.csv"))
per = runs.groupby(["cluster_id", "h", "model"]).agg(MAE=("MAE", "mean"), sMAPE=("sMAPE", "mean")).reset_index()
out_rows, rank_rows = [], []
for h in sorted(per.h.unique()):
    d = per[(per.h == h) & (per.model.isin(METHODS))].pivot(index="cluster_id", columns="model", values="MAE")[METHODS].dropna()
    n, k = d.shape
    # skill score vs naive, per cluster
    skill = 1 - d.div(d["naive"], axis=0)
    ranks = d.rank(axis=1, method="average")
    avg_rank = ranks.mean()
    chi2, p = stats.friedmanchisquare(*[d[m].values for m in METHODS])
    cd = nemenyi_cd(k, n)
    wins = (d.idxmin(axis=1).value_counts()).reindex(METHODS).fillna(0).astype(int)
    for m in METHODS:
        out_rows.append({"h": h, "model": m, "n_clusters": n, "skill_mean": skill[m].mean(), "skill_median": skill[m].median(),
                         "skill_q25": skill[m].quantile(0.25), "skill_q75": skill[m].quantile(0.75),
                         "avg_rank": avg_rank[m], "wins": wins[m],
                         "better_than_naive_frac": float((d[m] < d["naive"]).mean()),
                         "friedman_chi2": chi2, "friedman_p": p, "nemenyi_CD": cd,
                         "sig_vs_naive": bool(abs(avg_rank[m] - avg_rank["naive"]) > cd),
                         "sig_vs_best_single": bool(abs(avg_rank[m] - avg_rank["best_single"]) > cd)})
    # Wilcoxon berpasangan meta_select vs naive dan vs best_single (pelengkap)
    for ref in ["naive", "best_single"]:
        try:
            w = stats.wilcoxon(d["meta_select"], d[ref], alternative="less")
            rank_rows.append({"h": h, "test": f"wilcoxon meta_select < {ref}", "stat": w.statistic, "p": w.pvalue})
        except ValueError as e:
            rank_rows.append({"h": h, "test": f"wilcoxon meta_select < {ref}", "stat": np.nan, "p": np.nan})
res = pd.DataFrame(out_rows)
res.to_csv(os.path.join(RES, f"stats_friedman_{TAG}.csv"), index=False)
pd.DataFrame(rank_rows).to_csv(os.path.join(RES, f"stats_wilcoxon_{TAG}.csv"), index=False)
pd.set_option("display.width", 220)
for h in sorted(res.h.unique()):
    r = res[res.h == h]
    print(f"\n== h={h} ({int(r.n_clusters.iloc[0])} cluster): Friedman chi2={r.friedman_chi2.iloc[0]:.1f}, p={r.friedman_p.iloc[0]:.2e}, Nemenyi CD={r.nemenyi_CD.iloc[0]:.2f} ==")
    print(r[["model", "skill_mean", "skill_median", "avg_rank", "wins", "better_than_naive_frac", "sig_vs_naive", "sig_vs_best_single"]]
          .sort_values("avg_rank").round(3).to_string(index=False))
print("\n", pd.DataFrame(rank_rows).round(4).to_string(index=False))
