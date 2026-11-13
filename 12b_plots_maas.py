"""Tahap 6b: grafik & analisis tambahan Alibaba: distribusi skill score per metode, dan hubungan
kekuatan musiman harian (autokorelasi lag 288) dengan skill terbaik per cluster."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results", "maas")
FIG = os.path.join(BASE, "figures")
TAG = sys.argv[1] if len(sys.argv) > 1 else "full"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
runs = pd.read_csv(os.path.join(RES, f"metrics_runs_{TAG}.csv"))
per = runs.groupby(["cluster_id", "h", "model"]).agg(MAE=("MAE", "mean"), sMAPE=("sMAPE", "mean")).reset_index()
hmin = {1: 5, 3: 15, 6: 30, 12: 60}
METHODS = ["best_single", "nnls", "meta_select", "meta_ridge_regime", "avg"]
LAB = {"best_single": "model tunggal terbaik", "nnls": "NNLS", "meta_select": "meta_select", "meta_ridge_regime": "meta Ridge rezim", "avg": "rata-rata"}
COL = {"best_single": "#2a78d6", "nnls": "#1baf7a", "meta_select": "#eb6834", "meta_ridge_regime": "#6250d6", "avg": "#9a9a94"}

# fig11: distribusi skill score (1 - MAE/MAE_naive) per metode per horizon, boxplot
fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True)
for ax, h in zip(axes, [1, 3, 6, 12]):
    d = per[per.h == h].pivot(index="cluster_id", columns="model", values="MAE")
    skill = 1 - d.div(d["naive"], axis=0)
    data = [skill[m].clip(-1, 1).values for m in METHODS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True, flierprops=dict(marker=".", ms=3, alpha=0.5))
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(COL[m]); patch.set_alpha(0.35); patch.set_edgecolor(COL[m])
    for med in bp["medians"]:
        med.set_color(INK)
    ax.axhline(0, color=INK, lw=1, ls="--")
    ax.set_xticks(range(1, len(METHODS) + 1)); ax.set_xticklabels([LAB[m] for m in METHODS], rotation=35, ha="right", fontsize=8)
    ax.set_title(f"horizon {hmin[h]} menit", loc="left", color=INK)
axes[0].set_ylabel("skill score vs naive (1 = sempurna, 0 = setara naive)\ndipotong pada -1")
fig.suptitle("Alibaba, 42 cluster: skill score per metode; garis putus = naive. Metode kombinasi berada di sekitar nol, meta rezim merugikan",
             x=0.01, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(FIG, "fig11_maas_skill_boxplot.png"), dpi=150); plt.close(fig)

# kekuatan musiman harian per cluster (autokorelasi lag 288 pada deret), vs skill terbaik di h=12
w = pd.read_csv(os.path.join(BASE, "series", "maas_week5min_by_cluster.csv"))
acf = {}
for cid, d in w.groupby("cluster_id"):
    y = d.sort_values("time_s").sm_util_w.values
    if len(y) == 2016 and y.std() > 0:
        y0 = y - y.mean(); acf[cid] = float(np.dot(y0[288:], y0[:-288]) / np.dot(y0, y0))
d12 = per[per.h == 12].pivot(index="cluster_id", columns="model", values="MAE")
sk12 = (1 - d12.div(d12["naive"], axis=0))
best12 = sk12[["nnls", "meta_select", "best_single", "avg"]].max(axis=1)
tab = pd.DataFrame({"acf288": pd.Series(acf), "best_skill_h12": best12, "skill_nnls_h12": sk12["nnls"],
                    "skill_meta_select_h12": sk12["meta_select"]}).dropna()
tab.to_csv(os.path.join(RES, f"seasonality_vs_skill_{TAG}.csv"))
rho = tab[["acf288", "best_skill_h12"]].corr(method="spearman").iloc[0, 1]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(tab.acf288, tab.best_skill_h12.clip(-1, 1), color="#2a78d6", s=28, alpha=0.8)
ax.axhline(0, color=INK, lw=1, ls="--")
ax.set_xlabel("kekuatan musiman harian (autokorelasi lag 24 jam)"); ax.set_ylabel("skill score terbaik vs naive, horizon 60 menit")
ax.set_title(f"Alibaba: semakin kuat pola harian, semakin besar manfaat forecasting (Spearman rho = {rho:.2f})", loc="left", color=INK, fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig12_maas_seasonality_vs_skill.png"), dpi=150)
pd.set_option("display.width", 200)
print("sMAPE rata-rata lintas cluster:\n", per.pivot_table(index="h", columns="model", values="sMAPE")[
    ["naive", "seasonal_naive", "best_single", "nnls", "meta_select", "meta_ridge_regime", "oracle_single"]].round(2).to_string())
print("\nacf288: median", round(tab.acf288.median(), 3), "| cluster dgn acf288 > 0.5:", int((tab.acf288 > 0.5).sum()),
      "| skill terbaik h=12 > 0:", int((tab.best_skill_h12 > 0).sum()), "dari", len(tab), "| Spearman rho:", round(rho, 3))
print("\ncluster dengan GRU (5 terbesar), h=12:")
g = runs[runs.has_gru].groupby(["service", "model"]).sMAPE.mean().unstack()
print(g[[c for c in ["naive", "gru_raw", "gru_diff", "hgb_diff", "best_single", "nnls", "meta_select"] if c in g.columns]].round(2).to_string())
print("\nmeta_select memilih (h=12):", runs[(runs.h == 12) & (runs.model == "naive")].meta_select_is.value_counts().to_dict())
