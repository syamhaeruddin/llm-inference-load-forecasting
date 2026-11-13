"""English figures for the IEEE TCC manuscript (column width 3.4 in, 8-pt text), built from result files.
  figures/en/fig_regularity_skill.png  : daily-profile ratio vs best skill at 60 min (42 Alibaba clusters + 2 Azure services)
  figures/en/fig_regime_weights.png    : regime-aware Ridge effective weights, Azure conversation, 60 min
  figures/en/fig_autoscale_tradeoff.png: waste vs under-provisioning across margins, 60-min lead time
Usage: python 17_figures_en.py"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "figures", "en")
os.makedirs(OUT, exist_ok=True)
INK, GRID = "#1a1a1a", "#dddddd"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.5, "axes.edgecolor": INK, "savefig.dpi": 400, "savefig.bbox": "tight"})
W = 3.4

# ---------------------------------------------------------------- 1. regularity vs skill
reg = pd.read_csv(os.path.join(BASE, "results", "tables", "regularity_vs_skill_h12.csv"))
meta = pd.read_csv(os.path.join(BASE, "results", "meta", "metrics_summary_full5.csv"))
def skill(svc, model):
    d = meta[(meta.service == svc) & (meta.h == 12)].set_index("model").MAE
    return 1 - d[model] / d["naive"]
az = {"Azure code": (0.58, max(skill("code", m) for m in ["nnls", "meta_select", "best_single", "avg"])),
      "Azure conversation": (0.25, max(skill("conv", m) for m in ["nnls", "meta_select", "best_single", "avg"]))}
fig, ax = plt.subplots(figsize=(W, 2.5))
ax.axhline(0, color=INK, lw=0.7)
ax.scatter(reg.rasio_profil_harian, reg.best_skill, s=14, facecolor="#6b8fb8", edgecolor="white", lw=0.4,
           label="Alibaba cluster (n = 42)", zorder=3)
for name, (x, y) in az.items():
    ax.scatter([x], [y], marker="*", s=90, color="#c0392b", edgecolor="white", lw=0.4, zorder=4)
    ax.annotate(name, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7, color="#c0392b")
x = reg.rasio_profil_harian.values; y = reg.best_skill.values
b1, b0 = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, b0 + b1 * xs, color="#6b8fb8", lw=1, ls="--", label="least-squares fit (clusters)")
ax.set_xlabel(r"daily-profile variance ratio $R_{\mathrm{prof}}$")
ax.set_ylabel("best skill vs. persistence, 60 min")
ax.legend(frameon=False, loc="upper left")
fig.savefig(os.path.join(OUT, "fig_regularity_skill.png")); plt.close(fig)

# ---------------------------------------------------------------- 2. regime weights
w = pd.read_csv(os.path.join(BASE, "results", "meta", "weights_full5.csv"))
w = w[(w.meta == "meta_ridge_regime") & (w.h == 12) & (w.service == "conv") & (w.regime.isin(["weekday", "weekend"]))]
p = w.groupby(["base", "regime"]).weight.mean().unstack()
order = [b for b in ["naive", "seasonal_naive", "ar_ridge_raw", "ar_ridge_diff", "hgb_raw", "hgb_diff", "gru_raw", "gru_diff"] if b in p.index]
p = p.loc[order]
lab = {"naive": "persistence", "seasonal_naive": "seasonal pers.", "ar_ridge_raw": "AR (raw)", "ar_ridge_diff": "AR (diff)",
       "hgb_raw": "HGB (raw)", "hgb_diff": "HGB (diff)", "gru_raw": "GRU (raw)", "gru_diff": "GRU (diff)"}
fig, ax = plt.subplots(figsize=(W, 2.3))
yy = np.arange(len(p)); h = 0.38
ax.barh(yy - h / 2, p["weekday"], height=h, color="#9db4cf", label="weekday")
ax.barh(yy + h / 2, p["weekend"], height=h, color="#2f5d8a", label="weekend")
ax.set_yticks(yy); ax.set_yticklabels([lab[b] for b in p.index]); ax.invert_yaxis()
ax.axvline(0, color=INK, lw=0.7)
ax.set_xlabel("mean effective weight (test day)")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="y", visible=False)
fig.savefig(os.path.join(OUT, "fig_regime_weights.png")); plt.close(fig)

# ---------------------------------------------------------------- 3. autoscaling trade-off, 60-min lead time
s = pd.read_csv(os.path.join(BASE, "results", "autoscale", "sim_summary_full5.csv"))
POL = [("naive", "reactive", "#1a1a1a", "o", "-"), ("nnls", "NNLS", "#2e8b57", "s", "-"),
       ("meta_select", "meta_select", "#c0392b", "D", "-"), ("y_true", "oracle", "#888888", "^", "--")]
fig, axes = plt.subplots(2, 1, figsize=(W, 4.4))
for ax, svc, title in zip(axes, ["code", "conv"], ["(a) Azure code", "(b) Azure conversation"]):
    d = s[(s.service == svc) & (s.h == 12)]
    for pol, name, col, mk, ls in POL:
        q = d[d.policy == pol].sort_values("margin")
        ax.plot(q.underprov_frac * 100, q.waste_gpu_hours, marker=mk, ms=3.5, lw=1.1, ls=ls, color=col, label=name)
        for _, r in q.iterrows():
            if pol in ("naive", "meta_select") and svc == "code":
                ax.annotate(f"{int(round(r.margin * 100))}%", (r.underprov_frac * 100, r.waste_gpu_hours), xytext=(3, 2),
                            textcoords="offset points", fontsize=6, color=col)
    ax.set_title(title, loc="left")
    ax.set_xlabel("under-provisioned intervals (%)"); ax.set_ylabel("wasted GPU-hours per test day")
axes[0].legend(frameon=False, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_autoscale_tradeoff.png")); plt.close(fig)
print("written:", sorted(os.listdir(OUT)))
print("Azure skill 60 min:", {k: round(v[1], 3) for k, v in az.items()})
