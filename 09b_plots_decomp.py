"""Grafik ablasi tahap 4: sMAPE per varian representasi (raw/diff/causal/noncausal) vs horizon."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results", "decomp")
FIG = os.path.join(BASE, "figures")
INK, MUTED = "#0b0b0b", "#52514e"
COL = {"raw": "#2a78d6", "diff": "#1baf7a", "causal": "#eb6834", "noncausal": "#9a9a94"}
LAB = {"raw": "raw (nilai absolut)", "diff": "diff (selisih vs nilai terakhir)",
       "causal": "dekomposisi kausal", "noncausal": "dekomposisi non-kausal (bocor)"}
hmin = {1: 5, 3: 15, 6: 30, 12: 60}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
s = pd.read_csv(os.path.join(RES, "metrics_summary_all.csv"))
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
for i, svc in enumerate(["code", "conv"]):
    for j, model in enumerate(["hgb", "gru"]):
        ax = axes[i, j]
        d = s[(s.service == svc) & (s.model == model)]
        for v in ["raw", "diff", "causal", "noncausal"]:
            dv = d[d.variant == v].sort_values("h")
            if dv.empty:
                continue
            ax.errorbar([hmin[h] for h in dv.h], dv.sMAPE, yerr=dv.sMAPE_std.fillna(0), color=COL[v], marker="o",
                        ms=4, lw=1.8, capsize=2, label=LAB[v], ls="--" if v == "noncausal" else "-")
        nv = d[d.variant == "raw"].pivot_table(index="h", values="sMAPE") if False else \
            s[(s.service == svc) & (s.model == "naive") & (s.variant == "raw")].sort_values("h")
        ax.plot([hmin[h] for h in nv.h], nv.sMAPE, color=INK, lw=1.4, marker="s", ms=3, label="naive (pembanding)")
        ax.set_title(f"{svc} - {model.upper()}", loc="left", color=INK)
        ax.set_xticks([5, 15, 30, 60])
        if i == 1:
            ax.set_xlabel("horizon (menit)")
        if j == 0:
            ax.set_ylabel("sMAPE (%) data uji")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("Ablasi representasi target: target selisih (diff) stabil dan terbaik; dekomposisi kausal klasik memburuk; "
             "versi non-kausal tampak bagus karena bocor", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(FIG, "fig8_decomp_ablation.png"), dpi=150)
print("ok")
