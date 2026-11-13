"""Grafik tahap 5: pembanding kombinasi vs naive/best_single/oracle, dan bobot meta per rezim."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results", "meta")
FIG = os.path.join(BASE, "figures")
INK, MUTED = "#0b0b0b", "#52514e"
hmin = {1: 5, 3: 15, 6: 30, 12: 60}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
s = pd.read_csv(os.path.join(RES, "metrics_summary_full.csv"))
w = pd.read_csv(os.path.join(RES, "weights_full.csv"))

# fig9: sMAPE vs horizon untuk kombinasi utama
series = [("naive", "naive (persistence)", INK, "-", "s"), ("best_single", "model tunggal terbaik (dipilih dari OOF)", "#2a78d6", "-", "o"),
          ("nnls", "NNLS: bobot statis dipelajari", "#1baf7a", "-", "o"), ("meta_ridge_regime", "meta Ridge sadar-rezim", "#eb6834", "-", "o"),
          ("meta_hgb_regime", "meta HGB sadar-rezim", "#eda100", "-", "o"), ("oracle_single", "oracle: model tunggal terbaik di uji", "#9a9a94", "--", "^")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, svc in zip(axes, ["code", "conv"]):
    d = s[s.service == svc]
    for m, lab, c, ls, mk in series:
        dm = d[d.model == m].sort_values("h")
        ax.plot([hmin[h] for h in dm.h], dm.sMAPE, color=c, ls=ls, marker=mk, ms=4, lw=1.8 if m in ("nnls", "meta_ridge_regime") else 1.3, label=lab)
    ax.set_xticks([5, 15, 30, 60]); ax.set_xlabel("horizon (menit)"); ax.set_ylabel("sMAPE (%) data uji")
    ax.set_title(f"Layanan {svc}", loc="left", color=INK)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Tahap 5: NNLS statis terbaik pada code (pola harian stabil); meta sadar-rezim terbaik pada conv horizon 30-60 menit (level bergeser)",
             x=0.01, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(FIG, "fig9_meta_vs_horizon.png"), dpi=150); plt.close(fig)

# fig10: bobot meta_ridge_regime conv h=12 per rezim (hari kerja vs akhir pekan)
d = w[(w.meta == "meta_ridge_regime") & (w.service == "conv") & (w.h == 12) & (w.regime.isin(["weekday", "weekend"]))]
p = d.pivot_table(index="base", columns="regime", values="weight").reindex(
    ["naive", "seasonal_naive", "hgb_diff", "hgb_raw", "gru_diff", "gru_raw", "ar_ridge_diff", "ar_ridge_raw"])
fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(p)); wd = 0.38
ax.bar(x - wd / 2, p["weekday"], wd, color="#2a78d6", label="hari kerja (Jumat 17 Mei)")
ax.bar(x + wd / 2, p["weekend"], wd, color="#eb6834", label="akhir pekan (Sabtu 18 Mei)")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(p.index, rotation=25, ha="right"); ax.set_ylabel("bobot efektif rata-rata")
ax.set_title("conv, horizon 60 menit: meta Ridge sadar-rezim memindahkan bobot ke naive pada akhir pekan", loc="left", color=INK, fontsize=10.5)
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig10_meta_weights_regime.png"), dpi=150)
print("ok")
