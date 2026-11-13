"""Tahap 6b: grafik & tabel ringkas hasil eksperimen total_tokens (dari 05_experiment_tokens.py)."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results", "tokens")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)
TAG = sys.argv[1] if len(sys.argv) > 1 else "full"
INK, MUTED = "#0b0b0b", "#52514e"
COLORS = {"naive": "#0b0b0b", "seasonal_naive": "#9a9a94", "ar_ridge": "#2a78d6", "hgb": "#eb6834",
          "lstm": "#1baf7a", "gru": "#eda100", "tcn": "#e87ba4", "transformer": "#6250d6"}
LABEL = {"naive": "Naive", "seasonal_naive": "Seasonal naive", "ar_ridge": "AR Ridge", "hgb": "HGB",
         "lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "transformer": "Transformer"}
ORDER = list(COLORS)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

runs = pd.read_csv(os.path.join(RES, f"metrics_runs_{TAG}.csv"))
summ = pd.read_csv(os.path.join(RES, f"metrics_summary_{TAG}.csv"))
hmin = {1: 5, 3: 15, 6: 30, 12: 60}

# ---- fig5: sMAPE vs horizon per model, dua panel ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
for ax, svc in zip(axes, ["code", "conv"]):
    s = summ[summ.service == svc]
    for m in ORDER:
        d = s[s.model == m].sort_values("h")
        if d.empty:
            continue
        ax.errorbar([hmin[h] for h in d.h], d.sMAPE_mean, yerr=d.sMAPE_std.fillna(0), color=COLORS[m],
                    lw=2 if m in ("naive", "lstm", "gru", "hgb") else 1.2, marker="o", ms=4,
                    capsize=2, label=LABEL[m], alpha=0.95)
    ax.set_xticks([5, 15, 30, 60]); ax.set_xlabel("horizon (menit)")
    ax.set_ylabel("sMAPE (%) pada data uji")
    ax.set_title(f"Layanan {svc}: sMAPE vs horizon (rata-rata 2 fold x 2 seed)", loc="left", color=INK)
axes[1].legend(frameon=False, ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_tokens_smape_vs_horizon.png"), dpi=150); plt.close(fig)

# ---- fig6: per fold (hari uji) untuk conv --------------------------------------
pf = runs.groupby(["service", "h", "model", "fold"]).sMAPE.mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, h in zip(axes, [1, 12]):
    d = pf[(pf.service == "conv") & (pf.h == h)].pivot(index="model", columns="fold", values="sMAPE").reindex(ORDER)
    x = np.arange(len(d)); w = 0.38
    ax.bar(x - w / 2, d[0], w, color="#2a78d6", label="fold 0: uji Jumat 17 Mei")
    ax.bar(x + w / 2, d[1], w, color="#eb6834", label="fold 1: uji Sabtu 18 Mei")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in d.index], rotation=25, ha="right")
    ax.set_title(f"conv, horizon {hmin[h]} menit: sMAPE per hari uji", loc="left", color=INK)
axes[0].set_ylabel("sMAPE (%)"); axes[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_tokens_conv_per_fold.png"), dpi=150); plt.close(fig)

# ---- fig7: aktual vs prediksi code h=12 fold 1 ----------------------------------
p = pd.read_csv(os.path.join(RES, "pred_code_h12_fold1.csv"), parse_dates=["timestamp"])
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(p.timestamp, p.y_true / 1e6, color=INK, lw=1.6, label="aktual")
ax.plot(p.timestamp, p.naive_s0 / 1e6, color="#9a9a94", lw=1.1, label="naive (nilai 60 menit sebelumnya)")
ax.plot(p.timestamp, p.hgb_s0 / 1e6, color=COLORS["hgb"], lw=1.1, label="HGB")
lstm_cols = [c for c in p.columns if c.startswith("lstm_s")]
ax.plot(p.timestamp, p[lstm_cols].mean(axis=1) / 1e6, color=COLORS["lstm"], lw=1.1, label="LSTM (rata-rata seed)")
ax.set_ylabel("total token per 5 menit (juta)")
ax.set_title("Layanan code, horizon 60 menit, hari uji Kamis 16 Mei 2024 (UTC)", loc="left", color=INK)
ax.legend(frameon=False, ncol=4, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_tokens_code_h12_pred.png"), dpi=150); plt.close(fig)

# ---- tabel ringkas ------------------------------------------------------------
tab = summ.pivot(index=["service", "h"], columns="model", values="sMAPE_mean")[ORDER].round(2)
tab.to_csv(os.path.join(RES, f"table_smape_{TAG}.csv"))
tab_dm = summ.pivot(index=["service", "h"], columns="model", values="DM_p_median")[ORDER[1:]].round(3)
tab_dm.to_csv(os.path.join(RES, f"table_dm_p_{TAG}.csv"))
pd.set_option("display.width", 200)
print("sMAPE (%) rata-rata:\n", tab.to_string())
print("\nDM p-value median vs naive (negatif DM = lebih baik dari naive):\n", tab_dm.to_string())
print("\nconv per fold (sMAPE):\n", pf[pf.service == "conv"].pivot_table(index=["h", "model"], columns="fold", values="sMAPE").round(2).to_string())
print("\nfigures:", [f for f in sorted(os.listdir(FIG)) if "tokens" in f])
