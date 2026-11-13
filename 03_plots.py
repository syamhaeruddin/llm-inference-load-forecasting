"""Tahap 3: grafik eksplorasi dan aktual-vs-prediksi (PNG) untuk laporan."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
SER, RES = os.path.join(BASE, "series"), os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)
C1, C2, C3, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

# --- 1. Eksplorasi: Azure LLM 2024 requests per 5 menit -----------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=False)
for ax, name, title in [(axes[0], "code", "Azure LLM 2024 - layanan code"),
                        (axes[1], "conv", "Azure LLM 2024 - layanan conversation")]:
    df = pd.read_csv(os.path.join(SER, f"azure_llm_2024_{name}_5min.csv"), parse_dates=["timestamp"])
    ax.plot(df["timestamp"], df["requests"], color=C1, lw=1.2)
    ax.set_title(f"{title}: request per 5 menit (UTC)", loc="left", color=INK)
    ax.set_ylabel("request / 5 menit")
    split = df["timestamp"].iloc[int(len(df) * 0.8)]
    ax.axvline(split, color=MUTED, ls="--", lw=1)
    ax.text(split, ax.get_ylim()[1] * 0.97, " batas latih | uji", color=MUTED, va="top")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_azure_llm_2024_requests.png"), dpi=150)

# --- 2. Eksplorasi: DataCenter-Traces CPU -------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 7.5))
for ax, name, col, title in [(axes[0], "alibaba2018", "cpu_util_percent", "Alibaba 2018 - CPU util (%)"),
                             (axes[1], "google2019", "cpu_util_percent", "Google 2019 - CPU util (%)"),
                             (axes[2], "azure_v2", "cpu_usage", "Azure v2 - CPU usage (core, agregat)")]:
    df = pd.read_csv(os.path.join(SER, f"dct_{name}_300s.csv"))
    hours = np.arange(len(df)) * 300 / 3600
    ax.plot(hours, df[col], color=C1, lw=1.0)
    ax.set_title(f"DataCenter-Traces: {title}, per 300 detik", loc="left", color=INK)
    ax.set_xlabel("jam sejak awal trace")
    ax.axvline(hours[int(len(df) * 0.8)], color=MUTED, ls="--", lw=1)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_dct_cpu_series.png"), dpi=150)

# --- 3. Aktual vs prediksi (model terbaik vs seasonal naive) ------------------
met = pd.read_csv(os.path.join(RES, "metrics_summary.csv"))
for series in met["series"].unique():
    sub = met[met.series == series].sort_values("sMAPE")
    best = sub.iloc[0]["model"]
    pred = pd.read_csv(os.path.join(RES, f"pred_{series}.csv"))
    n_show = min(len(pred), 2 * 288)            # 2 hari pertama data uji
    x = np.arange(n_show) * 5 / 60
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(x, pred["y_true"][:n_show], color=INK, lw=1.4, label="aktual")
    ax.plot(x, pred["seasonal_naive"][:n_show], color=C2, lw=1.1, label="seasonal naive")
    if best != "seasonal_naive":
        ax.plot(x, pred[best][:n_show], color=C3, lw=1.1, label=f"{best} (terbaik)")
    ax.set_title(f"{series}: aktual vs prediksi, 2 hari pertama data uji", loc="left", color=INK)
    ax.set_xlabel("jam sejak awal data uji")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig3_pred_{series}.png"), dpi=150)
    plt.close(fig)

# --- 4. Ringkasan sMAPE per model (bar) ---------------------------------------
piv = met.pivot(index="series", columns="model", values="sMAPE")
fig, ax = plt.subplots(figsize=(11, 4))
piv.plot.bar(ax=ax, width=0.8, color=[C1, C2, C3, "#eda100", "#e87ba4", "#6250d6"][:piv.shape[1]])
ax.set_ylabel("sMAPE (%)"); ax.set_xlabel("")
ax.set_title("sMAPE pada data uji per deret dan model (lebih rendah lebih baik)", loc="left", color=INK)
ax.legend(frameon=False, ncol=3)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_smape_summary.png"), dpi=150)
print("figures:", sorted(os.listdir(FIG)))
