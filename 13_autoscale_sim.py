"""
Tahap 7: simulasi autoscaling berbasis prediksi (RQ4, H4) pada prediksi tahap 5 (results/meta/pred_<svc>_h<h>_fold<f>.csv).

Model sistem (sederhana, transparan):
  - Kapasitas satu GPU c = P95(Y pada periode latih) / N_REF token per 5 menit, sehingga beban P95 butuh N_REF GPU.
  - Horizon h = jeda penyediaan (waktu muat model + boot): keputusan pada t berlaku untuk t+h.
  - Kebijakan: GPU disediakan n_t = ceil( y_hat(t+h) * (1 + margin) / c ), dibatasi [N_MIN, N_MAX].
    * reactive  : y_hat = nilai terakhir (prediksi "naive" pada jeda h)  -> autoscaler reaktif klasik
    * <model>   : prediksi model (best_single, nnls, meta_ridge_regime, meta_select, ...)
    * oracle    : y_hat = y aktual (batas bawah biaya untuk margin tsb)
  - Kebutuhan r_t = ceil( y(t+h) / c ).
Metrik per (layanan, h, fold, kebijakan, margin):
  gpu_hours        : total GPU-jam disediakan
  waste_gpu_hours  : sum max(0, n - r) * 5/60
  underprov_frac   : porsi interval dengan n < r
  deficit_share    : sum max(0, y - n*c) / sum y   (porsi token yang melebihi kapasitas = proksi pelanggaran SLO)
  cost_index       : gpu_hours + PENALTY * deficit_gpu_hours, PENALTY = 5 (satu GPU-jam kekurangan dihitung 5x)
Pemakaian: python 13_autoscale_sim.py [--services code conv] [--horizons 3 12] [--margins 0 0.1 0.2] [--tag full2]
"""
import argparse, os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SER = os.path.join(BASE, "series")
META = os.path.join(BASE, "results", "meta")
RES = os.path.join(BASE, "results", "autoscale")
FIG = os.path.join(BASE, "figures")
os.makedirs(RES, exist_ok=True)
N_REF, N_MIN, N_MAX, PENALTY = 100, 1, 400, 5.0
POLICIES = ["naive", "best_single", "nnls", "meta_ridge_regime", "meta_select", "oracle_single", "y_true"]
LABEL = {"naive": "reaktif (naive)", "best_single": "model tunggal terbaik", "nnls": "NNLS", "meta_ridge_regime": "meta Ridge rezim",
         "meta_select": "meta_select", "oracle_single": "oracle model tunggal", "y_true": "oracle (aktual)"}


def simulate(y, yhat, c, margin):
    n = np.clip(np.ceil(np.maximum(yhat, 0) * (1 + margin) / c), N_MIN, N_MAX)
    r = np.clip(np.ceil(y / c), N_MIN, N_MAX)
    dt = 5 / 60
    deficit_tokens = np.maximum(0, y - n * c)
    return {"gpu_hours": float(n.sum() * dt), "waste_gpu_hours": float(np.maximum(0, n - r).sum() * dt),
            "underprov_frac": float((n < r).mean()), "deficit_share": float(deficit_tokens.sum() / y.sum()),
            "cost_index": float(n.sum() * dt + PENALTY * (deficit_tokens / c).sum() * dt)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--services", nargs="+", default=["code", "conv"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6, 12])
    ap.add_argument("--margins", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--tag", default="full2")
    ap.add_argument("--nref", type=int, default=100)
    ap.add_argument("--penalty", type=float, default=5.0)
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    N_REF, PENALTY = a.nref, a.penalty
    globals()["N_REF"], globals()["PENALTY"] = a.nref, a.penalty
    a.tag = f"{a.tag}_n{a.nref}_p{int(a.penalty)}" if (a.nref != 100 or a.penalty != 5.0) else a.tag
    rows = []
    for svc in a.services:
        y_all = pd.read_csv(os.path.join(SER, f"azure_llm_2024_{svc}_5min.csv"))["requests"]  # hanya untuk panjang
        feat = pd.read_csv(os.path.join(SER, f"azure_llm_2024_{svc}_features_5min.csv"))
        y_train = feat["total_tokens"].values[: int(0.7 * len(feat))]
        c = np.percentile(y_train, 95) / N_REF
        for h in a.horizons:
            for f in range(2):
                p = pd.read_csv(os.path.join(META, f"pred_{svc}_h{h}_fold{f}.csv"))
                y = p["y_true"].values
                for pol in POLICIES:
                    if pol not in p.columns:
                        continue
                    for m in a.margins:
                        s = simulate(y, p[pol].values, c, m)
                        rows.append({"service": svc, "h": h, "fold": f, "policy": pol, "margin": m, "capacity_per_gpu": c, **s})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"sim_runs_{a.tag}.csv"), index=False)
    summ = df.groupby(["service", "h", "policy", "margin"]).mean(numeric_only=True).drop(columns="fold").reset_index()
    summ.to_csv(os.path.join(RES, f"sim_summary_{a.tag}.csv"), index=False)
    with open(os.path.join(RES, f"config_{a.tag}.json"), "w") as fh:
        json.dump({"N_REF": N_REF, "N_MIN": N_MIN, "N_MAX": N_MAX, "PENALTY": PENALTY, **vars(a)}, fh, indent=2)

    if a.no_fig:
        pd.set_option("display.width", 220)
        print(summ[summ.margin == 0.1].pivot_table(index=["service", "h"], columns="policy", values="cost_index").round(1).to_string())
        raise SystemExit(0)
    # --- grafik: trade-off pemborosan vs under-provisioning (kurva margin) per kebijakan, h = 3 dan 12
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK = "#0b0b0b"
    COL = {"naive": INK, "best_single": "#2a78d6", "nnls": "#1baf7a", "meta_ridge_regime": "#6250d6",
           "meta_select": "#eb6834", "oracle_single": "#9a9a94", "y_true": "#9a9a94"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    hs = [h for h in [3, 12] if h in a.horizons]
    fig, axes = plt.subplots(len(a.services), len(hs), figsize=(6 * len(hs), 4.2 * len(a.services)), squeeze=False)
    for i, svc in enumerate(a.services):
        for j, h in enumerate(hs):
            ax = axes[i, j]
            d = summ[(summ.service == svc) & (summ.h == h)]
            for pol in ["naive", "best_single", "nnls", "meta_ridge_regime", "meta_select", "y_true"]:
                dp = d[d.policy == pol].sort_values("margin")
                if dp.empty:
                    continue
                ax.plot(dp.underprov_frac * 100, dp.waste_gpu_hours, marker="o", ms=4, lw=1.6, color=COL[pol],
                        ls="--" if pol == "y_true" else "-", label=LABEL[pol])
                for _, r in dp.iterrows():
                    ax.annotate(f"{int(r.margin * 100)}%", (r.underprov_frac * 100, r.waste_gpu_hours), fontsize=7,
                                xytext=(3, 3), textcoords="offset points", color=COL[pol])
            ax.set_xlabel("interval under-provisioned (%)"); ax.set_ylabel("GPU-jam terbuang per hari uji")
            ax.set_title(f"{svc}, jeda penyediaan {h * 5} menit", loc="left", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Simulasi autoscaling (P95 beban latih = {N_REF} GPU): tiap titik = satu margin (0-30%); kiri-bawah lebih baik",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(FIG, "fig13_autoscale_tradeoff.png"), dpi=150)

    pd.set_option("display.width", 220)
    show = summ[summ.margin == 0.1].pivot_table(index=["service", "h"], columns="policy", values=["waste_gpu_hours", "underprov_frac", "cost_index"])
    print("margin 10%:\n", show.round(2).to_string())
    print("\nselesai:", os.path.join(RES, f"sim_summary_{a.tag}.csv"))
