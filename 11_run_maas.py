"""
Tahap 6: pipeline level 0 + level 1 (10_metalearner.run_df) pada deret cluster Alibaba cluster-trace-v2026-maas.
Deret: series/maas_week5min_by_cluster.csv -> per cluster, Y = sm_util_w (utilisasi SM berbobot GPU, %), 2.016 interval.
Hanya cluster dengan 2.016 interval lengkap (45). Fitur kalender: hod dari time_s (hari dimulai pada kelipatan 86400);
is_weekend tidak diketahui (tanpa tanggal kalender) -> 0 untuk semua baris, dicatat sebagai keterbatasan.
GRU hanya pada --top-dl cluster terbesar (rata-rata gpu_count); model lain pada semua cluster.
Keluaran: results/maas/metrics_runs_<tag>.csv, metrics_summary_<tag>.csv, weights_<tag>.csv, oof_*.npz, pred_*.csv
Pemakaian: python 11_run_maas.py [--top-dl 5] [--seeds 0] [--horizons 1 3 6 12] [--max-clusters N] [--quick]
"""
import argparse, importlib, json, os, time
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SER = os.path.join(BASE, "series")
RES = os.path.join(BASE, "results", "maas")
os.makedirs(RES, exist_ok=True)
meta = importlib.import_module("10_metalearner")
TARGET = "sm_util_w"


def cluster_frames():
    w = pd.read_csv(os.path.join(SER, "maas_week5min_by_cluster.csv"))
    sizes = w.groupby("cluster_id").size()
    full = sizes[sizes == 2016].index
    w = w[w.cluster_id.isin(full)].sort_values(["cluster_id", "time_s"])
    rank = w.groupby("cluster_id").gpu_count.mean().sort_values(ascending=False)
    frames, excluded = {}, []
    for i, cid in enumerate(rank.index):
        d = w[w.cluster_id == cid].copy()
        yv = d[TARGET].values
        # cluster degenerate: deret konstan (telemetri kosong) atau > 50% interval bernilai nol
        if yv.std() < 1e-9 or (yv == 0).mean() > 0.5:
            excluded.append({"key": f"c{i:02d}", "cluster_id": cid, "gpu_count_mean": float(rank[cid]),
                             "std": float(yv.std()), "zero_frac": float((yv == 0).mean())})
            continue
        d.index = pd.to_datetime(d.time_s, unit="s")            # waktu relatif (bukan kalender)
        pos = (d.time_s % 86400) / 86400.0
        d["hod_sin"], d["hod_cos"] = np.sin(2 * np.pi * pos), np.cos(2 * np.pi * pos)
        d["is_weekend"] = 0
        frames[f"c{i:02d}"] = (cid, d[[TARGET, "gpu_count", "gpu_mem_util_w", "hod_sin", "hod_cos", "is_weekend"]])
    pd.DataFrame(excluded).to_csv(os.path.join(RES, "excluded_clusters.csv"), index=False)
    print(f"cluster lengkap {len(rank)}, dikeluarkan {len(excluded)} (konstan / >50% nol), dipakai {len(frames)}", flush=True)
    return frames, rank


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6, 12])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--top-dl", type=int, default=5, help="GRU hanya pada N cluster terbesar")
    ap.add_argument("--max-clusters", type=int, default=0, help="0 = semua cluster lengkap")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true", help="lewati cluster yang sudah ada di metrics_runs_<tag>.csv")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.patience, a.max_clusters, a.top_dl = [0], 3, 2, 2, 1
    import torch
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    tag = a.tag or ("quick" if a.quick else "full")
    frames, rank = cluster_frames()
    keys = list(frames)[: a.max_clusters or None]
    t_all, res, wts, log = time.time(), [], [], []
    runs_path = os.path.join(RES, f"metrics_runs_{tag}.csv")
    if a.resume and os.path.exists(runs_path):
        prev = pd.read_csv(runs_path)
        done = set(prev.service.unique())
        res.append(prev)
        if os.path.exists(os.path.join(RES, f"weights_{tag}.csv")):
            wts.append(pd.read_csv(os.path.join(RES, f"weights_{tag}.csv")))
        keys = [k for k in keys if k not in done]
        print(f"resume: {len(done)} cluster sudah ada, sisa {len(keys)}", flush=True)
    for k, key in enumerate(keys):
        cid, df = frames[key]
        base_models = ["ar_ridge", "hgb", "gru"] if k < a.top_dl else ["ar_ridge", "hgb"]
        t0 = time.time()
        r, w = meta.run_df(df, key, a.horizons, a.seeds, a.epochs, a.patience, target=TARGET,
                           base_models=base_models, res_dir=RES)
        r["cluster_id"], w["cluster_id"] = cid, cid
        r["gpu_count_mean"] = float(rank[cid]); r["has_gru"] = "gru" in base_models
        res.append(r); wts.append(w)
        log.append({"key": key, "cluster_id": cid, "gpu_count_mean": float(rank[cid]), "has_gru": "gru" in base_models,
                    "seconds": round(time.time() - t0)})
        pd.concat(res).to_csv(runs_path, index=False)                                        # simpan bertahap
        pd.concat(wts).to_csv(os.path.join(RES, f"weights_{tag}.csv"), index=False)
        print(f"### {key} selesai ({time.time() - t0:.0f}s), total {time.time() - t_all:.0f}s", flush=True)
    res, wts = pd.concat(res, ignore_index=True), pd.concat(wts, ignore_index=True)
    res.to_csv(os.path.join(RES, f"metrics_runs_{tag}.csv"), index=False)
    wts.to_csv(os.path.join(RES, f"weights_{tag}.csv"), index=False)
    summ = (res.groupby(["service", "cluster_id", "h", "model"]).agg(MAE=("MAE", "mean"), sMAPE=("sMAPE", "mean"))
               .reset_index())
    summ.to_csv(os.path.join(RES, f"metrics_summary_{tag}.csv"), index=False)
    with open(os.path.join(RES, f"config_{tag}.json"), "w") as fh:
        json.dump({**vars(a), "clusters": log, "total_seconds": round(time.time() - t_all)}, fh, indent=2)
    pd.set_option("display.width", 220)
    order = ["naive", "seasonal_naive", "best_single", "nnls", "meta_ridge_regime", "meta_hgb_noregime", "meta_select", "oracle_single"]
    piv = summ.pivot_table(index="h", columns="model", values="sMAPE")
    print("\n== sMAPE rata-rata lintas cluster ==\n", piv[[c for c in order if c in piv.columns]].round(2).to_string())
    print(f"\nTotal waktu: {time.time() - t_all:.0f} s untuk {len(keys)} cluster")
