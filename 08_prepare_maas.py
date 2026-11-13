"""
Tahap 1b: agregasi Alibaba cluster-trace-v2026-maas (ATC 2026) menjadi deret waktu.
Sumber (MAAS_DIR, lihat config.py):
  instance_week_5min/part_*.parquet : 1 minggu, per instance per 5 menit
  instance_daily/part_*.parquet     : 6 bulan, per instance per hari
  app_metrics.csv                   : 1 minggu, per aplikasi per jam (format panjang metric/value)
Output (series/):
  maas_week5min_by_cluster.csv  : per cluster x 5 menit: n_instances, gpu_count, sm_util_w, gpu_mem_util_w, cpu_usage, mem_usage
  maas_week5min_all.csv         : agregat seluruh platform per 5 menit
  maas_daily_by_cluster.csv     : per cluster x hari: n_instances, gpu_hours, gpu_count, sm_util_w, gpu_mem_util_w (+ online/offline)
  maas_daily_all.csv            : agregat seluruh platform per hari
  maas_app_hourly.csv           : per aplikasi x jam: qps_norm, input_tokens, output_tokens, gpu_count, ... (format lebar)
Catatan: sm_util/gpu_mem_util dirata-rata berbobot gpu_count (sesuai definisi skema: dinormalisasi per GPU equivalent).
Pemakaian: python 08_prepare_maas.py [--skip-daily] [--skip-week]
"""
import argparse, glob, os, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from config import MAAS_DIR

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "series")
os.makedirs(OUT, exist_ok=True)


def wmean_agg(df, keys, util_cols, w="gpu_count"):
    """Agregasi: jumlah untuk kolom sumber daya, rata-rata berbobot w untuk kolom utilisasi."""
    d = df.copy()
    d[w] = d[w].fillna(0).astype(float)
    for c in util_cols:
        d[f"_{c}_w"] = d[c].astype(float).fillna(0) * d[w]
    agg = {"n_instances": ("instance_id", "nunique"), w: (w, "sum")}
    for c in ["cpu_usage", "mem_usage", "gpu_hours"]:
        if c in d.columns:
            agg[c] = (c, "sum")
    for c in util_cols:
        agg[f"_{c}_w"] = (f"_{c}_w", "sum")
    g = d.groupby(keys).agg(**agg)
    for c in util_cols:
        g[f"{c}_w"] = g[f"_{c}_w"] / g[w].replace(0, np.nan)
        g = g.drop(columns=f"_{c}_w")
    return g


def merge_partials(parts, keys, util_cols, w="gpu_count"):
    """Gabungkan agregat parsial antar-shard (jumlah dijumlahkan, rata-rata berbobot digabung ulang)."""
    d = pd.concat(parts)
    for c in util_cols:
        d[f"_{c}_w"] = d[f"{c}_w"].fillna(0) * d[w]
    sums = d.groupby(level=list(range(len(keys)))).sum(numeric_only=True)
    for c in util_cols:
        sums[f"{c}_w"] = sums[f"_{c}_w"] / sums[w].replace(0, np.nan)
        sums = sums.drop(columns=f"_{c}_w")
    return sums


def prepare_week():
    files = sorted(glob.glob(os.path.join(MAAS_DIR, "instance_week_5min", "part_*.parquet")))
    assert files, "instance_week_5min/part_*.parquet tidak ditemukan"
    util = ["sm_util", "gpu_mem_util"]
    parts_c, parts_all, t0 = [], [], time.time()
    for i, f in enumerate(files):
        df = pq.read_table(f, columns=["instance_id", "time_s", "cluster_id", "gpu_count",
                                       "sm_util", "gpu_mem_util", "cpu_usage", "mem_usage"]).to_pandas()
        parts_c.append(wmean_agg(df, ["cluster_id", "time_s"], util))
        parts_all.append(wmean_agg(df, ["time_s"], util))
        print(f"  week shard {i + 1}/{len(files)}: {len(df):,} baris ({time.time() - t0:.0f}s)", flush=True)
    # n_instances per shard tidak bisa dijumlahkan jika instance terbelah antar-shard; hitung ulang kasar = jumlah
    byc = merge_partials(parts_c, ["cluster_id", "time_s"], util).reset_index().sort_values(["cluster_id", "time_s"])
    alls = merge_partials(parts_all, ["time_s"], util).reset_index().sort_values("time_s")
    byc.to_csv(os.path.join(OUT, "maas_week5min_by_cluster.csv"), index=False)
    alls.to_csv(os.path.join(OUT, "maas_week5min_all.csv"), index=False)
    print(f"week: {byc.cluster_id.nunique()} cluster, {alls.shape[0]} interval 5 menit "
          f"({alls.time_s.min()}..{alls.time_s.max()} s)")


def prepare_daily():
    files = sorted(glob.glob(os.path.join(MAAS_DIR, "instance_daily", "part_*.parquet")))
    assert files, "instance_daily/part_*.parquet tidak ditemukan"
    util = ["sm_util", "gpu_mem_util"]
    parts_c, parts_all, parts_wl, t0 = [], [], [], time.time()
    for i, f in enumerate(files):
        df = pq.read_table(f, columns=["day", "instance_id", "cluster_id", "workload", "gpu_count",
                                       "gpu_hours", "sm_util", "gpu_mem_util", "cpu_usage", "mem_usage"]).to_pandas()
        parts_c.append(wmean_agg(df, ["cluster_id", "day"], util))
        parts_all.append(wmean_agg(df, ["day"], util))
        parts_wl.append(wmean_agg(df, ["workload", "day"], util))
        print(f"  daily shard {i + 1}/{len(files)}: {len(df):,} baris ({time.time() - t0:.0f}s)", flush=True)
    byc = merge_partials(parts_c, ["cluster_id", "day"], util).reset_index().sort_values(["cluster_id", "day"])
    alls = merge_partials(parts_all, ["day"], util).reset_index().sort_values("day")
    bywl = merge_partials(parts_wl, ["workload", "day"], util).reset_index().sort_values(["workload", "day"])
    byc.to_csv(os.path.join(OUT, "maas_daily_by_cluster.csv"), index=False)
    alls.to_csv(os.path.join(OUT, "maas_daily_all.csv"), index=False)
    bywl.to_csv(os.path.join(OUT, "maas_daily_by_workload.csv"), index=False)
    print(f"daily: {byc.cluster_id.nunique()} cluster, {alls.shape[0]} hari")


def prepare_app():
    path = os.path.join(MAAS_DIR, "app_metrics.csv")
    df = pd.read_csv(path)
    wide = (df.pivot_table(index=["app_id", "time_s"], columns="metric", values="value", aggfunc="mean")
              .reset_index().sort_values(["app_id", "time_s"]))
    wide.to_csv(os.path.join(OUT, "maas_app_hourly.csv"), index=False)
    print(f"app: {wide.app_id.nunique()} aplikasi, {wide.time_s.nunique()} jam, metrik: {list(wide.columns[2:])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-week", action="store_true")
    ap.add_argument("--skip-daily", action="store_true")
    ap.add_argument("--skip-app", action="store_true")
    a = ap.parse_args()
    if not a.skip_app:
        prepare_app()
    if not a.skip_week:
        prepare_week()
    if not a.skip_daily:
        prepare_daily()
