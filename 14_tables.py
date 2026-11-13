"""
Tabel naskah dari file hasil (tidak diketik ulang).
  Tabel 2 : statistik deskriptif deret + dua ukuran regularitas (acf lag 24 jam; rasio varians profil harian)
  Tabel S : regularitas vs skill per cluster Alibaba (jika results/maas ada), Spearman untuk kedua ukuran
Keluaran: results/tables/table2_datasets.csv|.md, results/tables/regularity_vs_skill.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
SER, OUT = os.path.join(BASE, "series"), os.path.join(BASE, "results", "tables")
os.makedirs(OUT, exist_ok=True)
S = 288


def acf_lag(y, lag=S):
    y = np.asarray(y, float)
    if y.std() == 0:
        return np.nan
    y0 = y - y.mean()
    return float(np.dot(y0[lag:], y0[:-lag]) / np.dot(y0, y0))


def profile_ratio(y, period=S):
    """Rasio varians profil harian: Var(rata-rata per slot) / Var(y). 1 = seluruh variasi dijelaskan pola harian."""
    y = np.asarray(y, float)
    if y.std() == 0:
        return np.nan
    n = len(y) // period * period
    prof = y[:n].reshape(-1, period).mean(0)
    return float(prof.var() / y.var())


def describe(name, y, unit, step):
    y = np.asarray(y, float)
    return {"deret": name, "n": len(y), "langkah": step, "satuan": unit, "rata-rata": y.mean(), "sd": y.std(),
            "min": y.min(), "maks": y.max(), "acf_24h": acf_lag(y), "rasio_profil_harian": profile_ratio(y)}


rows = []
for svc in ["code", "conv"]:
    f = pd.read_csv(os.path.join(SER, f"azure_llm_2024_{svc}_features_5min.csv"))
    rows.append(describe(f"Azure {svc}: total_tokens", f["total_tokens"], "token/5 mnt", "5 mnt"))
    rows.append(describe(f"Azure {svc}: requests", f["requests"], "request/5 mnt", "5 mnt"))
w = pd.read_csv(os.path.join(SER, "maas_week5min_by_cluster.csv"))
g = w.groupby("cluster_id"); full = g.size()[g.size() == 2016].index
w = w[w.cluster_id.isin(full)].sort_values(["cluster_id", "time_s"])
reg = []
for cid, d in w.groupby("cluster_id"):
    y = d.sm_util_w.values
    if y.std() < 1e-9 or (y == 0).mean() > 0.5:
        continue
    reg.append({"cluster_id": cid, "gpu_count_mean": d.gpu_count.mean(), "acf_24h": acf_lag(y), "rasio_profil_harian": profile_ratio(y),
                "rata-rata": y.mean(), "sd": y.std()})
reg = pd.DataFrame(reg)
rows.append({"deret": f"Alibaba: sm_util_w, {len(reg)} cluster (median; min–maks)", "n": 2016, "langkah": "5 mnt", "satuan": "%",
             "rata-rata": reg["rata-rata"].median(), "sd": reg["sd"].median(), "min": np.nan, "maks": np.nan,
             "acf_24h": reg.acf_24h.median(), "rasio_profil_harian": reg.rasio_profil_harian.median()})
a = pd.read_csv(os.path.join(SER, "maas_week5min_all.csv")).sort_values("time_s")
rows.append(describe("Alibaba: sm_util_w, seluruh platform", a.sm_util_w, "%", "5 mnt"))
rows.append(describe("Alibaba: gpu_mem_util_w, seluruh platform", a.gpu_mem_util_w, "%", "5 mnt"))
for name, fn, col, unit in [("DCT Alibaba 2018: cpu", "dct_alibaba2018_300s.csv", "cpu_util_percent", "%"),
                            ("DCT Google 2019: cpu", "dct_google2019_300s.csv", "cpu_util_percent", "%"),
                            ("DCT Azure v2: cpu", "dct_azure_v2_300s.csv", "cpu_usage", "core")]:
    d = pd.read_csv(os.path.join(SER, fn)); rows.append(describe(name, d[col], unit, "300 dtk"))
t2 = pd.DataFrame(rows)
t2.to_csv(os.path.join(OUT, "table2_datasets.csv"), index=False)
with open(os.path.join(OUT, "table2_datasets.md"), "w") as fh:
    fh.write(t2.round(3).to_markdown(index=False))
pd.set_option("display.width", 220)
print(t2.round(3).to_string(index=False))
print(f"\nAlibaba per cluster: acf_24h median {reg.acf_24h.median():.3f} (>0,5: {(reg.acf_24h > 0.5).sum()}), "
      f"rasio profil median {reg.rasio_profil_harian.median():.3f} (>0,5: {(reg.rasio_profil_harian > 0.5).sum()})")

# regularitas vs skill (jika hasil tahap 6 ada)
runs_path = os.path.join(BASE, "results", "maas", "metrics_runs_full.csv")
if os.path.exists(runs_path):
    r = pd.read_csv(runs_path)
    per = r.groupby(["cluster_id", "h", "model"]).MAE.mean().reset_index()
    out = []
    for h in sorted(per.h.unique()):
        d = per[per.h == h].pivot(index="cluster_id", columns="model", values="MAE")
        skill = 1 - d.div(d["naive"], axis=0)
        best = skill[["nnls", "meta_select", "best_single", "avg"]].max(axis=1)
        m = reg.set_index("cluster_id").join(best.rename("best_skill")).dropna()
        for meas in ["acf_24h", "rasio_profil_harian"]:
            rho, p = stats.spearmanr(m[meas], m.best_skill)
            out.append({"h": h, "ukuran": meas, "spearman_rho": rho, "p": p, "n": len(m)})
        if h == 12:
            m.to_csv(os.path.join(OUT, "regularity_vs_skill_h12.csv"))
    out = pd.DataFrame(out); out.to_csv(os.path.join(OUT, "regularity_vs_skill.csv"), index=False)
    print("\nSpearman regularitas vs skill terbaik (Alibaba):\n", out.round(3).to_string(index=False))
