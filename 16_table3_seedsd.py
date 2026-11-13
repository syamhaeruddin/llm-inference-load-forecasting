"""Tabel 3 naskah: sMAPE model tunggal (rata-rata dua jendela uji) +- simpangan baku ANTAR-SEED
(tiap seed dirata-ratakan dulu lintas jendela, lalu sd lintas seed). Sumber: results/tokens/metrics_runs_<tag>.csv.
Keluaran: results/tokens/table3_<tag>_seedsd.csv|.md"""
import os, sys
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results", "tokens")
tag = sys.argv[1] if len(sys.argv) > 1 else "w144"
r = pd.read_csv(os.path.join(RES, f"metrics_runs_{tag}.csv"))
g = r.groupby(["service", "h", "model", "seed"]).sMAPE.mean().reset_index()
t = g.groupby(["service", "h", "model"]).sMAPE.agg(["mean", "std"]).reset_index()
t["cell"] = t.apply(lambda x: f"{x['mean']:.2f}" + (f" ± {x['std']:.2f}" if pd.notna(x["std"]) and x["std"] > 0 else ""), axis=1)
p = t.pivot(index=["service", "h"], columns="model", values="cell")
cols = [c for c in ["naive", "seasonal_naive", "ar_ridge", "hgb", "lstm", "gru", "tcn", "transformer"] if c in p.columns]
p = p[cols]
p.to_csv(os.path.join(RES, f"table3_{tag}_seedsd.csv"))
with open(os.path.join(RES, f"table3_{tag}_seedsd.md"), "w", encoding="utf-8") as fh:
    fh.write(p.to_markdown())
pd.set_option("display.width", 250)
print(p.to_string())
# waktu latih rata-rata per fit (detik)
f = r.groupby("model").fit_s.mean().round(1)
print("\nwaktu latih rata-rata per fit (s):\n", f.to_string())
