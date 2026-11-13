"""Ringkasan sensitivitas simulasi autoscaling (tahap 7) dari sim_summary_<tag>*.csv: pengurangan cost_index relatif
terhadap reaktif (naive) pada margin 10%, per layanan dan horizon, untuk tiap varian (penalti, N_REF).
Keluaran: results/autoscale/sensitivity_<tag>.csv|.md"""
import os, sys
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results", "autoscale")
tag = sys.argv[1] if len(sys.argv) > 1 else "full5"
variants = [("dasar (N_REF=100, penalti=5)", f"sim_summary_{tag}.csv"), ("penalti=2", f"sim_summary_{tag}_n100_p2.csv"),
            ("penalti=10", f"sim_summary_{tag}_n100_p10.csv"), ("N_REF=50", f"sim_summary_{tag}_n50_p5.csv"),
            ("N_REF=200", f"sim_summary_{tag}_n200_p5.csv")]
rows = []
for name, fn in variants:
    p = os.path.join(RES, fn)
    if not os.path.exists(p):
        continue
    s = pd.read_csv(p); s = s[s.margin == 0.1]
    for (svc, h), d in s.groupby(["service", "h"]):
        d = d.set_index("policy")
        base = d.loc["naive", "cost_index"]
        row = {"varian": name, "service": svc, "h": h, "cost_naive": round(base, 1)}
        for pol in ["best_single", "nnls", "meta_ridge_regime", "meta_select", "oracle_single", "y_true"]:
            if pol in d.index:
                row[f"d_{pol}_%"] = round(100 * (d.loc[pol, "cost_index"] / base - 1), 1)
        row["underprov_naive_%"] = round(100 * d.loc["naive", "underprov_frac"], 1)
        row["underprov_meta_select_%"] = round(100 * d.loc["meta_select", "underprov_frac"], 1)
        row["waste_naive"] = round(d.loc["naive", "waste_gpu_hours"], 1)
        row["waste_meta_select"] = round(d.loc["meta_select", "waste_gpu_hours"], 1)
        rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, f"sensitivity_{tag}.csv"), index=False)
with open(os.path.join(RES, f"sensitivity_{tag}.md"), "w") as fh:
    fh.write(out.to_markdown(index=False))
pd.set_option("display.width", 250)
print(out.to_string(index=False))
