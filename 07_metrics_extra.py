"""Metrik pendamping dari prediksi tersimpan: R^2 dan akurasi = 100 - sMAPE.
Pemakaian: python 07_metrics_extra.py [tag]   (default: full)
Output: results/tokens/table_r2_<tag>.csv, table_accuracy_<tag>.csv
"""
import glob, os, re, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results", "tokens")
TAG = sys.argv[1] if len(sys.argv) > 1 else "full"
ORDER = ["naive", "seasonal_naive", "ar_ridge", "hgb", "lstm", "gru", "tcn", "transformer"]

rows = []
for f in sorted(glob.glob(os.path.join(RES, "pred_*_h*_fold*.csv"))):
    svc, h, fold = re.search(r"pred_(\w+?)_h(\d+)_fold(\d+)", os.path.basename(f)).groups()
    p = pd.read_csv(f)
    y = p["y_true"].values
    ss_tot = np.sum((y - y.mean()) ** 2)
    for c in [c for c in p.columns if c not in ("timestamp", "y_true")]:
        model = c.rsplit("_s", 1)[0]
        q = p[c].values
        d = np.abs(y) + np.abs(q)
        smape = 100 * np.mean(2 * np.abs(y - q) / d)
        r2 = 1 - np.sum((y - q) ** 2) / ss_tot
        rows.append(dict(service=svc, h=int(h), fold=int(fold), model=model,
                         accuracy=100 - smape, r2=r2))

df = pd.DataFrame(rows)
g = df.groupby(["service", "h", "model"]).mean(numeric_only=True).reset_index()
acc = g.pivot(index=["service", "h"], columns="model", values="accuracy")[ORDER].round(1)
r2 = g.pivot(index=["service", "h"], columns="model", values="r2")[ORDER].round(3)
acc.to_csv(os.path.join(RES, f"table_accuracy_{TAG}.csv"))
r2.to_csv(os.path.join(RES, f"table_r2_{TAG}.csv"))
pd.set_option("display.width", 200)
print("Akurasi = 100 - sMAPE (%):\n", acc.to_string())
print("\nR^2 (1 = sempurna; 0 = setara rata-rata data uji; negatif = lebih buruk dari rata-rata):\n", r2.to_string())
