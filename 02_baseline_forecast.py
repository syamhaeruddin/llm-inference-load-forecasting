"""
Tahap 2: baseline forecasting satu langkah ke depan (one-step-ahead), horizon 5 menit.
Deret yang diuji:
  - Azure LLM 2024 code & conv : requests per 5 menit (musiman harian = 288 langkah)
  - DCT Alibaba 2018           : cpu_util_percent per 300 detik
  - DCT Google 2019            : cpu_util_percent per 300 detik
  - DCT Azure v2               : cpu_usage per 300 detik
Pembagian data kronologis: 80% awal = latih, 20% akhir = uji (tidak diacak).
Model:
  naive            : y[t-1]
  seasonal_naive   : y[t-288]
  holt_winters     : ETS aditif (level+trend+musiman 288), parameter dioptimasi pada data latih
                     (implementasi numpy sendiri; versi statsmodels terlalu lambat untuk s=288)
  ar_ridge         : regresi linear Ridge pada 288 lag + fitur jam (setara AR(288))
  gbr_lags         : GradientBoosting pada 288 lag + fitur jam
  mlp_lags         : MLP (128-64) pada 288 lag + fitur jam
Metrik: MAE, RMSE, sMAPE. MAPE tidak dipakai karena ada nilai mendekati nol.
Catatan: SARIMA(1,1,1)(1,0,1,288) dicoba lebih dulu tetapi tidak selesai dalam 10 menit
untuk satu deret, sehingga dikeluarkan dari baseline tahap ini.
"""
import os, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
SER = os.path.join(BASE, "series")
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)
SEASON = 288          # 24 jam / 5 menit
N_LAGS = 288
SEED = 42

SERIES = {
    "azure_llm_2024_code_requests": ("azure_llm_2024_code_5min.csv", "requests"),
    "azure_llm_2024_conv_requests": ("azure_llm_2024_conv_5min.csv", "requests"),
    "dct_alibaba2018_cpu":          ("dct_alibaba2018_300s.csv", "cpu_util_percent"),
    "dct_google2019_cpu":           ("dct_google2019_300s.csv", "cpu_util_percent"),
    "dct_azure_v2_cpu":             ("dct_azure_v2_300s.csv", "cpu_usage"),
}


def smape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = (np.abs(y) + np.abs(yhat))
    mask = denom != 0
    return 100 * np.mean(2 * np.abs(y[mask] - yhat[mask]) / denom[mask])


def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {"MAE": float(np.mean(np.abs(y - yhat))),
            "RMSE": float(np.sqrt(np.mean((y - yhat) ** 2))),
            "sMAPE": float(smape(y, yhat))}


def holt_winters_additive(y, alpha, beta, gamma, m):
    """Kembalikan prediksi satu-langkah y_hat[t] untuk t = m..n-1 (inisialisasi dari musim pertama)."""
    y = np.asarray(y, float)
    n = len(y)
    level = y[:m].mean()
    trend = (y[m:2 * m].mean() - y[:m].mean()) / m if n >= 2 * m else 0.0
    season = y[:m] - level
    yhat = np.full(n, np.nan)
    for t in range(m, n):
        s_idx = t % m
        yhat[t] = level + trend + season[s_idx]
        new_level = alpha * (y[t] - season[s_idx]) + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        season[s_idx] = gamma * (y[t] - new_level) + (1 - gamma) * season[s_idx]
        level = new_level
    return yhat


def fit_holt_winters(y_tr, m):
    def sse(p):
        a, b, g = np.clip(p, 1e-4, 0.9999)
        f = holt_winters_additive(y_tr, a, b, g, m)
        return np.nanmean((y_tr[m:] - f[m:]) ** 2)
    best = None
    for x0 in [(0.3, 0.01, 0.3), (0.6, 0.05, 0.1), (0.1, 0.001, 0.5)]:
        r = minimize(sse, x0, method="Nelder-Mead", options={"maxiter": 150, "xatol": 1e-3, "fatol": 1e-3})
        if best is None or r.fun < best.fun:
            best = r
    return np.clip(best.x, 1e-4, 0.9999)


def make_lag_matrix(values, n_lags):
    """X[t] = [y[t-1..t-n_lags], jam-sin, jam-cos]; target y[t]."""
    v = np.asarray(values, float)
    n = len(v)
    idx = np.arange(n_lags, n)
    lags = np.stack([v[t - n_lags:t][::-1] for t in idx])
    pos = (idx % SEASON) / SEASON
    X = np.column_stack([lags, np.sin(2 * np.pi * pos), np.cos(2 * np.pi * pos)])
    return X, v[idx]


def run_series(name, fname, col):
    df = pd.read_csv(os.path.join(SER, fname))
    y = df[col].astype(float).values
    n = len(y)
    split = int(n * 0.8)
    y_tr, y_te = y[:split], y[split:]
    print(f"\n=== {name}: n={n}, latih={split}, uji={n - split}", flush=True)
    preds, timing, params = {}, {}, {}

    preds["naive"] = y[split - 1:n - 1]
    preds["seasonal_naive"] = y[split - SEASON:n - SEASON]

    t0 = time.time()
    a, b, g = fit_holt_winters(y_tr, SEASON)
    hw = holt_winters_additive(y, a, b, g, SEASON)       # parameter tetap, state terus diperbarui
    preds["holt_winters"] = hw[split:]
    timing["holt_winters"] = time.time() - t0
    params["holt_winters"] = f"alpha={a:.3f}, beta={b:.4f}, gamma={g:.3f}"

    X, Y = make_lag_matrix(y, N_LAGS)
    off = N_LAGS
    Xtr, Ytr = X[:split - off], Y[:split - off]
    Xte, Yte = X[split - off:], Y[split - off:]
    assert len(Yte) == len(y_te) and np.allclose(Yte, y_te)
    sx = StandardScaler().fit(Xtr)
    sy = StandardScaler().fit(Ytr.reshape(-1, 1))
    Xtr_s, Xte_s = sx.transform(Xtr), sx.transform(Xte)
    Ytr_s = sy.transform(Ytr.reshape(-1, 1)).ravel()
    inv = lambda p: sy.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()

    t0 = time.time()
    ridge = Ridge(alpha=1.0).fit(Xtr_s, Ytr_s)
    preds["ar_ridge"] = inv(ridge.predict(Xte_s))
    timing["ar_ridge"] = time.time() - t0

    t0 = time.time()
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=SEED).fit(Xtr, Ytr)
    preds["gbr_lags"] = gbr.predict(Xte)
    timing["gbr_lags"] = time.time() - t0

    t0 = time.time()
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True,
                       random_state=SEED).fit(Xtr_s, Ytr_s)
    preds["mlp_lags"] = inv(mlp.predict(Xte_s))
    timing["mlp_lags"] = time.time() - t0

    rows = []
    for m, p in preds.items():
        mt = metrics(y_te, p)
        mt.update({"series": name, "model": m, "n_test": len(y_te),
                   "fit_seconds": round(timing.get(m, 0.0), 1), "params": params.get(m, "")})
        rows.append(mt)
        print(f"  {m:15s} MAE={mt['MAE']:12.2f} RMSE={mt['RMSE']:12.2f} sMAPE={mt['sMAPE']:6.2f}%  "
              f"({mt['fit_seconds']}s)", flush=True)

    out = pd.DataFrame({"y_true": y_te, **{k: np.asarray(v) for k, v in preds.items()}})
    out.to_csv(os.path.join(RES, f"pred_{name}.csv"), index=False)
    return rows


all_rows = []
for name, (fname, col) in SERIES.items():
    all_rows += run_series(name, fname, col)
res = pd.DataFrame(all_rows)[["series", "model", "n_test", "MAE", "RMSE", "sMAPE", "fit_seconds", "params"]]
res.to_csv(os.path.join(RES, "metrics_summary.csv"), index=False)
print("\n", res.drop(columns="params").to_string(index=False))
