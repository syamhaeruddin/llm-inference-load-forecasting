"""
Tahap 4: dekomposisi kausal + fitur rezim + target selisih, dengan ablasi dan uji kebocoran.
Deret: Azure LLM 2024 code & conv, Y = total_tokens per 5 menit (dari 04_feature_table.py).

Varian representasi (ablasi RQ2 dan H2):
  raw        : model memprediksi Y(t+h) langsung dari 288 lag Y                       [ablasi (a)]
  diff       : model memprediksi Y(t+h) - Y(t-1); prediksi akhir = Y(t-1) + d_hat     [ablasi (e)]
  causal     : dekomposisi klasik KAUSAL pada window masa lalu: tren = rata-rata 24 jam terakhir,
               musiman = profil slot-harian dari W hari terakhir, residu = Y - tren - musiman.
               Model memprediksi residu(t+h); prediksi akhir = tren(t-1) + musiman[slot(t+h)] + r_hat   [ablasi (b)]
  noncausal  : dekomposisi yang SAMA tetapi dihitung dari SELURUH deret (termasuk data uji), seperti
               lazim di literatur; tren(t+h) dan musiman(t+h) yang "bocor" dipakai saat rekonstruksi  [ablasi (c)]
Fitur rezim (dipakai semua varian kecuali naive): vol_1h, level_ratio_1h_24h, dev_profile, hod_sin/cos, is_weekend.
Model: naive, seasonal_naive, ar_ridge, hgb, gru (seed ganda). Protokol fold/horizon/metrik = 05_experiment_tokens.py.
Uji kebocoran: (1) assert semua fitur sampel t hanya memakai indeks <= t-1 (dicek dengan mengacak Y[t..] dan
memastikan fitur tak berubah); (2) uji shuffle-target: dengan Y diacak, tidak ada model yang boleh jauh lebih baik
dari naive.
Pemakaian: python 09_decompose.py [--quick] [--services code conv] [--horizons 1 3 6 12]
           [--variants raw diff causal noncausal] [--models naive seasonal_naive ar_ridge hgb gru] [--tag t]
"""
import argparse, importlib, json, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
SER = os.path.join(BASE, "series")
RES = os.path.join(BASE, "results", "decomp")
os.makedirs(RES, exist_ok=True)
exp = importlib.import_module("05_experiment_tokens")          # fungsi bersama
metrics, diebold_mariano, rolling_folds, train_dl = exp.metrics, exp.diebold_mariano, exp.rolling_folds, exp.train_dl

SEASON, TARGET = 288, "total_tokens"
WINDOW = 144      # 12 jam lag (bukan 288 seperti di 05): komponen kausal baru ada setelah 2 hari, sehingga
                  # window 288 akan memangkas data latih menjadi ~1,5 hari. Sama untuk semua varian (ablasi adil).
W_DAYS = 1                                   # siklus terakhir saja: profil 3 hari tercampur akhir pekan (encer)
START = 2 * SEASON                           # indeks sampel pertama: residu kausal ada sejak hari ke-2, musiman
                                             # slot target (tgt-288) ada sejak 288, lag 144 terpenuhi
VARIANTS = ["raw", "diff", "causal", "noncausal"]
MODELS = ["naive", "seasonal_naive", "ar_ridge", "hgb", "gru"]


# ----------------------------------------------------------------------------- dekomposisi
def decompose_causal(y, season=SEASON, w_days=W_DAYS):
    """Komponen pada indeks i hanya dari y[:i+1]. tren[i] = mean(y[i-season+1..i]);
    musiman[i] = mean(y[j]-tren[j]) untuk j = i-season, i-2*season, ... (w_days siklus terakhir, slot sama);
    residu = y - tren - musiman. Profil musiman untuk prediksi masa depan: profile[i, slot] = musiman slot tsb
    yang diketahui sampai i (dipakai untuk slot(t+h))."""
    n = len(y)
    # tren kausal: rata-rata 24 jam terakhir; pada 24 jam pertama rata-rata ekspansif (tetap hanya masa lalu)
    trend = pd.Series(y).rolling(season, min_periods=1).mean().values
    detr = y - trend
    seasonal = np.full(n, np.nan)
    for i in range(season, n):
        vals = [detr[i - k * season] for k in range(1, w_days + 1) if i - k * season >= 0]
        seasonal[i] = np.mean(vals)
    resid = y - trend - seasonal
    return trend, seasonal, resid


def decompose_noncausal(y, season=SEASON):
    """Versi BOCOR: tren = moving average terpusat (memakai masa depan), musiman = profil slot dari seluruh deret."""
    n = len(y)
    half = season // 2
    trend = pd.Series(y).rolling(season, center=True, min_periods=half).mean().values
    detr = y - trend
    profile = np.array([np.nanmean(detr[s::season]) for s in range(season)])
    seasonal = profile[np.arange(n) % season]
    resid = y - trend - seasonal
    return trend, seasonal, resid, profile


# ----------------------------------------------------------------------------- fitur
def regime_features(df):
    y = df[TARGET].values.astype(float)
    s = pd.Series(y)
    vol_1h = (s.rolling(12).std() / s.rolling(12).mean().replace(0, np.nan)).values
    level_ratio = (s.rolling(12).mean() / s.rolling(288).mean().replace(0, np.nan)).values
    seas_naive_prev = np.r_[np.full(SEASON, np.nan), y[:-SEASON]]
    dev_profile = (y - seas_naive_prev) / np.where(seas_naive_prev == 0, np.nan, seas_naive_prev)
    return np.column_stack([vol_1h, level_ratio, dev_profile])


def build(df, h, variant):
    """Kembalikan dict: idx (indeks sampel t), lags (window dari deret yang dimodelkan), extra (fitur t-1 + kalender
    target), Y_true (nilai asli target), Y_model (target yang dilatih), recon(pred_model) -> prediksi skala asli."""
    y = df[TARGET].values.astype(float)
    n = len(y)
    cal = df[["hod_sin", "hod_cos", "is_weekend"]].values.astype(float)
    reg = regime_features(df)
    if variant in ("raw", "diff"):
        series, trend, seasonal = y, None, None
    elif variant == "causal":
        trend, seasonal, resid = decompose_causal(y)
        series = resid
    elif variant == "noncausal":
        trend, seasonal, resid, profile = decompose_noncausal(y)
        series = resid
    idx = np.arange(START, n - h + 1)
    tgt = idx + h - 1
    lags = np.stack([series[i - WINDOW:i] for i in idx])
    extra = np.column_stack([cal[tgt], reg[idx - 1]])
    Y_true = y[tgt]
    last = y[idx - 1]
    if variant == "raw":
        Y_model, recon = Y_true, (lambda p, sl: p)
    elif variant == "diff":
        Y_model, recon = Y_true - last, (lambda p, sl: last[sl] + p)
    elif variant == "causal":
        # tren terakhir diketahui + musiman slot target dari W_DAYS siklus terakhir yang sudah diketahui
        # (indeks tgt - k*288 <= idx-1 karena h < 288), sehingga tetap kausal
        detr = y - trend
        base = trend[idx - 1] + np.mean([detr[tgt - k * SEASON] for k in range(1, W_DAYS + 1)], axis=0)
        Y_model, recon = Y_true - base, (lambda p, sl: base[sl] + p)
    else:  # noncausal: komponen pada t+h sendiri (bocor)
        base = trend[tgt] + seasonal[tgt]
        Y_model, recon = Y_true - base, (lambda p, sl: base[sl] + p)
    return dict(idx=idx, tgt=tgt, lags=lags, extra=extra, Y_true=Y_true, Y_model=Y_model, recon=recon,
                last=last, seas=y[tgt - SEASON], y=y)


# ----------------------------------------------------------------------------- uji kebocoran
def leakage_unit_test(df, h=3):
    """Fitur dan target-model pada sampel t tidak boleh berubah jika Y pada indeks >= t diacak (kecuali Y_true/noncausal)."""
    rng = np.random.default_rng(0)
    out = {}
    for variant in VARIANTS:
        a = build(df, h, variant)
        k = len(a["idx"]) // 2
        t = a["idx"][k]
        df2 = df.copy()
        future = df2.index[t:]
        df2.loc[future, TARGET] = rng.permutation(df2.loc[future, TARGET].values)
        b = build(df2, h, variant)
        same = (np.allclose(a["lags"][k], b["lags"][k], equal_nan=True)
                and np.allclose(a["extra"][k], b["extra"][k], equal_nan=True))
        same_base = np.isclose(a["Y_true"][k] - a["Y_model"][k], b["Y_true"][k] - b["Y_model"][k], equal_nan=True)
        out[variant] = {"features_unchanged": bool(same), "baseline_unchanged": bool(same_base)}
    return out


# ----------------------------------------------------------------------------- eksperimen
def run(service, horizons, variants, models, n_folds, seeds, epochs, patience, shuffle=False):
    df = pd.read_csv(os.path.join(SER, f"azure_llm_2024_{service}_features_5min.csv"), index_col=0, parse_dates=True)
    if shuffle:
        df[TARGET] = np.random.default_rng(1).permutation(df[TARGET].values)
    rows = []
    for h in horizons:
        for variant in variants:
            a = build(df, h, variant)
            n = len(a["Y_true"])
            for f, (tr0, va0, te0, te1) in enumerate(rolling_folds(n, n_folds)):
                sl = slice(te0, te1)
                y_te = a["Y_true"][sl]
                preds = {"naive": a["last"][sl], "seasonal_naive": a["seas"][sl]}
                # fitur terstandar
                Ltr, Lva, Lte = a["lags"][tr0:va0], a["lags"][va0:te0], a["lags"][te0:te1]
                Etr, Eva, Ete = a["extra"][tr0:va0], a["extra"][va0:te0], a["extra"][te0:te1]
                ytr, yva = a["Y_model"][tr0:va0], a["Y_model"][va0:te0]
                s_y = StandardScaler().fit(ytr.reshape(-1, 1)); s_l = StandardScaler().fit(Ltr.reshape(-1, 1))
                s_e = StandardScaler().fit(np.nan_to_num(Etr))
                scl = lambda L: (L - s_l.mean_[0]) / s_l.scale_[0]
                sce = lambda E: s_e.transform(np.nan_to_num(E))
                sc = lambda v: s_y.transform(v.reshape(-1, 1)).ravel()
                inv = lambda p: s_y.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()
                Ftr = np.column_stack([scl(Ltr), sce(Etr)]); Fva = np.column_stack([scl(Lva), sce(Eva)])
                Fte = np.column_stack([scl(Lte), sce(Ete)])
                timing = {}
                if "ar_ridge" in models:
                    t0 = time.time()
                    m = Ridge(alpha=1.0).fit(np.vstack([Ftr, Fva]), np.concatenate([sc(ytr), sc(yva)]))
                    preds["ar_ridge"] = a["recon"](inv(m.predict(Fte)), sl); timing["ar_ridge"] = time.time() - t0
                if "hgb" in models:
                    t0 = time.time()
                    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=4, early_stopping=True,
                                                      validation_fraction=0.15, random_state=0)
                    m.fit(np.vstack([Ftr, Fva]), np.concatenate([sc(ytr), sc(yva)]))
                    preds["hgb"] = a["recon"](inv(m.predict(Fte)), sl); timing["hgb"] = time.time() - t0
                for name, p in preds.items():
                    if name not in models:
                        continue
                    mt = metrics(y_te, p)
                    dm, pv = diebold_mariano(y_te, p, preds["naive"], h) if name != "naive" else (np.nan, np.nan)
                    rows.append({"service": service, "h": h, "variant": variant, "model": name, "fold": f, "seed": 0,
                                 **mt, "DM_vs_naive": dm, "DM_p": pv, "fit_s": round(timing.get(name, 0), 1), "epochs": 0})
                print(f"[{service} h={h:2d} {variant:9s} fold={f}] " +
                      " | ".join(f"{k}:{metrics(y_te, v)['sMAPE']:.2f}%" for k, v in preds.items() if k in models), flush=True)
                if "gru" in models:
                    for seed in seeds:
                        t0 = time.time()
                        p_s, ep = train_dl("gru", scl(Ltr), sce(Etr), sc(ytr), scl(Lva), sce(Eva), sc(yva),
                                           scl(Lte), sce(Ete), seed, epochs, patience)
                        p = a["recon"](inv(p_s), sl)
                        mt = metrics(y_te, p)
                        dm, pv = diebold_mariano(y_te, p, preds["naive"], h)
                        rows.append({"service": service, "h": h, "variant": variant, "model": "gru", "fold": f, "seed": seed,
                                     **mt, "DM_vs_naive": dm, "DM_p": pv, "fit_s": round(time.time() - t0, 1), "epochs": ep})
                        print(f"[{service} h={h:2d} {variant:9s} fold={f}] gru seed={seed} sMAPE={mt['sMAPE']:.2f}% "
                              f"ep={ep} ({time.time() - t0:.0f}s)", flush=True)
                if not shuffle:
                    out = pd.DataFrame({"timestamp": df.index[a["tgt"][sl]], "y_true": y_te,
                                        **{k: np.asarray(v) for k, v in preds.items()}})
                    out.to_csv(os.path.join(RES, f"pred_{service}_h{h}_{variant}_fold{f}.csv"), index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--services", nargs="+", default=["code", "conv"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6, 12])
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.quick:
        a.folds, a.seeds, a.epochs, a.patience = 1, [0], 3, 2
    import torch
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    tag = a.tag or ("quick" if a.quick else "full")
    t_all = time.time()

    # 1) uji kebocoran fitur
    df0 = pd.read_csv(os.path.join(SER, "azure_llm_2024_code_features_5min.csv"), index_col=0, parse_dates=True)
    leak = leakage_unit_test(df0)
    print("Uji kebocoran fitur (harus True untuk raw/diff/causal; noncausal boleh False):", leak, flush=True)

    # 2) eksperimen utama
    res = pd.concat([run(s, a.horizons, a.variants, a.models, a.folds, a.seeds, a.epochs, a.patience)
                     for s in a.services], ignore_index=True)
    res.to_csv(os.path.join(RES, f"metrics_runs_{tag}.csv"), index=False)
    summ = (res.groupby(["service", "h", "variant", "model"])
               .agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), sMAPE=("sMAPE", "mean"), sMAPE_std=("sMAPE", "std"),
                    DM_p=("DM_p", "median"), fit_s=("fit_s", "mean"), runs=("MAE", "size")).reset_index().round(3))
    summ.to_csv(os.path.join(RES, f"metrics_summary_{tag}.csv"), index=False)

    # 3) uji shuffle-target (hanya model cepat): tidak boleh ada yang jauh lebih baik dari naive
    sh = run("code", [3], ["raw", "causal", "noncausal"], ["naive", "seasonal_naive", "ar_ridge", "hgb"],
             a.folds, [], 0, 0, shuffle=True)
    sh_s = sh.groupby(["variant", "model"]).sMAPE.mean().round(2)
    sh_s.to_csv(os.path.join(RES, f"shuffle_test_{tag}.csv"))
    print("\nUji shuffle-target (sMAPE, Y diacak; model terlatih seharusnya ~ naive atau lebih buruk):\n", sh_s.to_string())

    with open(os.path.join(RES, f"config_{tag}.json"), "w") as fh:
        json.dump({**vars(a), "window": WINDOW, "season": SEASON, "w_days": W_DAYS, "leakage_test": leak,
                   "total_seconds": round(time.time() - t_all)}, fh, indent=2)
    pd.set_option("display.width", 220)
    print("\n== sMAPE rata-rata per varian ==")
    print(summ.pivot_table(index=["service", "h", "model"], columns="variant", values="sMAPE").round(2).to_string())
    print(f"\nTotal waktu: {time.time() - t_all:.0f} s")
