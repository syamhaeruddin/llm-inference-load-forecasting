"""
Tahap 5: meta-learner sadar-rezim (stacking) di atas base learner raw + diff + persistence.
Deret: Azure LLM 2024 code & conv, Y = total_tokens per 5 menit. Representasi dari 09_decompose.build().

Skema per (layanan, horizon, fold):
  train [0, va0) | val [va0, te0) | test [te0, te1)     (rolling_folds dari 05, 2 fold @ 1 hari uji)
  Out-of-fold (OOF) untuk meta-learner: inner expanding window, N_OOF jendela @ 144 (default 3 -> 432 baris)
      fit base pada [0, c_k) -> prediksi [c_k, c_{k+1}) untuk c = [va0-288, va0-144, va0, te0]
  Base untuk test: fit pada [0, te0) (train+val), prediksi test.
  Meta-learner dilatih pada OOF (2 x 144 = 288 baris), diuji pada test.
Base learner (level 0): naive, seasonal_naive, ar_ridge_raw/diff, hgb_raw/diff, gru_raw/diff (rata-rata N seed).
Level 1 / pembanding:
  best_single      : base dengan MAE OOF terkecil (pemilihan model)
  avg              : rata-rata semua base
  inv_err          : bobot statis ~ 1/MAE_OOF
  nnls             : bobot non-negatif berjumlah 1, dipelajari dari OOF (statis)
  meta_ridge_regime: Ridge pada [pred_base, pred_base x fitur_rezim]  -> bobot linear terhadap rezim
  meta_hgb_regime  : HGB pada [pred_base, fitur_rezim]
  meta_hgb_noregime: HGB pada [pred_base] saja                        -> ablasi (d)
  meta_select      : pilih combiner (naive/best_single/avg/nnls/ridge_regime/hgb_noregime) dari OOF secara nested
  oracle_single    : base terbaik pada TEST (batas atas pemilihan model, bukan metode)
Matriks OOF/test base disimpan (oof_*.npz) agar combiner baru bisa diuji tanpa melatih ulang base learner.
run_df() generik: dipakai 11_run_maas.py untuk deret cluster Alibaba.
Keluaran: results/meta/metrics_runs_<tag>.csv, metrics_summary_<tag>.csv, weights_<tag>.csv (bobot per rezim),
          pred_<svc>_h<h>_fold<f>.csv
Pemakaian: python 10_metalearner.py [--quick] [--services code conv] [--horizons 1 3 6 12] [--seeds 0 1 2] [--tag t]
"""
import argparse, importlib, json, os, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
SER = os.path.join(BASE, "series")
RES = os.path.join(BASE, "results", "meta")
os.makedirs(RES, exist_ok=True)
exp = importlib.import_module("05_experiment_tokens")
dec = importlib.import_module("09_decompose")
metrics, diebold_mariano, rolling_folds, train_dl = exp.metrics, exp.diebold_mariano, exp.rolling_folds, exp.train_dl

OOF_LEN = 144
N_OOF = 3            # jumlah jendela inner (OOF = N_OOF x 144 baris)
META_REGIME_IDX = [2, 3, 4, 5]   # is_weekend, vol_1h, level_ratio, dev_profile (tanpa hod sin/cos: interaksi terlalu banyak)
REGIME_COLS = ["hod_sin", "hod_cos", "is_weekend", "vol_1h", "level_ratio", "dev_profile"]   # urutan di build().extra
BASE_MODELS = ["ar_ridge", "hgb", "gru"]
REPS = ["raw", "diff"]


# ----------------------------------------------------------------------------- base learner
def fit_predict(model, rep, tr, pr, seeds, epochs, patience):
    """Latih `model` pada rep[tr] (slice), prediksi rep[pr]; kembalikan prediksi skala asli (rata-rata seed utk GRU)."""
    Ltr, Etr, ytr = rep["lags"][tr], rep["extra"][tr], rep["Y_model"][tr]
    Lpr, Epr = rep["lags"][pr], rep["extra"][pr]
    s_y = StandardScaler().fit(ytr.reshape(-1, 1)); s_l = StandardScaler().fit(Ltr.reshape(-1, 1))
    s_e = StandardScaler().fit(np.nan_to_num(Etr))
    scl = lambda L: (L - s_l.mean_[0]) / s_l.scale_[0]
    sce = lambda E: s_e.transform(np.nan_to_num(E))
    sc = lambda v: s_y.transform(v.reshape(-1, 1)).ravel()
    inv = lambda p: s_y.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()
    if model == "ar_ridge":
        m = Ridge(alpha=1.0).fit(np.column_stack([scl(Ltr), sce(Etr)]), sc(ytr))
        p = inv(m.predict(np.column_stack([scl(Lpr), sce(Epr)])))
    elif model == "hgb":
        m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=4, early_stopping=True,
                                          validation_fraction=0.15, random_state=0)
        m.fit(np.column_stack([scl(Ltr), sce(Etr)]), sc(ytr))
        p = inv(m.predict(np.column_stack([scl(Lpr), sce(Epr)])))
    elif model == "gru":
        n = len(ytr); cut = n - 72                      # 6 jam terakhir rentang latih = validasi early stopping
        ps = []
        for seed in seeds:
            p_s, _ = train_dl("gru", scl(Ltr[:cut]), sce(Etr[:cut]), sc(ytr[:cut]), scl(Ltr[cut:]), sce(Etr[cut:]),
                              sc(ytr[cut:]), scl(Lpr), sce(Epr), seed, epochs, patience)
            ps.append(inv(p_s))
        p = np.mean(ps, axis=0)
    else:
        raise ValueError(model)
    return rep["recon"](p, pr)


# ----------------------------------------------------------------------------- meta-learner
def meta_fit_predict(kind, P_oof, R_oof, y_oof, P_te, R_te, scale):
    """P = prediksi base (n x k), R = fitur rezim (n x 6), y = target. Semua pada skala asli; dibagi `scale`."""
    Po, Pt, yo = P_oof / scale, P_te / scale, y_oof / scale
    if kind == "avg":
        return Pt.mean(1) * scale, None
    if kind == "inv_err":
        w = 1 / np.maximum(np.abs(Po - yo[:, None]).mean(0), 1e-9); w /= w.sum()
        return Pt @ w * scale, w
    if kind == "nnls":
        w, _ = nnls(Po, yo); w = w / w.sum() if w.sum() > 0 else np.full(Po.shape[1], 1 / Po.shape[1])
        return Pt @ w * scale, w
    if kind == "meta_ridge_regime":
        Ro, Rt = np.nan_to_num(R_oof[:, META_REGIME_IDX]), np.nan_to_num(R_te[:, META_REGIME_IDX])
        sr = StandardScaler().fit(Ro); Ro, Rt = sr.transform(Ro), sr.transform(Rt)
        X = lambda P, R: np.column_stack([P] + [P * R[:, [j]] for j in range(R.shape[1])])
        m = Ridge(alpha=5.0).fit(X(Po, Ro), yo)
        k = Po.shape[1]
        coef = m.coef_.reshape(1 + Ro.shape[1], k)                    # [1+6, k]
        w_eff = coef[0][None, :] + Rt @ coef[1:]                      # bobot efektif per baris test (n x k)
        return m.predict(X(Pt, Rt)) * scale, w_eff
    if kind in ("meta_hgb_regime", "meta_hgb_noregime"):
        Xo = np.column_stack([Po, np.nan_to_num(R_oof[:, META_REGIME_IDX])]) if kind == "meta_hgb_regime" else Po
        Xt = np.column_stack([Pt, np.nan_to_num(R_te[:, META_REGIME_IDX])]) if kind == "meta_hgb_regime" else Pt
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=10,
                                          random_state=0).fit(Xo, yo)
        return m.predict(Xt) * scale, None
    raise ValueError(kind)


# ----------------------------------------------------------------------------- eksperimen
def run(service, horizons, seeds, epochs, patience, n_folds=2):
    df = pd.read_csv(os.path.join(SER, f"azure_llm_2024_{service}_features_5min.csv"), index_col=0, parse_dates=True)
    return run_df(df, service, horizons, seeds, epochs, patience, n_folds=n_folds)


META_SELECT_CANDIDATES = ["naive", "best_single", "avg", "nnls", "meta_ridge_regime", "meta_hgb_noregime"]


def run_df(df, service, horizons, seeds, epochs, patience, n_folds=2, target="total_tokens",
           base_models=None, res_dir=None, partial_path=None, done=()):
    """Pipeline level 0 + level 1 untuk satu deret (df harus punya kolom target, hod_sin, hod_cos, is_weekend).
    base_models: subset BASE_MODELS (mis. tanpa 'gru' untuk cluster kecil). OOF disimpan ke res_dir/oof_*.npz."""
    base_models = base_models or BASE_MODELS
    res_dir = res_dir or RES
    dec.TARGET = target
    rows, wrows = [], []
    for h in horizons:
        reps = {r: dec.build(df, h, r) for r in REPS}
        n = len(reps["raw"]["Y_true"]); y = reps["raw"]["Y_true"]
        for f, (tr0, va0, te0, te1) in enumerate(rolling_folds(n, n_folds)):
            if (service, h, f) in done:
                print(f"[{service} h={h:2d} fold={f}] dilewati (resume)", flush=True)
                continue
            cuts = [va0 - k * OOF_LEN for k in range(N_OOF - 1, -1, -1)] + [te0]   # [c1, c2, va0, te0]
            c1 = cuts[0]
            oof = slice(c1, te0); te = slice(te0, te1)
            t0 = time.time()
            # --- base learner: OOF (2 inner fit) dan test (1 fit)
            P_oof, P_te, names = [], [], []
            for nm, arr_o, arr_t in [("naive", reps["raw"]["last"][oof], reps["raw"]["last"][te]),
                                     ("seasonal_naive", reps["raw"]["seas"][oof], reps["raw"]["seas"][te])]:
                P_oof.append(arr_o); P_te.append(arr_t); names.append(nm)
            for model in base_models:
                for r in REPS:
                    rep = reps[r]
                    p_in = [fit_predict(model, rep, slice(0, cuts[k]), slice(cuts[k], cuts[k + 1]), seeds, epochs, patience)
                            for k in range(len(cuts) - 1)]
                    pt = fit_predict(model, rep, slice(0, te0), te, seeds, epochs, patience)
                    P_oof.append(np.concatenate(p_in)); P_te.append(pt); names.append(f"{model}_{r}")
                    print(f"[{service} h={h:2d} fold={f}] base {model}_{r:4s} test sMAPE={metrics(y[te], pt)['sMAPE']:.2f}%"
                          f" ({time.time() - t0:.0f}s)", flush=True)
            P_oof, P_te = np.column_stack(P_oof), np.column_stack(P_te)
            for P in (P_oof, P_te):                      # prediksi non-finit (overflow) -> naive, agar level 1 aman
                bad = ~np.isfinite(P)
                if bad.any():
                    P[bad] = np.broadcast_to(P[:, [0]], P.shape)[bad]
            R_oof, R_te = reps["raw"]["extra"][oof], reps["raw"]["extra"][te]
            y_oof, y_te = y[oof], y[te]
            scale = max(float(np.mean(np.abs(y[:va0]))), 1e-9)        # skala normalisasi level 1 (aman utk deret ~0)
            preds = {nm: P_te[:, i] for i, nm in enumerate(names)}
            # --- pemilihan & kombinasi
            mae_oof = np.abs(P_oof - y_oof[:, None]).mean(0)
            best_i = int(np.argmin(mae_oof)); preds["best_single"] = P_te[:, best_i]
            best_name = names[best_i]
            orc_i = int(np.argmin(np.abs(P_te - y_te[:, None]).mean(0))); preds["oracle_single"] = P_te[:, orc_i]
            weights = {}
            for kind in ["avg", "inv_err", "nnls", "meta_ridge_regime", "meta_hgb_regime", "meta_hgb_noregime"]:
                preds[kind], w = meta_fit_predict(kind, P_oof, R_oof, y_oof, P_te, R_te, scale)
                if w is not None:
                    weights[kind] = w
            # meta_select (nested): combiner dilatih pada OOF awal (2/3), dinilai pada OOF akhir (1/3); yang
            # MAE-nya terkecil dipakai (versi yang sudah dilatih pada seluruh OOF) untuk data uji
            cut = len(y_oof) - OOF_LEN
            inner = {}
            for cand in META_SELECT_CANDIDATES:
                if cand == "naive":
                    pv = P_oof[cut:, names.index("naive")]
                elif cand == "best_single":
                    bi = int(np.argmin(np.abs(P_oof[:cut] - y_oof[:cut, None]).mean(0))); pv = P_oof[cut:, bi]
                else:
                    pv, _ = meta_fit_predict(cand, P_oof[:cut], R_oof[:cut], y_oof[:cut], P_oof[cut:], R_oof[cut:], scale)
                inner[cand] = float(np.abs(pv - y_oof[cut:]).mean())
            sel = min(inner, key=inner.get)
            preds["meta_select"] = preds[sel]
            np.savez_compressed(os.path.join(res_dir, f"oof_{service}_h{h}_fold{f}.npz"), names=np.array(names),
                                P_oof=P_oof, P_te=P_te, R_oof=R_oof, R_te=R_te, y_oof=y_oof, y_te=y_te, scale=scale,
                                meta_select=sel)
            # --- metrik
            for nm, p in preds.items():
                mt = metrics(y_te, p)
                dm_n, p_n = diebold_mariano(y_te, p, preds["naive"], h) if nm != "naive" else (np.nan, np.nan)
                dm_b, p_b = diebold_mariano(y_te, p, preds["best_single"], h) if nm != "best_single" else (np.nan, np.nan)
                rows.append({"service": service, "h": h, "fold": f, "model": nm, **mt, "DM_vs_naive": dm_n, "DM_p_naive": p_n,
                             "DM_vs_best": dm_b, "DM_p_best": p_b, "best_single_is": best_name,
                             "oracle_single_is": names[orc_i], "meta_select_is": sel, "n_test": len(y_te)})
            print(f"[{service} h={h:2d} fold={f}] best_single={best_name} | " +
                  " | ".join(f"{k}:{metrics(y_te, preds[k])['sMAPE']:.2f}%" for k in
                             ["naive", "best_single", "nnls", "meta_ridge_regime", "meta_select", "oracle_single"]) +
                  f" | select={sel}",
                  flush=True)
            # --- bobot per rezim (meta_ridge_regime: bobot efektif per baris; nnls/inv_err: statis)
            is_we = R_te[:, 2] > 0.5
            lvl = np.nan_to_num(R_te[:, 4]); shift = np.abs(lvl - 1) > 0.2
            for kind, w in weights.items():
                if w.ndim == 1:
                    for i, nm in enumerate(names):
                        wrows.append({"service": service, "h": h, "fold": f, "meta": kind, "regime": "all", "base": nm, "weight": w[i]})
                else:
                    for reg, mask in [("all", np.ones(len(w), bool)), ("weekend", is_we), ("weekday", ~is_we),
                                      ("level_shift", shift), ("level_stable", ~shift)]:
                        if mask.sum() == 0:
                            continue
                        for i, nm in enumerate(names):
                            wrows.append({"service": service, "h": h, "fold": f, "meta": kind, "regime": reg, "base": nm,
                                          "weight": float(w[mask, i].mean())})
            out = pd.DataFrame({"timestamp": df.index[reps["raw"]["tgt"][te]], "y_true": y_te, **preds})
            out.to_csv(os.path.join(res_dir, f"pred_{service}_h{h}_fold{f}.csv"), index=False)
            if partial_path:                                   # simpan bertahap agar tahan terhadap proses mati
                pd.DataFrame(rows).to_csv(partial_path, index=False)
                pd.DataFrame(wrows).to_csv(partial_path.replace("runs", "weights"), index=False)
    return pd.DataFrame(rows), pd.DataFrame(wrows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--services", nargs="+", default=["code", "conv"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6, 12])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-gru", action="store_true")
    ap.add_argument("--resume", action="store_true", help="lewati (layanan,h,fold) yang sudah ada di partial")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.patience = [0], 3, 2
    if a.no_gru:
        BASE_MODELS.remove("gru")
    import torch
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    tag = a.tag or ("quick" if a.quick else "full")
    t_all = time.time()
    res, wts, done = [], [], set()
    partial = os.path.join(RES, f"metrics_runs_{tag}_partial.csv")
    if a.resume and os.path.exists(partial):
        prev = pd.read_csv(partial); res.append(prev)
        pw = partial.replace("runs", "weights")
        if os.path.exists(pw):
            wts.append(pd.read_csv(pw))
        done = set(zip(prev.service, prev.h, prev.fold))
        print(f"resume: {len(done)} (layanan,h,fold) sudah ada", flush=True)
        partial = os.path.join(RES, f"metrics_runs_{tag}_partial2.csv")   # file baru untuk sisa
    for s in a.services:
        df_s = pd.read_csv(os.path.join(SER, f"azure_llm_2024_{s}_features_5min.csv"), index_col=0, parse_dates=True)
        r, w = run_df(df_s, s, a.horizons, a.seeds, a.epochs, a.patience, partial_path=partial, done=done)
        res.append(r); wts.append(w)
    res, wts = pd.concat(res, ignore_index=True), pd.concat(wts, ignore_index=True)
    res.to_csv(os.path.join(RES, f"metrics_runs_{tag}.csv"), index=False)
    wts.to_csv(os.path.join(RES, f"weights_{tag}.csv"), index=False)
    summ = (res.groupby(["service", "h", "model"])
               .agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), sMAPE=("sMAPE", "mean"), sMAPE_std=("sMAPE", "std"),
                    DM_p_naive=("DM_p_naive", "median"), DM_p_best=("DM_p_best", "median")).reset_index().round(3))
    summ.to_csv(os.path.join(RES, f"metrics_summary_{tag}.csv"), index=False)
    with open(os.path.join(RES, f"config_{tag}.json"), "w") as fh:
        json.dump({**vars(a), "oof_len": OOF_LEN, "n_oof": N_OOF, "meta_regime_idx": META_REGIME_IDX, "base_models": BASE_MODELS, "reps": REPS,
                   "total_seconds": round(time.time() - t_all)}, fh, indent=2)
    pd.set_option("display.width", 220)
    order = ["naive", "seasonal_naive", "best_single", "avg", "inv_err", "nnls", "meta_ridge_regime",
             "meta_hgb_regime", "meta_hgb_noregime", "meta_select", "oracle_single"]
    piv = summ.pivot(index=["service", "h"], columns="model", values="sMAPE")
    print("\n== sMAPE ==\n", piv[[c for c in order if c in piv.columns]].round(2).to_string())
    print("\n== bobot meta_ridge_regime per rezim (rata-rata fold), h=12 ==")
    w12 = wts[(wts.meta == "meta_ridge_regime") & (wts.h == 12)]
    print(w12.pivot_table(index=["service", "regime"], columns="base", values="weight").round(2).to_string())
    print(f"\nTotal waktu: {time.time() - t_all:.0f} s")
