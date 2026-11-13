"""
Tahap 3-6: eksperimen forecasting Y = total_tokens per 5 menit, trace Azure LLM 2024.

Protokol
--------
- Input  : series/azure_llm_2024_<code|conv>_features_5min.csv (dari 04_feature_table.py)
- Target : total_tokens pada t+h-1, h in HORIZONS langkah (1,3,6,12 = 5,15,30,60 menit),
           diprediksi dari informasi sampai t-1 (strategi direct: satu model per horizon)
- Fitur  : window W=288 lag total_tokens (24 jam) + kalender waktu target (hod_sin, hod_cos,
           is_weekend) + komposisi beban pada t-1 (ctx_mean, gen_mean, long_prompt_share)
- Split  : rolling-origin, K fold. Fold k menguji 1 hari penuh (288 langkah); data sebelumnya
           jadi latih, dengan 12 jam terakhir latih sebagai validasi (early stopping / tuning).
- Scaler : StandardScaler di-fit hanya pada data latih fold tsb.
- Model  : naive, seasonal_naive, ar_ridge, hgb (HistGradientBoosting), lstm, gru, tcn, transformer
- Metrik : MAE, RMSE, sMAPE per fold; DL diulang N_SEEDS seed. Uji Diebold-Mariano vs naive.

Pemakaian
---------
python 05_experiment_tokens.py                      # penuh
python 05_experiment_tokens.py --quick              # uji cepat (epoch sedikit, 1 seed, 1 fold)
python 05_experiment_tokens.py --services code --horizons 1 12 --models naive lstm
"""
import argparse, json, os, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
SER = os.path.join(BASE, "series")
RES = os.path.join(BASE, "results", "tokens")
os.makedirs(RES, exist_ok=True)

SEASON = 288
WINDOW = 288
TARGET = "total_tokens"
CAL_COLS = ["hod_sin", "hod_cos", "is_weekend"]
COMP_COLS = ["ctx_mean", "gen_mean", "long_prompt_share"]
HORIZONS = [1, 3, 6, 12]
STAT_MODELS = ["naive", "seasonal_naive", "ar_ridge", "hgb"]
DL_MODELS = ["lstm", "gru", "tcn", "transformer"]


# ----------------------------------------------------------------------------- util
def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = np.abs(y) + np.abs(p)
    m = d != 0
    return 100 * np.mean(2 * np.abs(y[m] - p[m]) / d[m])


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": float(np.mean(np.abs(y - p))),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "sMAPE": float(smape(y, p))}


def diebold_mariano(y, p1, p2, h=1):
    """DM test, loss = squared error, d = e1^2 - e2^2. Negatif & p kecil => model 1 lebih baik.
    Varian long-run dengan Bartlett window hingga lag h-1 (Diebold & Mariano 1995), koreksi HLN."""
    y, p1, p2 = map(lambda a: np.asarray(a, float), (y, p1, p2))
    d = (y - p1) ** 2 - (y - p2) ** 2
    T = len(d)
    dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    var = gamma0
    for k in range(1, h):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * gk
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var / T)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    pval = 2 * (1 - stats.t.cdf(abs(hln), df=T - 1))
    return float(hln), float(pval)


def build_supervised(df, h):
    """Bangun X (window lag + kalender target + komposisi t-1) dan y untuk horizon h.
    Indeks sampel i berarti: informasi sampai baris i-1, target baris i+h-1."""
    y = df[TARGET].values.astype(float)
    cal = df[CAL_COLS].values.astype(float)
    comp = df[COMP_COLS].values.astype(float)
    n = len(y)
    idx = np.arange(WINDOW, n - h + 1)
    lags = np.stack([y[i - WINDOW:i] for i in idx])                 # urut lama -> baru
    tgt_i = idx + h - 1
    X_extra = np.column_stack([cal[tgt_i], comp[idx - 1]])
    Y = y[tgt_i]
    return idx, lags, X_extra, Y


def rolling_folds(n, k, test_len=SEASON, val_len=SEASON // 2):
    """Fold terakhir menguji test_len titik terakhir; fold sebelumnya digeser mundur 1 hari."""
    folds = []
    for j in range(k):
        te_end = n - j * test_len
        te_start = te_end - test_len
        va_start = te_start - val_len
        folds.append((0, va_start, te_start, te_end))      # train [0,va_start), val, test
    return folds[::-1]


# ----------------------------------------------------------------------------- model DL
def make_dl_model(kind, n_lags, n_extra, hidden=64, dropout=0.1):
    import torch, torch.nn as nn

    class RNNNet(nn.Module):
        def __init__(self, cell):
            super().__init__()
            self.rnn = cell(1, hidden, num_layers=2, batch_first=True, dropout=dropout)
            self.head = nn.Sequential(nn.Linear(hidden + n_extra, hidden), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, seq, extra):
            out, _ = self.rnn(seq.unsqueeze(-1))
            return self.head(torch.cat([out[:, -1], extra], 1)).squeeze(-1)

    class TCNNet(nn.Module):
        def __init__(self, channels=32, levels=6, k=3):
            super().__init__()
            layers, in_ch = [], 1
            for l in range(levels):
                d = 2 ** l
                layers += [nn.Conv1d(in_ch, channels, k, padding=(k - 1) * d, dilation=d),
                           nn.ReLU(), nn.Dropout(dropout)]
                in_ch = channels
            self.net = nn.Sequential(*layers)
            self.k, self.levels = k, levels
            self.head = nn.Sequential(nn.Linear(channels + n_extra, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 1))

        def forward(self, seq, extra):
            x = seq.unsqueeze(1)
            for layer in self.net:
                x = layer(x)
                if isinstance(layer, nn.Conv1d):
                    x = x[:, :, :seq.shape[1]]          # causal crop
            return self.head(torch.cat([x[:, :, -1], extra], 1)).squeeze(-1)

    class TransNet(nn.Module):
        def __init__(self, d_model=32, nhead=4, nlayers=2):
            super().__init__()
            self.inp = nn.Linear(1, d_model)
            self.pos = nn.Parameter(torch.randn(1, n_lags, d_model) * 0.02)
            enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                             dropout=dropout, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, nlayers)
            self.head = nn.Sequential(nn.Linear(d_model + n_extra, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 1))

        def forward(self, seq, extra):
            x = self.inp(seq.unsqueeze(-1)) + self.pos
            x = self.enc(x)
            return self.head(torch.cat([x[:, -1], extra], 1)).squeeze(-1)

    if kind == "lstm":
        return RNNNet(nn.LSTM)
    if kind == "gru":
        return RNNNet(nn.GRU)
    if kind == "tcn":
        return TCNNet()
    if kind == "transformer":
        return TransNet()
    raise ValueError(kind)


def train_dl(kind, Xl_tr, Xe_tr, y_tr, Xl_va, Xe_va, y_va, Xl_te, Xe_te,
             seed, epochs, patience, lr=1e-3, batch=64, min_epochs=15):
    """Early stopping baru boleh menghentikan setelah min_epochs (ditambahkan tahap 4: dengan data latih
    < 1000 sampel, beberapa seed berhenti di epoch ~10 pada minimum lokal yang buruk)."""
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_dl_model(kind, Xl_tr.shape[1], Xe_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xl_tr, Xe_tr, y_tr_t = t(Xl_tr), t(Xe_tr), t(y_tr)
    Xl_va, Xe_va, y_va_t = t(Xl_va), t(Xe_va), t(y_va)
    best, best_state, bad = np.inf, None, 0
    n = len(y_tr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = torch.mean((model(Xl_tr[b], Xe_tr[b]) - y_tr_t[b]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(torch.mean((model(Xl_va, Xe_va) - y_va_t) ** 2))
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience and ep + 1 >= min_epochs:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(t(Xl_te), t(Xe_te)).numpy(), ep + 1


# ----------------------------------------------------------------------------- eksperimen
def run(service, horizons, models, n_folds, seeds, epochs, patience):
    df = pd.read_csv(os.path.join(SER, f"azure_llm_2024_{service}_features_5min.csv"),
                     index_col=0, parse_dates=True)
    rows, preds_store = [], {}
    for h in horizons:
        idx, lags, extra, Y = build_supervised(df, h)
        n = len(Y)
        folds = rolling_folds(n, n_folds)
        for f, (tr0, va0, te0, te1) in enumerate(folds):
            y_te = Y[te0:te1]
            y_hist = df[TARGET].values.astype(float)
            # baseline tanpa pelatihan: naive = nilai terakhir yang diketahui (t-1),
            # seasonal naive = nilai 24 jam sebelum waktu target
            last_known = lags[te0:te1, -1]
            tgt_rows = idx[te0:te1] + h - 1
            seas = y_hist[tgt_rows - SEASON]
            preds = {"naive": last_known, "seasonal_naive": seas}

            # fitur untuk model terlatih
            Xl_tr, Xe_tr, y_tr = lags[tr0:va0], extra[tr0:va0], Y[tr0:va0]
            Xl_va, Xe_va, y_va = lags[va0:te0], extra[va0:te0], Y[va0:te0]
            Xl_te, Xe_te = lags[te0:te1], extra[te0:te1]
            s_y = StandardScaler().fit(y_tr.reshape(-1, 1))
            s_e = StandardScaler().fit(Xe_tr)
            sc = lambda a: s_y.transform(a.reshape(-1, 1)).ravel()
            scl = lambda L: (L - s_y.mean_[0]) / s_y.scale_[0]
            inv = lambda a: s_y.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()
            Ltr, Lva, Lte = scl(Xl_tr), scl(Xl_va), scl(Xl_te)
            Etr, Eva, Ete = s_e.transform(Xe_tr), s_e.transform(Xe_va), s_e.transform(Xe_te)
            ytr_s, yva_s = sc(y_tr), sc(y_va)
            Ftr = np.column_stack([Ltr, Etr]); Fva = np.column_stack([Lva, Eva]); Fte = np.column_stack([Lte, Ete])

            timing = {}
            if "ar_ridge" in models:
                t0 = time.time()
                m = Ridge(alpha=1.0).fit(np.vstack([Ftr, Fva]), np.concatenate([ytr_s, yva_s]))
                preds["ar_ridge"] = inv(m.predict(Fte)); timing["ar_ridge"] = time.time() - t0
            if "hgb" in models:
                t0 = time.time()
                m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=4,
                                                  early_stopping=True, validation_fraction=0.15,
                                                  random_state=0).fit(np.vstack([Ftr, Fva]),
                                                                      np.concatenate([ytr_s, yva_s]))
                preds["hgb"] = inv(m.predict(Fte)); timing["hgb"] = time.time() - t0

            for name, p in preds.items():
                if name not in models:
                    continue
                mt = metrics(y_te, p)
                dm, pv = diebold_mariano(y_te, p, preds["naive"], h) if name != "naive" else (np.nan, np.nan)
                rows.append({"service": service, "h": h, "model": name, "fold": f, "seed": 0,
                             **mt, "DM_vs_naive": dm, "DM_p": pv, "fit_s": round(timing.get(name, 0), 1),
                             "epochs": 0, "n_test": len(y_te)})
                preds_store[(service, h, name, f, 0)] = p
            print(f"[{service} h={h:2d} fold={f}] " +
                  " | ".join(f"{k}:{metrics(y_te, v)['sMAPE']:.2f}%" for k, v in preds.items() if k in models),
                  flush=True)

            for kind in [m for m in models if m in DL_MODELS]:
                for seed in seeds:
                    t0 = time.time()
                    p_s, ep = train_dl(kind, Ltr, Etr, ytr_s, Lva, Eva, yva_s, Lte, Ete,
                                       seed, epochs, patience)
                    p = inv(p_s)
                    mt = metrics(y_te, p)
                    dm, pv = diebold_mariano(y_te, p, preds["naive"], h)
                    rows.append({"service": service, "h": h, "model": kind, "fold": f, "seed": seed,
                                 **mt, "DM_vs_naive": dm, "DM_p": pv, "fit_s": round(time.time() - t0, 1),
                                 "epochs": ep, "n_test": len(y_te)})
                    preds_store[(service, h, kind, f, seed)] = p
                    print(f"[{service} h={h:2d} fold={f}] {kind:12s} seed={seed} sMAPE={mt['sMAPE']:.2f}% "
                          f"MAE={mt['MAE']:.0f} ep={ep} ({time.time() - t0:.0f}s)", flush=True)
            # simpan prediksi per fold
            out = pd.DataFrame({"timestamp": df.index[tgt_rows], "y_true": y_te,
                                **{f"{k[2]}_s{k[4]}": v for k, v in preds_store.items()
                                   if k[0] == service and k[1] == h and k[3] == f}})
            out.to_csv(os.path.join(RES, f"pred_{service}_h{h}_fold{f}.csv"), index=False)
    return pd.DataFrame(rows)


def summarize(res):
    g = res.groupby(["service", "h", "model"])
    s = g.agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
              RMSE_mean=("RMSE", "mean"), sMAPE_mean=("sMAPE", "mean"), sMAPE_std=("sMAPE", "std"),
              DM_mean=("DM_vs_naive", "mean"), DM_p_median=("DM_p", "median"),
              fit_s_mean=("fit_s", "mean"), runs=("MAE", "size")).reset_index()
    return s.round(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--services", nargs="+", default=["code", "conv"])
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    ap.add_argument("--models", nargs="+", default=STAT_MODELS + DL_MODELS)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--window", type=int, default=288, help="jumlah lag (288 = 24 jam, 144 = 12 jam)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.quick:
        a.folds, a.seeds, a.epochs, a.patience = 1, [0], 3, 2
    WINDOW = a.window
    globals()["WINDOW"] = a.window
    import torch
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    t_all = time.time()
    all_res = pd.concat([run(s, a.horizons, a.models, a.folds, a.seeds, a.epochs, a.patience)
                         for s in a.services], ignore_index=True)
    tag = a.tag or ("quick" if a.quick else "full")
    all_res.to_csv(os.path.join(RES, f"metrics_runs_{tag}.csv"), index=False)
    summ = summarize(all_res)
    summ.to_csv(os.path.join(RES, f"metrics_summary_{tag}.csv"), index=False)
    with open(os.path.join(RES, f"config_{tag}.json"), "w") as fh:
        json.dump({**vars(a), "window": WINDOW, "season": SEASON, "target": TARGET,
                   "cal_cols": CAL_COLS, "comp_cols": COMP_COLS,
                   "total_seconds": round(time.time() - t_all)}, fh, indent=2)
    pd.set_option("display.width", 200)
    print("\n== Ringkasan (rata-rata lintas fold & seed) ==")
    print(summ[["service", "h", "model", "MAE_mean", "sMAPE_mean", "sMAPE_std", "DM_mean", "DM_p_median", "fit_s_mean"]]
          .to_string(index=False))
    print(f"\nTotal waktu: {time.time() - t_all:.0f} s")
