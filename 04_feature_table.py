"""
Tahap 4: tabel fitur per interval 5 menit dari trace Azure LLM 2024.
Dari data mentah (1 baris = 1 request: TIMESTAMP, ContextTokens, GeneratedTokens)
dibangun fitur per interval yang bisa dipakai sebagai target (Y) maupun fitur (X).
Output: series/azure_llm_2024_<code|conv>_features_5min.csv
"""
import os
import numpy as np
import pandas as pd

from config import AZURE_LLM_DIR as RAW
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "series")
FREQ = "5min"


def build(name):
    path = os.path.join(RAW, f"AzureLLMInferenceTrace_{name}_1week.csv")
    parts = []
    for ch in pd.read_csv(path, chunksize=2_000_000):
        ch["TIMESTAMP"] = pd.to_datetime(ch["TIMESTAMP"], utc=True, format="ISO8601")
        ch["t"] = ch["TIMESTAMP"].dt.floor(FREQ)
        ch["total"] = ch["ContextTokens"] + ch["GeneratedTokens"]
        g = ch.groupby("t").agg(
            requests=("ContextTokens", "size"),
            context_tokens=("ContextTokens", "sum"),
            generated_tokens=("GeneratedTokens", "sum"),
            total_tokens=("total", "sum"),
            ctx_mean=("ContextTokens", "mean"),
            ctx_p95=("ContextTokens", lambda s: s.quantile(0.95)),
            gen_mean=("GeneratedTokens", "mean"),
            gen_p95=("GeneratedTokens", lambda s: s.quantile(0.95)),
            long_prompt_share=("ContextTokens", lambda s: (s > 4000).mean()),
        )
        parts.append(g)
    df = pd.concat(parts)
    # gabungkan chunk yang terbelah pada interval yang sama (bobot = requests)
    sums = df.groupby(level=0)[["requests", "context_tokens", "generated_tokens", "total_tokens"]].sum()
    w = df["requests"]
    wavg = (df[["ctx_mean", "ctx_p95", "gen_mean", "gen_p95", "long_prompt_share"]]
            .mul(w, axis=0).groupby(level=0).sum().div(sums["requests"], axis=0))
    df = pd.concat([sums, wavg], axis=1)
    idx = pd.date_range(df.index.min(), df.index.max(), freq=FREQ, tz="UTC")
    df = df.reindex(idx)
    df.index.name = "timestamp"

    # fitur kalender (UTC)
    df["hour"] = df.index.hour
    df["minute_of_day"] = df.index.hour * 60 + df.index.minute
    df["dow"] = df.index.day_of_week          # 0 = Senin
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hod_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["hod_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)

    # contoh fitur lag (sisanya dibuat saat pemodelan)
    for col in ["requests", "total_tokens"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag12"] = df[col].shift(12)      # 1 jam lalu
        df[f"{col}_lag288"] = df[col].shift(288)    # 24 jam lalu
        df[f"{col}_roll12_mean"] = df[col].rolling(12).mean()

    df.round(3).to_csv(os.path.join(OUT, f"azure_llm_2024_{name}_features_5min.csv"))
    return df


if __name__ == "__main__":
    for n in ["code", "conv"]:
        d = build(n)
        print(f"\n== {n}: {d.shape[0]} interval x {d.shape[1]} kolom")
        print(d.iloc[300:305, :9].round(1).to_string())
