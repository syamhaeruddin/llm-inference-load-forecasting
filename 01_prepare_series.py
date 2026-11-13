"""
Tahap 1: siapkan deret waktu untuk forecasting.
- Azure LLM 2024 (code & conv): agregasi per-request -> per 1 menit dan per 5 menit
  (jumlah request, total ContextTokens, total GeneratedTokens).
- DataCenter-Traces (Alibaba 2018, Google 2019, Azure v2): sudah agregat per 300 detik,
  cukup diberi indeks waktu.
Output: folder series/ berisi CSV siap pakai.
"""
import os
import pandas as pd

from config import AZURE_LLM_DIR as RAW
BASE = os.path.dirname(os.path.abspath(__file__))
try:
    import datacentertracesdatasets
    DCT = os.path.dirname(datacentertracesdatasets.__file__)   # paket PyPI (lintas-OS)
except ImportError:
    DCT = None
OUT = os.path.join(BASE, "series")
os.makedirs(OUT, exist_ok=True)


def aggregate_llm(name):
    path = os.path.join(RAW, f"AzureLLMInferenceTrace_{name}_1week.csv")
    parts = []
    for chunk in pd.read_csv(path, usecols=["TIMESTAMP", "ContextTokens", "GeneratedTokens"],
                             chunksize=2_000_000):
        chunk["TIMESTAMP"] = pd.to_datetime(chunk["TIMESTAMP"], utc=True, format="ISO8601")
        chunk["minute"] = chunk["TIMESTAMP"].dt.floor("1min")
        g = chunk.groupby("minute").agg(requests=("ContextTokens", "size"),
                                        context_tokens=("ContextTokens", "sum"),
                                        generated_tokens=("GeneratedTokens", "sum"))
        parts.append(g)
    df = pd.concat(parts).groupby(level=0).sum()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1min", tz="UTC")
    df = df.reindex(full_idx, fill_value=0)
    df.index.name = "timestamp"
    df.to_csv(os.path.join(OUT, f"azure_llm_2024_{name}_1min.csv"))
    df5 = df.resample("5min").sum()
    df5.to_csv(os.path.join(OUT, f"azure_llm_2024_{name}_5min.csv"))
    print(f"{name}: {len(df)} menit, {df['requests'].sum():,} request, "
          f"{df.index.min()} -> {df.index.max()}")
    return df


for n in ["code", "conv"]:
    aggregate_llm(n)

# DataCenter-Traces: beri indeks waktu sintetis (t0 = 0, langkah 300 detik)
if DCT is None:
    print("paket datacentertracesdatasets tidak ada; bagian DCT dilewati")
    raise SystemExit(0)
for name, rel in [("alibaba2018", "alibaba2018/machine_usage_grouped_300_seconds.csv"),
                  ("google2019", "google2019/machine_usage_grouped_300_seconds_percent.csv"),
                  ("azure_v2", "azure_v2/machine_usage_grouped_300_seconds.csv")]:
    df = pd.read_csv(os.path.join(DCT, rel))
    df.index = pd.to_timedelta(range(len(df)), unit="s") * 300
    df.index.name = "elapsed"
    df.to_csv(os.path.join(OUT, f"dct_{name}_300s.csv"))
    print(f"{name}: {len(df)} baris x {list(df.columns)}")
