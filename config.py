"""Lokasi data mentah. Default: folder data/ di sebelah skrip; bisa dialihkan lewat
variabel lingkungan FORECAST_DATA (mis. untuk menyimpan 17 GB data mentah di luar folder repo).
Contoh: setx FORECAST_DATA C:\\data\\forecasting
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("FORECAST_DATA") or os.path.join(BASE, "data")
AZURE_LLM_DIR = os.path.join(DATA_DIR, "azure_llm")
MAAS_DIR = os.path.join(DATA_DIR, "alibaba_maas")
