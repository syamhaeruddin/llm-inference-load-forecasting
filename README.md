# LLM Inference Load Forecasting: When Does It Pay Off?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24189585.svg)](https://doi.org/10.5281/zenodo.24189585)

Code, processed series, and experiment results for the manuscript

> Haeruddin, M. I. A. Tresnawan, S. E. Prasetyo, G. Wijaya, and W. Chen,
> "When Does Workload Forecasting Pay Off? A Cross-Provider Benchmark of LLM Inference Load with a Regime-Safe Combiner Selector,"
> submitted to *IEEE Transactions on Cloud Computing*, 2026.

* Repository: https://github.com/syamhaeruddin/llm-inference-load-forecasting
* Archived release (Zenodo): https://doi.org/10.5281/zenodo.24189585

The repository benchmarks persistence, statistical, tree-based, recurrent, and stacked forecasters on two production LLM inference traces
(Azure LLM Inference Trace 2024 and 42 GPU clusters of Alibaba cluster-trace-v2026-maas) at 5-60 min horizons, measures decomposition
leakage against a causal reference, evaluates regime-aware combiners and a nested out-of-fold selector (`meta_select`), and translates
forecast accuracy into GPU provisioning cost with a simulation.

## Repository layout

```
config.py                  location of raw data (env var FORECAST_DATA, default ./data)
00_download_data.(ps1|sh)  download Azure LLM Inference Trace 2024
00_download_maas.ps1       download Alibaba cluster-trace-v2026-maas (15.3 GB zip)
01_prepare_series.py       aggregate Azure requests/tokens to 1-min and 5-min series
08_prepare_maas.py         aggregate Alibaba instance_week_5min to per-cluster 5-min series
05_experiment_tokens.py    single models, rolling origin, horizons 5/15/30/60 min      (RQ1)
06_plots_tokens.py, 07_metrics_extra.py
09_decompose.py            raw / diff / causal / non-causal decomposition + leakage checks (RQ2)
09b_plots_decomp.py
10_metalearner.py          static and regime-aware combiners, nested selector (Azure)    (RQ3)
10b_plots_meta.py
11_run_maas.py             combiners on 42 Alibaba clusters                              (RQ3)
12_stats_maas.py           Friedman-Nemenyi, DM-HLN, regularity vs. skill
12b_plots_maas.py
13_autoscale_sim.py        GPU provisioning simulation                                   (RQ4)
14_tables.py, 15_sens_table.py, 16_table3_seedsd.py   manuscript tables
17_figures_en.py           manuscript figures
02_baseline_forecast.py, 03_plots.py, 04_feature_table.py   exploratory stage (DataCenter-Traces)
series/                    processed series used in the paper (CSV)
results/                   per-run metrics, summaries, simulation outputs (point predictions excluded)
figures/                   figures (figures/en: English versions used in the manuscript)
```

## Setup

Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

The processed series in `series/` are enough to rerun every experiment. Raw traces are only needed to rebuild them:

```bash
export FORECAST_DATA=/path/to/raw/data   # Windows: setx FORECAST_DATA C:\path\to\raw\data
bash 00_download_data.sh                  # Azure (about 1.8 GB)
# Alibaba: run 00_download_maas.ps1 or download cluster-trace-v2026-maas from github.com/alibaba/clusterdata
python 01_prepare_series.py
python 08_prepare_maas.py --skip-daily --skip-app
```

## Reproducing the manuscript

`run_all.sh` (Linux/macOS) and `run_all.ps1` (Windows) run the full pipeline with the settings reported in the paper.
On a laptop CPU the complete run takes roughly 10-12 hours, dominated by the GRU seeds; `--quick` flags on the
experiment scripts give a smoke test in minutes.

| Step | Command | Paper |
|---|---|---|
| Single models | `python 05_experiment_tokens.py --window 144 --seeds 0 1 2 --epochs 60 --patience 8 --folds 2 --tag w144` | RQ1 |
| Decomposition | `python 09_decompose.py --variants raw diff causal --seeds 0 1 --tag full` and `--variants noncausal --models naive seasonal_naive ar_ridge hgb --tag full_noncausal` | RQ2 |
| Combiners (Azure) | `python 10_metalearner.py --seeds 0 1 2 --epochs 60 --patience 8 --tag full5` | RQ3 |
| Alibaba clusters | `python 11_run_maas.py --top-dl 5 --seeds 0 --epochs 60 --patience 8 --tag full` then `python 12_stats_maas.py full` | RQ3 |
| Provisioning | `python 13_autoscale_sim.py --tag full5` (+ `--nref 50/200`, `--penalty 2/10` for sensitivity), `python 15_sens_table.py full5` | RQ4 |
| Tables, figures | `python 14_tables.py`, `python 16_table3_seedsd.py w144`, `python 17_figures_en.py` | |

## Data sources and licenses

* **Azure LLM Inference Trace 2024**, Microsoft Azure Public Dataset (github.com/Azure/AzurePublicDataset), CC BY 4.0.
* **Alibaba cluster-trace-v2026-maas** (github.com/alibaba/clusterdata). Only aggregated per-cluster 5-min series are included here;
  use of the original trace is subject to the terms of the Alibaba clusterdata repository.
* **DataCenter-Traces** (exploratory stage only), PyPI `datacentertracesdatasets`, doi:10.5281/zenodo.14564935, CC BY 4.0.

The derived series and results in this repository are released under CC BY 4.0; the code is released under the MIT License (see `LICENSE`).

## Citation

If you use this repository, please cite the manuscript above and the archived code and data:

> Haeruddin, M. I. A. Tresnawan, S. E. Prasetyo, G. Wijaya, and W. Chen, "LLM Inference Load Forecasting: code and data," Zenodo, v1.0.1, 2026, doi: 10.5281/zenodo.24189585.

See also `CITATION.cff`. The manuscript reference will be updated after publication.

## Contact

Haeruddin, Department of Information Technology, Faculty of Computer Science, Universitas Internasional Batam, Indonesia (haeruddin@uib.ac.id).
