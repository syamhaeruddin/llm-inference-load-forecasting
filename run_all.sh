#!/usr/bin/env bash
# Reproduce all experiments reported in the manuscript from the processed series in series/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/tokens results/decomp results/meta results/maas results/autoscale results/tables
python -u 05_experiment_tokens.py --window 144 --seeds 0 1 2 --epochs 60 --patience 8 --folds 2 --tag w144
python -u 07_metrics_extra.py w144
python -u 06_plots_tokens.py w144
python -u 09_decompose.py --variants raw diff causal --seeds 0 1 --epochs 60 --patience 8 --tag full
python -u 09_decompose.py --variants noncausal --models naive seasonal_naive ar_ridge hgb --tag full_noncausal
python -u 09b_plots_decomp.py
python -u 10_metalearner.py --seeds 0 1 2 --epochs 60 --patience 8 --tag full5 --resume
python -u 10b_plots_meta.py
python -u 11_run_maas.py --top-dl 5 --seeds 0 --epochs 60 --patience 8 --tag full --resume
python -u 12_stats_maas.py full
python -u 12b_plots_maas.py full
python -u 13_autoscale_sim.py --tag full5
for n in 50 200; do python -u 13_autoscale_sim.py --tag full5 --nref $n --no-fig; done
for p in 2 10; do python -u 13_autoscale_sim.py --tag full5 --penalty $p --no-fig; done
python -u 15_sens_table.py full5
python -u 14_tables.py
python -u 16_table3_seedsd.py w144
python -u 17_figures_en.py
