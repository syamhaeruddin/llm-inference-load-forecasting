# Reproduce all experiments reported in the manuscript from the processed series in series\ (Windows).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
foreach ($d in "tokens","decomp","meta","maas","autoscale","tables") { New-Item -ItemType Directory -Force -Path "results\$d" | Out-Null }
$steps = @(
  "05_experiment_tokens.py --window 144 --seeds 0 1 2 --epochs 60 --patience 8 --folds 2 --tag w144",
  "07_metrics_extra.py w144",
  "06_plots_tokens.py w144",
  "09_decompose.py --variants raw diff causal --seeds 0 1 --epochs 60 --patience 8 --tag full",
  "09_decompose.py --variants noncausal --models naive seasonal_naive ar_ridge hgb --tag full_noncausal",
  "09b_plots_decomp.py",
  "10_metalearner.py --seeds 0 1 2 --epochs 60 --patience 8 --tag full5 --resume",
  "10b_plots_meta.py",
  "11_run_maas.py --top-dl 5 --seeds 0 --epochs 60 --patience 8 --tag full --resume",
  "12_stats_maas.py full",
  "12b_plots_maas.py full",
  "13_autoscale_sim.py --tag full5",
  "13_autoscale_sim.py --tag full5 --nref 50 --no-fig",
  "13_autoscale_sim.py --tag full5 --nref 200 --no-fig",
  "13_autoscale_sim.py --tag full5 --penalty 2 --no-fig",
  "13_autoscale_sim.py --tag full5 --penalty 10 --no-fig",
  "15_sens_table.py full5",
  "14_tables.py",
  "16_table3_seedsd.py w144",
  "17_figures_en.py"
)
foreach ($s in $steps) {
  Write-Host ">> python $s"
  & python -u @($s -split ' ')
  if ($LASTEXITCODE -ne 0) { throw "failed: $s" }
}
