# Unduh Alibaba cluster-trace-v2026-maas (ATC 2026) ke folder data (default: $env:FORECAST_DATA\alibaba_maas).
# Pemakaian: powershell -File 00_download_maas.ps1 [-DataDir C:\data\forecasting\alibaba_maas]
param([string]$DataDir = $(if ($env:FORECAST_DATA) { Join-Path $env:FORECAST_DATA "alibaba_maas" } else { Join-Path $PSScriptRoot "data\alibaba_maas" }))
$ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
Set-Location $DataDir
$base = "https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-v2026-maas/data/"
# ukuran resmi (docs/data_download.md): app_metrics 99.947 B; instance_week_5min 4.812.189.789 B; instance_daily 10.528.173.427 B
foreach ($f in "app_metrics.csv.zip", "instance_week_5min.zip", "instance_daily.zip") {
    $t0 = Get-Date
    curl.exe -fL -C - -s -o $f ($base + $f)
    $len = (Get-Item $f).Length
    $sec = [int]((Get-Date) - $t0).TotalSeconds
    Add-Content download.log ("{0} done {1} {2} bytes, {3} s" -f (Get-Date -Format "HH:mm:ss"), $f, $len, $sec)
}
foreach ($f in "app_metrics.csv.zip", "instance_week_5min.zip", "instance_daily.zip") {
    Expand-Archive -Path $f -DestinationPath $DataDir -Force
    Add-Content download.log ("{0} extracted {1}" -f (Get-Date -Format "HH:mm:ss"), $f)
}
Add-Content download.log "ALL_DONE"
