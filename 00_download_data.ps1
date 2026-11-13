# Unduh data mentah (Windows / PowerShell). Versi bash: 00_download_data.sh
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force -Path data\azure_llm | Out-Null
$rel = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/"
foreach ($f in "AzureLLMInferenceTrace_code_1week.csv", "AzureLLMInferenceTrace_conv_1week.csv") {
    curl.exe -L -C - -o "data\azure_llm\$f" "$rel$f"
}
$raw = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/"
foreach ($f in "AzureLLMInferenceTrace_code.csv", "AzureLLMInferenceTrace_conv.csv") {
    curl.exe -L -o "data\azure_llm\$f" "$raw$f"
}
# DataCenter-Traces (Alibaba 2018, Google 2019, Azure v2)
python -m pip install datacentertracesdatasets==0.3.2
