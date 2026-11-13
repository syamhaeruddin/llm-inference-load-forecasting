#!/usr/bin/env bash
# Unduh data mentah yang dipakai di folder ini.
set -e
mkdir -p data/azure_llm
cd data/azure_llm
# Azure LLM Inference Trace 2024 (DynamoLLM, HPCA 2025) - 1 minggu, ~692 MB + ~1,14 GB
curl -L -C - -o AzureLLMInferenceTrace_code_1week.csv \
  https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv
curl -L -C - -o AzureLLMInferenceTrace_conv_1week.csv \
  https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv
# Azure LLM Inference Trace 2023 (Splitwise, ISCA 2024) - ~1 jam, kecil
curl -L -o AzureLLMInferenceTrace_code.csv \
  https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv
curl -L -o AzureLLMInferenceTrace_conv.csv \
  https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv
cd ../..
# DataCenter-Traces (Alibaba 2018, Google 2019, Azure v2, agregat 300 detik)
pip install datacentertracesdatasets==0.3.2
