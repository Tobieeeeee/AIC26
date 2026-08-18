param(
    [string]$InputCsv = "artifacts\verification\tkis_query_translation\queries.csv"
)

Write-Host "=========================================="
Write-Host "1. Starting Translation Phase (HPLT Model)"
Write-Host "=========================================="

python -m scripts.verification.run_tkis_query_translation_benchmark `
  --input $InputCsv `
  --output "artifacts\verification\tkis_query_translation\results_hplt" `
  --methods "local_vi2en" `
  --local-model "Helsinki-NLP/opus-mt-vi-en"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Translation phase failed! Stopping script."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "============================================="
Write-Host "2. Starting Decomposition Phase (Qwen Model) "
Write-Host "============================================="

python -m scripts.verification.run_tkis_query_decomposition_benchmark `
  --input "artifacts\verification\tkis_query_translation\results_hplt\translation_results.json" `
  --output "artifacts\verification\tkis_query_decomposition_hplt" `
  --source-method "local_vi2en" `
  --qwen-model "Qwen/Qwen1.5-1.8B" `
  --device cpu `
  --no-cache

if ($LASTEXITCODE -ne 0) {
    Write-Error "Decomposition phase failed!"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Pipeline Complete! Results are located in:"
Write-Host " - artifacts\verification\tkis_query_translation\results_hplt"
Write-Host " - artifacts\verification\tkis_query_decomposition_hplt"
Write-Host "=========================================="
