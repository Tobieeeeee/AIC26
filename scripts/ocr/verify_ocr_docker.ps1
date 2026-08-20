Write-Host "Verifying DeepSolo Docker..."
docker run --rm --gpus all -v "$(Get-Location):/workspace" mvr/deepsolo python /workspace/docker/ocr/worker/verify_deepsolo.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "DeepSolo verification failed."
}

Write-Host "Verifying PARSeq-Vietnamese Docker..."
docker run --rm --gpus all -v "$(Get-Location):/workspace" mvr/parseq_vi python /workspace/docker/ocr/worker/verify_parseq.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "PARSeq-Vietnamese verification failed."
}

Write-Host "Verification complete."
