$ErrorActionPreference = "Stop"

$datasetRoot = "D:\AIC26\data\raw"
$fullIndex = "D:\AIC26\artifacts\indexes\aic_full"
$smokeIndex = "D:\AIC26\artifacts\indexes\aic_smoke"

if (-not (Test-Path -LiteralPath $datasetRoot)) {
    throw "Dataset root not found: $datasetRoot"
}

if (-not (Test-Path -LiteralPath (Join-Path $fullIndex "manifest.json"))) {
    Write-Host "Full FAISS index is not built yet: $fullIndex" -ForegroundColor Yellow
    Write-Host "Smoke index exists at: $smokeIndex" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Build the full index first:" -ForegroundColor Cyan
    Write-Host "python -m mvr.cli build-aic-index --dataset-root `"$datasetRoot`" --output `"$fullIndex`" --device cpu"
    exit 2
}

$env:MVR_AIC_DATASET_ROOT = $datasetRoot
$env:MVR_KEYFRAME_DIR = Join-Path $datasetRoot "keyframes"
$env:MVR_VIDEO_DIR = Join-Path $datasetRoot "video"
$env:MVR_SEMANTIC_BACKEND = "faiss_reference"
$env:MVR_SEMANTIC_INDEX = $fullIndex
$env:MVR_DEVICE = "cpu"

python -m uvicorn mvr.api.app:app --host 127.0.0.1 --port 8000
