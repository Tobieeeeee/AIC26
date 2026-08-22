param(
    [string]$MappingCsv = "data\mapping.csv",
    [string]$VideoRoot = "data\videos",
    [string]$OutputDir = "artifacts\_ocr_runtime\output",
    [string]$TempDir = "artifacts\_ocr_runtime",
    [string]$ModelsJson = "docker\ocr\model_sources.lock.json",
    [string]$WeightsRoot = "artifacts\_ocr_models"
)

# Convert to absolute paths
$MappingCsvPath = (Resolve-Path $MappingCsv -ErrorAction SilentlyContinue).Path
if (-not $MappingCsvPath) {
    $MappingCsvPath = Join-Path (Get-Location) $MappingCsv
}

$VideoRootPath = (Resolve-Path $VideoRoot -ErrorAction SilentlyContinue).Path
if (-not $VideoRootPath) {
    $VideoRootPath = Join-Path (Get-Location) $VideoRoot
}

# Add mvr to PYTHONPATH
$env:PYTHONPATH = "src"

Write-Host "Running Docker OCR Pipeline via Host Bridge..."
py -3.12 src\mvr\enrichment\ocr\docker_ocr.py `
    --mapping $MappingCsvPath `
    --videos $VideoRootPath `
    --output $OutputDir `
    --temp $TempDir `
    --model-lock $ModelsJson `
    --weights-root $WeightsRoot

if ($LASTEXITCODE -ne 0) {
    Write-Error "OCR Pipeline failed."
    exit $LASTEXITCODE
}

Write-Host "OCR Pipeline finished successfully."
