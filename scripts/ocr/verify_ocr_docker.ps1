param(
    [string]$ModelsJson = "docker\ocr\model_sources.lock.json",
    [string]$WeightsRoot = "artifacts\_ocr_models",
    [string]$Image = ""
)

$previousPythonPath = $env:PYTHONPATH
try {
    $srcPath = (Resolve-Path -LiteralPath "src").Path
    if ($previousPythonPath) {
        $env:PYTHONPATH = "$srcPath$([IO.Path]::PathSeparator)$previousPythonPath"
    } else {
        $env:PYTHONPATH = $srcPath
    }

    py -3.12 -m mvr.enrichment.ocr.model_integrity verify `
        --lock $ModelsJson `
        --weights-root $WeightsRoot
    if ($LASTEXITCODE -ne 0) {
        Write-Error "OCR model preflight failed closed."
        exit $LASTEXITCODE
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}

$lock = Get-Content -Raw -LiteralPath $ModelsJson | ConvertFrom-Json
$deepsolo = $lock.models.deepsolo
$parseq = $lock.models.parseq_vi
$workspace = (Get-Location).Path
$weights = (Resolve-Path -LiteralPath $WeightsRoot).Path
$modelMount = "${weights}:/models:ro"
$workspaceMount = "${workspace}:/workspace:ro"

Write-Host "Verifying DeepSolo Docker..."
$deepArgs = @(
    "run", "--rm", "--gpus", "all",
    "-v", $workspaceMount,
    "-v", $modelMount,
    "mvr-ocr-deepsolo:task04b",
    "python", "/workspace/docker/ocr/worker/verify_deepsolo.py",
    "--config", $deepsolo.runtime_config,
    "--weights", $deepsolo.runtime_path,
    "--expected-sha256", $deepsolo.sha256
)
if ($Image) {
    $deepArgs += @("--image", $Image)
}
docker @deepArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "DeepSolo verification failed."
    exit $LASTEXITCODE
}

Write-Host "Verifying PARSeq-Vietnamese Docker..."
$parseqArgs = @(
    "run", "--rm", "--gpus", "all",
    "-v", $workspaceMount,
    "-v", $modelMount,
    "mvr-ocr-parseq-vi:task04b",
    "python", "/workspace/docker/ocr/worker/verify_parseq.py",
    "--weights", $parseq.runtime_path,
    "--expected-sha256", $parseq.sha256
)
if ($Image) {
    $parseqArgs += @("--image", $Image)
}
docker @parseqArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "PARSeq-Vietnamese verification failed."
    exit $LASTEXITCODE
}

Write-Host "All OCR Docker verifications passed."
