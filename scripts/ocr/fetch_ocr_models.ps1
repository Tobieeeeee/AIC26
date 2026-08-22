param (
    [string]$ModelsJson = "docker\ocr\model_sources.lock.json",
    [string]$WeightsRoot = "artifacts\_ocr_models"
)

$previousPythonPath = $env:PYTHONPATH
try {
    $srcPath = (Resolve-Path -LiteralPath "src").Path
    if ($previousPythonPath) {
        $env:PYTHONPATH = "$srcPath$([IO.Path]::PathSeparator)$previousPythonPath"
    } else {
        $env:PYTHONPATH = $srcPath
    }

    py -3.12 -m mvr.enrichment.ocr.model_integrity fetch `
        --lock $ModelsJson `
        --weights-root $WeightsRoot
    if ($LASTEXITCODE -ne 0) {
        Write-Error "OCR model fetch failed closed."
        exit $LASTEXITCODE
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
