param (
    [switch]$NoCache
)

$dockerDir = "docker\ocr"

$buildArgs = @()
if ($NoCache) {
    $buildArgs += "--no-cache"
}

Write-Host "Building DeepSolo image..."
# Notice context is $dockerDir so it can access local files if needed
docker build $buildArgs -t mvr-ocr-deepsolo:task04b -f "$dockerDir\deepsolo.Dockerfile" $dockerDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build DeepSolo image."
    exit $LASTEXITCODE
}

Write-Host "Building PARSeq-Vietnamese image..."
docker build $buildArgs -t mvr-ocr-parseq-vi:task04b -f "$dockerDir\parseq_vi.Dockerfile" $dockerDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build PARSeq-Vietnamese image."
    exit $LASTEXITCODE
}

Write-Host "Successfully built OCR Docker images."
