param (
    [string]$ModelsJson = "docker\ocr\model_sources.lock.json",
    [string]$WeightsRoot = "artifacts\_ocr_models"
)

if (-not (Test-Path $WeightsRoot)) {
    New-Item -ItemType Directory -Force -Path $WeightsRoot | Out-Null
}

$models = Get-Content $ModelsJson | ConvertFrom-Json

foreach ($key in $models.psobject.properties.name) {
    $model = $models.$key
    $url = $model.url
    $filename = $model.filename
    $dirName = $model.dir

    $modelDir = Join-Path $WeightsRoot $dirName
    if (-not (Test-Path $modelDir)) {
        New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
    }

    $dest = Join-Path $modelDir $filename

    if (-not (Test-Path $dest)) {
        Write-Host "Downloading $key -> $dest"
        try {
            Invoke-WebRequest -Uri $url -OutFile $dest -ErrorAction Stop
            Write-Host "Successfully downloaded $key."
        } catch {
            Write-Error "Failed to download $key from $url. Error: $_"
            if (Test-Path $dest) { Remove-Item $dest -Force }
            exit 1
        }
    } else {
        Write-Host "$dest already exists."
    }
}
