param(
    [Parameter(Mandatory = $true)][string]$MappingCsv,
    [Parameter(Mandatory = $true)][string]$KeyframeRoot,
    [string]$VideoMetadataCsv,
    [string]$VideoRoot,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$EmbeddingProviders = "beit3,metaclip2,perception_encoder",
    [string]$ProviderSpecs = "configs/embedding_providers.json",
    [string]$Beit3Checkpoint,
    [string]$MetaClip2Model,
    [string]$PerceptionEncoderCheckpoint,
    [int]$EmbeddingBatchSize = 32,
    [string]$Device = "auto",
    [switch]$AllowModelDownloads
)

$ErrorActionPreference = "Stop"

$runnerArgs = @(
    "scripts/run_feature_extraction.py",
    "--mapping-csv", $MappingCsv,
    "--keyframe-root", $KeyframeRoot,
    "--output", $OutputRoot,
    "--embedding-providers", $EmbeddingProviders,
    "--provider-specs", $ProviderSpecs,
    "--embedding-batch-size", $EmbeddingBatchSize,
    "--device", $Device,
    "--skip-objects"
)

if ($Beit3Checkpoint) {
    $runnerArgs += @("--beit3-checkpoint", $Beit3Checkpoint)
}
if ($MetaClip2Model) {
    $runnerArgs += @("--metaclip2-model", $MetaClip2Model)
}
if ($PerceptionEncoderCheckpoint) {
    $runnerArgs += @("--perception-encoder-checkpoint", $PerceptionEncoderCheckpoint)
}
if ($AllowModelDownloads) {
    $runnerArgs += "--allow-model-downloads"
}

# Reserved for later full-pipeline stages; accepted now to keep one stable CLI.
$null = $VideoMetadataCsv
$null = $VideoRoot

& python @runnerArgs
if ($LASTEXITCODE -ne 0) {
    throw "Feature extraction failed with exit code $LASTEXITCODE"
}
