<#
.SYNOPSIS
    Start the MVR retrieval backend against an already-built semantic index.

.DESCRIPTION
    Every path is a parameter. The previous version hardcoded one machine's
    D:\AIC26 layout, so it only ever ran there -- and its dataset guard was a
    bare Test-Path, which an empty directory satisfies, so it could export
    MVR_KEYFRAME_DIR pointing at nothing and start a backend that serves
    nothing.

.EXAMPLE
    .\scripts\run_backend_full.ps1 -DatasetRoot E:\AIC26_data `
        -Index .\artifacts\indexes\qna_organizer_all
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,

    [Parameter(Mandatory = $true)]
    [string]$Index,

    [string]$KeyframeDir,
    [string]$VideoDir,

    # Without this, KeyframeFrameIndexResolver is never built, every candidate
    # fails to resolve to an original frame index, and KIS/TRAKE answer
    # 200 OK with zero results -- which reads like "nothing matched" rather
    # than "the backend was misconfigured".
    [string]$KeyframeMappingCsv,
    # Bare "python" resolves to whatever is first on PATH, which on a machine
    # with a system Python is not the venv that has uvicorn installed.
    [string]$Python = "python",
    [string]$Device = "cpu",
    [string]$SemanticBackend = "faiss_reference",
    [string]$BindAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

function Assert-PopulatedDirectory {
    param([string]$Path, [string]$Label)

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label not found: $Path"
    }
    # Test-Path alone is satisfied by an empty directory -- and by one holding
    # nothing but a .gitkeep, which is exactly what data/raw is in a fresh
    # clone. That is how this script used to pass its dataset check and then
    # start a backend with nothing behind it.
    $entries = Get-ChildItem -LiteralPath $Path -Force |
        Where-Object { $_.Name -notin @(".gitkeep", ".gitignore") }
    if (-not ($entries | Select-Object -First 1)) {
        throw "$Label holds no dataset (placeholder files only): $Path"
    }
}

Assert-PopulatedDirectory -Path $DatasetRoot -Label "Dataset root"

$manifest = Join-Path $Index "manifest.json"
if (-not (Test-Path -LiteralPath $manifest)) {
    Write-Host "No index manifest at: $manifest" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Build one first:" -ForegroundColor Cyan
    Write-Host "$Python -m mvr.cli build-aic-index --dataset-root `"$DatasetRoot`" --output `"$Index`" --device $Device"
    exit 2
}

if (-not $KeyframeDir) { $KeyframeDir = Join-Path $DatasetRoot "keyframes" }
if (-not $VideoDir) { $VideoDir = Join-Path $DatasetRoot "video" }

# Warn rather than throw: the raw organizer download nests these under
# Keyframes_L26_a\keyframes and Videos_L21_a\video, and the operator can point
# at them with -KeyframeDir / -VideoDir. Retrieval still works without them;
# only the thumbnail and video URLs go missing.
foreach ($asset in @(
        @{ Path = $KeyframeDir; Label = "Keyframe directory"; Flag = "-KeyframeDir" },
        @{ Path = $VideoDir; Label = "Video directory"; Flag = "-VideoDir" })) {
    if (-not (Test-Path -LiteralPath $asset.Path -PathType Container)) {
        Write-Host "$($asset.Label) not found: $($asset.Path)" -ForegroundColor Yellow
        Write-Host "  media URLs will 404; pass $($asset.Flag) for a non-standard layout." -ForegroundColor Yellow
    }
}

if ($KeyframeMappingCsv) {
    if (-not (Test-Path -LiteralPath $KeyframeMappingCsv -PathType Leaf)) {
        throw "Keyframe mapping CSV not found: $KeyframeMappingCsv"
    }
    $env:MVR_KEYFRAME_MAPPING_CSV = $KeyframeMappingCsv
}
else {
    Write-Host "No -KeyframeMappingCsv given." -ForegroundColor Yellow
    Write-Host "  KIS and TRAKE will return 200 OK with zero results: no candidate" -ForegroundColor Yellow
    Write-Host "  can be resolved to an original frame index without the mapping." -ForegroundColor Yellow
}

$env:MVR_AIC_DATASET_ROOT = $DatasetRoot
$env:MVR_KEYFRAME_DIR = $KeyframeDir
$env:MVR_VIDEO_DIR = $VideoDir
$env:MVR_SEMANTIC_BACKEND = $SemanticBackend
$env:MVR_SEMANTIC_INDEX = $Index
$env:MVR_DEVICE = $Device

Write-Host "dataset root : $DatasetRoot"
Write-Host "index        : $Index"
Write-Host "keyframes    : $KeyframeDir"
Write-Host "videos       : $VideoDir"
if ($KeyframeMappingCsv) { Write-Host "mapping      : $KeyframeMappingCsv" }
Write-Host "serving      : http://${BindAddress}:$Port"

& $Python -m uvicorn mvr.api.app:app --host $BindAddress --port $Port
