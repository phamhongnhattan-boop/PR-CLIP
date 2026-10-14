param(
    [ValidateSet('paper', 'released')][string]$Objective = 'paper',
    [string]$Checkpoint = '',
    [switch]$FullValidation,
    [ValidateSet('rsitmd', 'rsicd')][string]$Dataset = 'rsitmd',
    [int]$TrainImages = 2
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$work = (Resolve-Path (Join-Path $repo '..\..\work')).Path
$python = Join-Path $work 'venv\Scripts\python.exe'
$images = Join-Path $work 'images\image_all'
$cache = Join-Path $work 'openclip-cache'
$trainCsv = Join-Path $work 'tmp\ret2_official_smoke.csv'
$fullValSource = Join-Path $repo "dataset\test\${Dataset}_val.csv"
$valCsv = Join-Path $work "tmp\${Dataset}_official_val_smoke.csv"
$env:PYTHONPATH = Join-Path $work 'deps'
$env:HF_HOME = Join-Path $work 'hf-cache'
$env:TEMP = Join-Path $work 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'

Push-Location $repo
try {
    & $python -m scripts.prepare_official_smoke `
        --train-csv dataset/train/ret2.csv --val-csv $fullValSource `
        --train-output $trainCsv --val-output $valCsv --train-images $TrainImages
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare smoke CSVs' }

    if ($FullValidation) {
        $valCsv = $fullValSource
    }
    $runName = if ($FullValidation) { "official-$Objective-$Dataset-full-val-$TrainImages" } else { "official-$Objective-$Dataset-smoke-$TrainImages" }
    $argsList = @(
        '-m', 'training.main',
        '--train-data', $trainCsv, '--images-dir', $images,
        '--csv-separator', ',', '--csv-img-key', 'filename', '--csv-caption-key', 'title',
        '--retrieval-data', $valCsv, '--retrieval-images-dir', $images,
        '--retrieval-csv-separator', ',', '--retrieval-csv-img-key', 'filename',
        '--retrieval-csv-caption-key', 'title', '--retrieval-frequency', '1',
        '--epochs', '1', '--no-save-checkpoints', '--batch-size', '2', '--workers', '0',
        '--lr', '5e-6', '--warmup', '1', '--weight_decay', '0.5', '--max-grad-norm', '50',
        '--amp-init-scale', '1',
        '--image-model', 'ViT-B-32', '--image-model-builder', 'openclip',
        '--text-model', 'ViT-B-32', '--text-model-builder', 'openclip',
        '--pretrained-image-model', '--pretrained-text-model', '--loss', 'InfoNCE',
        '--name', $runName, '--cache-dir', $cache,
        '--logs', (Join-Path $work 'official-runs'), '--prclip-objective', $Objective
    )
    if ($Checkpoint) {
        $argsList += @('--load-clip-weights-from', $Checkpoint)
    } else {
        $argsList += '--no-load-clip-weights'
    }
    & $python @argsList
    if ($LASTEXITCODE -ne 0) { throw "ItraModel smoke run failed with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
