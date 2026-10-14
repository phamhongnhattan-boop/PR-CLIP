param(
    [ValidateSet('rsitmd', 'rsicd')][string]$Dataset = 'rsitmd',
    [string]$Checkpoint = '',
    [switch]$InitializeFromOpenAI,
    [int]$BatchSize = 0,
    [int]$Epochs = 0,
    [switch]$NoSaveCheckpoints
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$work = (Resolve-Path (Join-Path $repo '..\..\work')).Path
$python = Join-Path $work 'venv\Scripts\python.exe'
$images = Join-Path $work 'images\image_all'
$cache = Join-Path $work 'openclip-cache'
$trainCsv = Join-Path $repo 'dataset\train\ret2.csv'
$valCsv = Join-Path $repo "dataset\test\${Dataset}_val.csv"

if (-not (Test-Path -LiteralPath $python)) { throw "Python environment is missing: $python" }
if (-not (Test-Path -LiteralPath $images)) { throw "Image directory is missing: $images" }
if (-not (Test-Path -LiteralPath $trainCsv)) { throw "RET-2 CSV is missing: $trainCsv" }
if (-not (Test-Path -LiteralPath $valCsv)) { throw "Validation CSV is missing: $valCsv" }
if ($InitializeFromOpenAI -and $Checkpoint) {
    throw 'Choose either -InitializeFromOpenAI or -Checkpoint.'
}
if (-not $InitializeFromOpenAI) {
    if (-not $Checkpoint) { $Checkpoint = Join-Path $repo 'cache\weights\RET-2.pt' }
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "RET-2.pt is missing: $Checkpoint. Pass -InitializeFromOpenAI to train with different initialization."
    }
}

if ($Dataset -eq 'rsitmd') {
    $paperBatch = 100
    $paperEpochs = 7
    $paperLr = '5e-6'
} else {
    $paperBatch = 160
    $paperEpochs = 20
    $paperLr = '5e-5'
    Write-Warning 'The released rsicd_val.csv is byte-identical to rsicd_test.csv. Do not select a final model on this file and then report its test score as independent.'
}
if ($BatchSize -le 0) { $BatchSize = $paperBatch }
if ($Epochs -le 0) { $Epochs = $paperEpochs }
if ($BatchSize -ne $paperBatch -or $Epochs -ne $paperEpochs) {
    Write-Warning 'Batch size or epoch count differs from the paper; retrieval scores are not directly comparable.'
}

$env:PYTHONPATH = Join-Path $work 'deps'
$env:HF_HOME = Join-Path $work 'hf-cache'
$env:TEMP = Join-Path $work 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'

$runName = "prclip-paper-$Dataset"
$argsList = @(
    '-m', 'training.main',
    '--train-data', $trainCsv, '--images-dir', $images,
    '--csv-separator', ',', '--csv-img-key', 'filename', '--csv-caption-key', 'title',
    '--retrieval-data', $valCsv, '--retrieval-images-dir', $images,
    '--retrieval-csv-separator', ',', '--retrieval-csv-img-key', 'filename',
    '--retrieval-csv-caption-key', 'title', '--retrieval-frequency', '1',
    '--epochs', "$Epochs", '--batch-size', "$BatchSize", '--workers', '0',
    '--lr', $paperLr, '--warmup', '100', '--weight_decay', '0.5', '--max-grad-norm', '50',
    '--amp-init-scale', '1',
    '--image-model', 'ViT-B-32', '--image-model-builder', 'openclip',
    '--text-model', 'ViT-B-32', '--text-model-builder', 'openclip',
    '--pretrained-image-model', '--pretrained-text-model', '--loss', 'InfoNCE',
    '--name', $runName, '--cache-dir', $cache,
    '--logs', (Join-Path $work 'official-runs'), '--prclip-objective', 'paper'
)
if ($InitializeFromOpenAI) {
    $argsList += '--no-load-clip-weights'
} else {
    $argsList += @('--load-clip-weights-from', (Resolve-Path -LiteralPath $Checkpoint).Path)
}
if ($NoSaveCheckpoints) { $argsList += '--no-save-checkpoints' }

Push-Location $repo
try {
    & $python @argsList
    if ($LASTEXITCODE -ne 0) { throw "PR-CLIP training failed with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
