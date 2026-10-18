param([switch]$Gpu, [switch]$SceneVerifier, [switch]$Learned, [switch]$Training, [switch]$Segmentation, [switch]$SkipModels, [ValidateSet('4B','8B')][string]$SceneModel='4B', [ValidateSet('tiny.en','small.en')][string]$SpeechModel='tiny.en')
$ErrorActionPreference = 'Stop'
if ($SkipModels -and ($SceneVerifier -or $Segmentation)) { throw 'SkipModels cannot be combined with SceneVerifier or Segmentation downloads.' }
$appDirectory = $PSScriptRoot
$environmentDirectory = Join-Path $appDirectory '.venv'
$runtimePython = Join-Path $environmentDirectory 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) {
    python -m venv $environmentDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11-3.13 and retry.' }
}
$dependencyFile = Join-Path $appDirectory 'requirements.txt'
$cpuLockFile = Join-Path $appDirectory 'requirements-cpu-win-py313.lock.txt'
$runtimeTarget = & $runtimePython -c "import sys; print('win313' if sys.platform == 'win32' and sys.version_info[:2] == (3, 13) else 'other')"
if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect the Python environment.' }
if ($runtimeTarget -eq 'win313' -and (Test-Path -LiteralPath $cpuLockFile)) { $dependencyFile = $cpuLockFile }
# CPU and GPU wheels overwrite the same module. Uninstall both before the CPU
# dependency install when switching from GPU, or pip can retain CPU metadata for
# files removed by the GPU uninstall. Do not repair a running service in place.
$hasGpuRuntime = & $runtimePython -c "import importlib.metadata as m; print(any(d.metadata['Name'].lower() == 'onnxruntime-gpu' for d in m.distributions()))"
if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect the inference packages.' }
if ($hasGpuRuntime -eq 'True') {
    & $runtimePython -m pip uninstall -y onnxruntime onnxruntime-gpu
    if ($LASTEXITCODE -ne 0) { throw 'Unable to remove overlapping inference packages.' }
}
& $runtimePython -m pip install --disable-pip-version-check --no-cache-dir -r $dependencyFile
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if ($Learned) {
    & $runtimePython -m pip install --disable-pip-version-check -r (Join-Path $appDirectory 'requirements-learned.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Learned runtime dependency installation failed.' }
}
if ($Training -or $Segmentation) {
    $trainingDirectory = Join-Path $appDirectory '.train-venv'
    $trainingPython = Join-Path $trainingDirectory 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $trainingPython)) {
        python -m venv $trainingDirectory
        if ($LASTEXITCODE -ne 0) { throw 'Training environment creation failed.' }
    }
    & $trainingPython -m pip install --disable-pip-version-check torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'Training torch installation failed.' }
    & $trainingPython -m pip install --disable-pip-version-check -r (Join-Path $appDirectory 'requirements-training.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed.' }
    if ($Segmentation) {
        & $trainingPython -m pip install --disable-pip-version-check -r (Join-Path $appDirectory 'requirements-segmentation.txt') --index-url https://download.pytorch.org/whl/cu128
        if ($LASTEXITCODE -ne 0) { throw 'Segmentation image processor installation failed.' }
    }
}
if ($Gpu) {
    # Both ORT distributions share a namespace; the GPU wheel must be installed last.
    & $runtimePython -m pip uninstall -y onnxruntime
    if ($LASTEXITCODE -ne 0) { throw 'Unable to switch to the GPU inference package.' }
    & $runtimePython -m pip install --disable-pip-version-check --no-cache-dir --force-reinstall -r (Join-Path $appDirectory 'requirements-gpu.txt') -c (Join-Path $appDirectory 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'GPU dependency installation failed.' }
}
if (-not $env:ADAPTIVE_CENSOR_WORKDIR) { $env:ADAPTIVE_CENSOR_WORKDIR = Join-Path $appDirectory '.runtime' }
if (-not $SkipModels) {
    $modelArguments = @((Join-Path $appDirectory 'model_setup.py'), '--speech-model', $SpeechModel)
    if ($Gpu) { $modelArguments += '--gpu' }
    & $runtimePython @modelArguments
    if ($LASTEXITCODE -ne 0) { throw 'Model setup failed. Retry when network access is available.' }
}
if ($SceneVerifier) {
    & $runtimePython (Join-Path $appDirectory 'setup_verifier.py') --model $SceneModel
    if ($LASTEXITCODE -ne 0) { throw 'Scene verifier installation failed.' }
}
if ($Segmentation) {
    & $trainingPython (Join-Path $appDirectory 'setup_segmentation.py')
    if ($LASTEXITCODE -ne 0) { throw 'Segmentation model installation failed.' }
}
Write-Host 'Ready. Run Start.ps1 to open Adaptive Censor.'
if ($SkipModels) { Write-Host 'Separate model downloads skipped. Try the generated controls demo and Check setup in the player.' }
