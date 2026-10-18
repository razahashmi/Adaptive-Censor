$ErrorActionPreference = 'Stop'
$appDirectory = $PSScriptRoot
$workspaceDirectory = Split-Path (Split-Path $appDirectory -Parent) -Parent
$runtimePython = Join-Path $appDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) {
    $runtimePython = Join-Path $workspaceDirectory 'work\venv\Scripts\python.exe'
    if (-not $env:ADAPTIVE_CENSOR_WORKDIR) { $env:ADAPTIVE_CENSOR_WORKDIR = Join-Path $workspaceDirectory 'work\adaptive-censor' }
}
& $runtimePython (Join-Path $appDirectory 'launcher.py') stop
