param([int]$Port = 8502, [switch]$NoBrowser, [switch]$Foreground)
$ErrorActionPreference = 'Stop'
$appDirectory = $PSScriptRoot
$workspaceDirectory = Split-Path (Split-Path $appDirectory -Parent) -Parent
$preparedPython = Join-Path $workspaceDirectory 'work\venv\Scripts\python.exe'
$localPython = Join-Path $appDirectory '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $localPython) {
    $runtimePython = $localPython
    if (-not $env:ADAPTIVE_CENSOR_WORKDIR) { $env:ADAPTIVE_CENSOR_WORKDIR = Join-Path $appDirectory '.runtime' }
} elseif (Test-Path -LiteralPath $preparedPython) {
    $runtimePython = $preparedPython
    if (-not $env:ADAPTIVE_CENSOR_WORKDIR) { $env:ADAPTIVE_CENSOR_WORKDIR = Join-Path $workspaceDirectory 'work\adaptive-censor' }
} else {
    & (Join-Path $appDirectory 'Setup.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Setup failed.' }
    $runtimePython = $localPython
}
if ($Foreground) {
    & $runtimePython (Join-Path $appDirectory 'server.py') --port $Port
} else {
    $launcherArguments = @((Join-Path $appDirectory 'launcher.py'), 'start', '--port', "$Port")
    if ($NoBrowser) { $launcherArguments += '--no-browser' }
    & $runtimePython @launcherArguments
}
if ($LASTEXITCODE -ne 0) { throw 'Application startup failed. See the runtime server.log file.' }
