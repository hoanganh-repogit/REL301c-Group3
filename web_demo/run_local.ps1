$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot '.venv-web-demo'
$pythonPath = Join-Path $venvPath 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host 'Creating Python 3.11 virtual environment...'
    if ($env:PYTHON) {
        $bootstrapPython = $env:PYTHON
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $bootstrapPython = (& py -3.11 -c "import sys; print(sys.executable)").Trim()
    } else {
        throw 'Python 3.11 was not found. Install it, or set the PYTHON environment variable to its executable.'
    }
    if (-not $bootstrapPython) {
        throw 'Python 3.11 was not found. Install it, or set the PYTHON environment variable to its executable.'
    }
    & $bootstrapPython -m venv $venvPath
}

& $pythonPath -m pip install --upgrade pip
& $pythonPath -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')

Write-Host 'Open http://127.0.0.1:8000 in a browser. Press Ctrl+C to stop.'
Set-Location $projectRoot
& $pythonPath -m uvicorn web_demo.app:app --host 0.0.0.0 --port 8000
