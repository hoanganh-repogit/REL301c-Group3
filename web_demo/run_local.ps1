$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot '.venv-web-demo'
$pythonPath = Join-Path $venvPath 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host 'Creating Python 3.11 virtual environment...'
    if ($env:PYTHON) {
        $bootstrapPython = $env:PYTHON
        if (-not (Test-Path -LiteralPath $bootstrapPython -PathType Leaf)) {
            throw "The PYTHON environment variable does not point to an executable file: $bootstrapPython"
        }
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $installedPythons = & py -0p
        $python311 = $installedPythons |
            Where-Object { $_ -match '^\s*(?:-V:3\.11|-3\.11(?:-\d+)?)(?:\s+\*)?\s+' } |
            Select-Object -First 1
        if ($python311) {
            $bootstrapPython = ($python311 -replace '^\s*(?:-V:3\.11|-3\.11(?:-\d+)?)(?:\s+\*)?\s+', '').Trim()
        } else {
            throw "Python 3.11 was not found. Install it with 'winget install --id Python.Python.3.11 -e', then open a new PowerShell window and rerun this script. If Python 3.11 is installed in Conda, set PYTHON to its python.exe path."
        }
    } else {
        throw "The Python launcher 'py' was not found. Install Python 3.11 with 'winget install --id Python.Python.3.11 -e', then open a new PowerShell window and rerun this script."
    }
    & $bootstrapPython -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the virtual environment with '$bootstrapPython'. Verify that it is a working Python 3.11 executable."
    }
}

& $pythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw 'Could not upgrade pip in the web demo virtual environment.'
}
& $pythonPath -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) {
    throw 'Could not install web demo dependencies. Review the pip error above and retry.'
}

Write-Host 'Open http://127.0.0.1:8000 in a browser. Press Ctrl+C to stop.'
Set-Location $projectRoot
& $pythonPath -m uvicorn web_demo.app:app --host 0.0.0.0 --port 8000
