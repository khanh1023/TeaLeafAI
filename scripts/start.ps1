param([switch]$WithBackend)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Run-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE)." }
}

$oldMode = $env:VITE_INFERENCE_MODE
$oldUrl = $env:VITE_API_URL
try {
    $env:VITE_INFERENCE_MODE = "browser"
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw "Install Node.js 20.19+ or 22+ from https://nodejs.org and reopen this terminal." }
    Run-Checked "node" @("-e", "const [a,b]=process.versions.node.split('.').map(Number);process.exit(a>20||(a===20&&b>=19)?0:1)")
    $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
    if (-not (Test-Path "$root/frontend/node_modules/.bin/vite.cmd")) {
        Run-Checked $npm @("ci", "--prefix", "$root/frontend")
    }
    if ($WithBackend) {
        $python = "$root/backend/.venv/Scripts/python.exe"
        if (-not (Test-Path $python)) {
            if (Get-Command py -ErrorAction SilentlyContinue) {
                Run-Checked "py" @("-3", "-m", "venv", "$root/backend/.venv")
            } elseif (Get-Command python -ErrorAction SilentlyContinue) {
                Run-Checked "python" @("-m", "venv", "$root/backend/.venv")
            } else { throw "Install Python 3.11 or 3.12 from https://python.org for the optional backend." }
        }
        Run-Checked $python @("-m", "pip", "install", "-r", "$root/backend/requirements.txt")
        $backend = Start-Process -FilePath $python -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "9000") -WorkingDirectory "$root/backend" -PassThru -NoNewWindow
        $env:VITE_API_URL = "http://127.0.0.1:9000"
        $env:VITE_INFERENCE_MODE = "api"
    }
    Write-Host "Opening TeaLeafAI at http://localhost:5173. Keep this window open; press Ctrl+C to stop."
    Set-Location "$root/frontend"
    Run-Checked $npm @("run", "dev", "--", "--port", "5173", "--strictPort", "--open")
} finally {
    $env:VITE_INFERENCE_MODE = $oldMode
    $env:VITE_API_URL = $oldUrl
    if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue }
}
