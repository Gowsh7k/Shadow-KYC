$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workDir = Join-Path $scriptDir "ShadowKYC2-main\ShadowKYC2-main"
$venvPython = Join-Path $scriptDir "ShadowKYC2-main\venv\Scripts\python.exe"

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "          Starting ShadowKYC Platform              " -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Backend API : http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "Frontend UI : http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "API Docs    : http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location -Path $workDir

if (Test-Path $venvPython) {
    & $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
} else {
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
}

