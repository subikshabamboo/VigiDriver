# VigiDriver - One-Click PowerShell Launcher
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "               VigiDriver: Deep Learning Safety-Critical ADAS                  " -ForegroundColor Yellow
Write-Host "         Test & Evaluation Verified Architecture (DoD/CDAO Framework)          " -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python was not found in PATH. Please install Python 3.10+." -ForegroundColor Red
    exit 1
}

Write-Host "[*] Launching VigiDriver Full-Stack Server on port 8000..." -ForegroundColor Green
Write-Host "[*] Interactive Dashboard: http://127.0.0.1:8000" -ForegroundColor White
Write-Host "[*] Swagger API Docs:      http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host ""

Start-Job -ScriptBlock {
    Start-Sleep -Seconds 3
    Start-Process "http://127.0.0.1:8000"
} | Out-Null

python backend/run.py --port 8000
