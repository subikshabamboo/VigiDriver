@echo off
title VigiDriver - Real-Time Driver Drowsiness Monitoring System
color 0A
cls

echo ===============================================================================
echo                VigiDriver: Deep Learning Safety-Critical ADAS                 
echo          Test & Evaluation Verified Architecture (DoD/CDAO Framework)         
echo ===============================================================================
echo.
echo [*] Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH!
    echo Please install Python 3.10+ and re-run.
    pause
    exit /b 1
)

echo [*] Launching VigiDriver Full-Stack Server on port 8000...
echo [*] Interactive Dashboard: http://127.0.0.1:8000
echo [*] Swagger API Docs:      http://127.0.0.1:8000/docs
echo.
echo [*] Opening default web browser in 3 seconds...
start "" timeout /t 3 /nobreak >nul & start http://127.0.0.1:8000

echo [*] Starting backend server (Press Ctrl+C to stop)...
echo.
python backend/run.py --port 8000
pause
