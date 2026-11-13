@echo off
setlocal enabledelayedexpansion
title ShadowKYC Server

echo ===================================================
echo           Starting ShadowKYC Platform
echo ===================================================
echo Backend API : http://127.0.0.1:8000
echo Frontend UI : http://127.0.0.1:8000
echo API Docs    : http://127.0.0.1:8000/docs
echo ===================================================

cd /d "%~dp0"
if exist "..\venv\Scripts\python.exe" (
    "..\venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
) else (
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
)
pause
