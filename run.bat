@echo off
setlocal enabledelayedexpansion
title ShadowKYC Platform

echo ===================================================
echo           Starting ShadowKYC Platform
echo ===================================================
echo Backend API : http://127.0.0.1:8000
echo Frontend UI : http://127.0.0.1:8000
echo API Docs    : http://127.0.0.1:8000/docs
echo ===================================================
echo.

cd /d "%~dp0ShadowKYC2-main\ShadowKYC2-main"
if exist "%~dp0ShadowKYC2-main\venv\Scripts\python.exe" (
    "%~dp0ShadowKYC2-main\venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
) else (
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
)

pause

