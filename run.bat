@echo off
setlocal enabledelayedexpansion
title MARK XL Launcher

echo ===================================================
echo   🤖 MARK XL — Local AI Assistant Launcher
echo ===================================================
echo.

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your system PATH.
    echo Please install Python 3.11 or 3.12 (remember to check "Add Python to PATH").
    echo.
    pause
    exit /b 1
)

:: 2. Check and setup virtual environment
if not exist .venv (
    echo [SYSTEM] Virtual environment (.venv) not found. Creating one...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        echo Please ensure you have full write permissions and Python is installed correctly.
        echo.
        pause
        exit /b 1
    )
    echo [SYSTEM] Virtual environment created successfully.
    echo.
)

:: 3. Activate virtual environment
echo [SYSTEM] Activating virtual environment...
call .venv\Scripts\activate.bat

:: 4. Upgrade pip and install/verify base dependencies
echo [SYSTEM] Verifying pip and launching application...
python -m pip install --upgrade pip --quiet

:: 5. Launch the application
:: Note: main.py handles base package bootstrapping (PyQt6, sounddevice, etc.)
:: and checks/installs engine packages dynamically on startup.
python main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
)
