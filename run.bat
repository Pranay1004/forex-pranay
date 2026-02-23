@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  run.bat  —  ForexBot Windows Launcher
REM  Runs in the FOREGROUND.  Press Ctrl+C to stop cleanly.
REM  GPU:  automatically uses NVIDIA CUDA (T1000, RTX, etc.) if detected via
REM        nvidia-smi.  Falls back to CPU if no GPU is found.
REM ─────────────────────────────────────────────────────────────────────────────

REM Change to the directory containing this script (project root)
cd /d "%~dp0"

REM Clear Mac-specific library vars (harmless no-ops on Windows)
set DYLD_LIBRARY_PATH=
set LD_LIBRARY_PATH=

REM Add project root to PYTHONPATH so "import forexbot" resolves
set PYTHONPATH=%~dp0;%PYTHONPATH%

REM ── Check Python is available ────────────────────────────────────────────────
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] python not found in PATH.
    echo         Install Python 3.10+ from https://python.org and add it to PATH.
    pause
    exit /b 1
)

REM ── Show GPU status before starting ─────────────────────────────────────────
echo ─────────────────────────────────────────────────────────────────────────────
echo  ForexBot  —  Windows launch
echo ─────────────────────────────────────────────────────────────────────────────
nvidia-smi -L >nul 2>&1
if %ERRORLEVEL%==0 (
    echo  GPU detected — XGBoost + LightGBM will use CUDA acceleration.
    nvidia-smi -L
) else (
    echo  No NVIDIA GPU detected — running on CPU.
)
echo.
echo  Press Ctrl+C at any time to stop cleanly.
echo ─────────────────────────────────────────────────────────────────────────────
echo.

REM ── Launch ForexBot (foreground — this window IS the bot) ────────────────────
REM  Pass any extra arguments: e.g.  run.bat auto
python forexbot\main.py %*

REM ── Bot stopped ──────────────────────────────────────────────────────────────
echo.
echo ─────────────────────────────────────────────────────────────────────────────
echo  ForexBot stopped.
echo ─────────────────────────────────────────────────────────────────────────────
pause
