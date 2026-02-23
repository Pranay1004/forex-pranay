@echo off
cd /d "%~dp0"
set DYLD_LIBRARY_PATH=
set LD_LIBRARY_PATH=
set PYTHONPATH=%~dp0;%PYTHONPATH%

echo Starting ForexBot web frontend...
python -m streamlit run forexbot\web\app.py --server.port 8501 --server.headless true
pause
