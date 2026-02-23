#!/bin/bash
# run_web.sh — Launch ForexBot web frontend (IST + live chart + multi-TF backtest)

cd "$(dirname "$0")"
unset DYLD_LIBRARY_PATH
unset LD_LIBRARY_PATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

PYTHON_CMD="python3"
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
fi

echo "Starting ForexBot web frontend with $PYTHON_CMD..."
exec $PYTHON_CMD -m streamlit run forexbot/web/app.py --server.port 8501 --server.headless true
