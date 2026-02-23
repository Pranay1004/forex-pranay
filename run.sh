#!/bin/bash
# run.sh — Launcher for ForexBot

# Ensure we're in the project root
cd "$(dirname "$0")"

# MacOS specific fix for library loading (OpenFOAM conflicts etc)
unset DYLD_LIBRARY_PATH
unset LD_LIBRARY_PATH

# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Use python3 (assuming python 3.10+ is available as python3 or python)
PYTHON_CMD="python3"
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
fi

echo "Starting ForexBot with $PYTHON_CMD..."
exec $PYTHON_CMD forexbot/main.py "$@"
