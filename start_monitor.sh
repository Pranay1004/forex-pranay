#!/bin/bash
# Launch Streamlit monitor in background

set -e

cd "$(dirname "$0")"

# Kill any existing monitors
pkill -f monitor_streamlit.py 2>/dev/null || true
sleep 1

# Launch monitor in background
echo "Starting Streamlit uptime monitor..."
nohup /opt/homebrew/bin/python3.10 monitor_streamlit.py > logs/monitor_background.log 2>&1 &
MONITOR_PID=$!

echo "Monitor started with PID: $MONITOR_PID"
echo "Monitor running in background. Check logs/streamlit_monitor.log for details."
echo ""
echo "View live stats:"
echo "  cat logs/streamlit_stats.json"
echo ""
echo "View logs:"
echo "  tail -f logs/streamlit_monitor.log"
echo ""
echo "Stop monitor:"
echo "  pkill -f monitor_streamlit.py"
