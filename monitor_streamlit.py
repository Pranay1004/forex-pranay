#!/usr/bin/env python3
"""
Continuous uptime and functionality monitor for Streamlit app.
Runs in background, pings the app every N seconds, logs results.
"""

import requests
import time
import json
from datetime import datetime
from pathlib import Path

# Configuration
STREAMLIT_URL = "https://forexp.streamlit.app"
CHECK_INTERVAL_SECONDS = 60  # Check every minute
LOG_FILE = Path(__file__).parent / "logs" / "streamlit_monitor.log"
STATS_FILE = Path(__file__).parent / "logs" / "streamlit_stats.json"

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_event(event_type: str, message: str, response_time: float = None, status_code: int = None):
    """Log monitoring events."""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "type": event_type,
        "message": message,
        "response_time_ms": response_time,
        "status_code": status_code,
    }
    
    # Append to log file
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    # Also print to stdout for visibility
    print(f"[{timestamp}] {event_type}: {message} (Status: {status_code}, Time: {response_time}ms)")


def check_streamlit_uptime():
    """Check if Streamlit app is reachable and measure response time."""
    try:
        start_time = time.time()
        response = requests.get(STREAMLIT_URL, timeout=15)
        elapsed_ms = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            # Check for key content indicators
            content = response.text.lower()
            if "forexbot" in content or "backtest" in content or "streamlit" in content:
                log_event("UP", f"Streamlit app is healthy", elapsed_ms, 200)
                return True
            else:
                log_event("PARTIAL", f"App loaded but content check failed", elapsed_ms, 200)
                return False
        else:
            log_event("DOWN", f"HTTP {response.status_code}", elapsed_ms, response.status_code)
            return False
    except requests.Timeout:
        log_event("TIMEOUT", f"Request timed out", None, None)
        return False
    except requests.ConnectionError:
        log_event("ERROR", f"Connection error", None, None)
        return False
    except Exception as e:
        log_event("ERROR", f"{type(e).__name__}: {str(e)}", None, None)
        return False


def update_stats(is_up: bool):
    """Update running statistics."""
    if not STATS_FILE.exists():
        stats = {"total_checks": 0, "successful": 0, "failed": 0, "uptime_pct": 0.0}
    else:
        with open(STATS_FILE, "r") as f:
            stats = json.load(f)
    
    stats["total_checks"] += 1
    if is_up:
        stats["successful"] += 1
    else:
        stats["failed"] += 1
    
    stats["uptime_pct"] = (stats["successful"] / stats["total_checks"]) * 100 if stats["total_checks"] > 0 else 0.0
    stats["last_check"] = datetime.now().isoformat()
    
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)


def monitor_loop():
    """Main monitoring loop."""
    print(f"Starting Streamlit uptime monitor for {STREAMLIT_URL}")
    print(f"Check interval: {CHECK_INTERVAL_SECONDS} seconds")
    print(f"Logs: {LOG_FILE}")
    print(f"Stats: {STATS_FILE}")
    print("-" * 80)
    
    log_event("START", f"Monitor started for {STREAMLIT_URL}", None, None)
    
    try:
        while True:
            is_up = check_streamlit_uptime()
            update_stats(is_up)
            time.sleep(CHECK_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        log_event("STOP", "Monitor stopped by user", None, None)
        print("\nMonitor stopped.")
    except Exception as e:
        log_event("FATAL", f"{type(e).__name__}: {str(e)}", None, None)
        raise


if __name__ == "__main__":
    monitor_loop()
