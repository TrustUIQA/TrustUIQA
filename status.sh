#!/bin/bash
# Pilot status: running processes, finished metrics, last log lines.  Usage: ./status.sh
cd "$(dirname "$0")"
pgrep -af "run_pilot|score.py" | grep -v pgrep || echo "(pilot not running)"
echo "finished metrics: $(ls out/scores 2>/dev/null | wc -l)/22"
tail -c 2000 pilot.log | tr '\r' '\n' | grep -v '^$' | tail -6
