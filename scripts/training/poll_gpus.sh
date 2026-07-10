#!/bin/bash
# One-shot GPU + training poll for the Young Families run.
# Prints a timestamp, per-GPU util/mem/temp, and whether the training PID
# is alive plus the last log lines. Intended to be called ~every 5 min.
#
#   bash scripts/poll_gpus.sh

cd ~/concert_engagement

echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits \
    | awk -F',' '{printf "GPU %2s | util %3s%% | mem %5s/%5s MiB | %s C\n", $1,$2,$3,$4,$5}'

PIDFILE="logs/young_families.pid"
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Training PID $PID: RUNNING"
    else
        echo "Training PID $PID: NOT running (finished or died)"
    fi
fi

LATEST_LOG=$(ls -t logs/train_young_families_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "--- last 8 lines of $LATEST_LOG ---"
    tail -n 8 "$LATEST_LOG"
fi
