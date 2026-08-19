#!/bin/bash
# One-shot progress snapshot for the Family-Concert v1 training run.
# Prints run status, the training GPUs, and the latest progress lines.
# Meant to be called once per refresh (the caller controls cadence).
cd ~/concert_engagement
GPUS="${GPUS:-3,4}"
PIDF=logs/family_concert.pid
LOG=$(ls -t logs/train_family_concert_*.log 2>/dev/null | head -1)

echo "==================== Family-Concert v1 ===================="
echo "time:   $(date '+%Y-%m-%d %H:%M:%S')"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "status: RUNNING (pid $(cat "$PIDF"))"
else
    echo "status: NOT RUNNING (finished, not started, or crashed)"
fi
echo "log:    $LOG"
echo "--- GPUs $GPUS (index, util%, mem, temp) ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader -i "$GPUS" 2>/dev/null
echo "--- latest progress (tail) ---"
if [ -n "$LOG" ]; then
    grep -aE 'Epoch|Train|Val|Acc|Loss|[Bb]est|Saved|Family-Concert|kept classes|num_classes|Error|Traceback|RuntimeError' "$LOG" | tail -16
else
    echo "(no log yet)"
fi
echo "==========================================================="
