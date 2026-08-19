#!/bin/bash
# Combined server dashboard: GPU-by-user table + Family-Concert v1 training status.
# Run standalone or on a timer (e.g. watch -n 60 bash scripts/server_dashboard.sh).
cd ~/concert_engagement 2>/dev/null || exit 1

bash scripts/gpu_users.sh

echo
echo "--------------- Family-Concert v1 training ---------------"
PIDF=logs/family_concert.pid
L=$(ls -t logs/train_family_concert_*.log 2>/dev/null | head -1)
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "status: RUNNING (pid $(cat "$PIDF"))"
else
    echo "status: NOT RUNNING (finished / not started / crashed)"
fi
if [ -n "$L" ]; then
    echo "log: $L"
    tr '\r' '\n' < "$L" | grep -aE 'Epoch [0-9]+ total|Train: |Val: |[Bb]est|Saved|COMPLETE|Traceback|Error' | tail -6
else
    echo "(no training log found)"
fi
