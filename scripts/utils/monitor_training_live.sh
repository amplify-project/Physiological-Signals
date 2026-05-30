#!/bin/bash
# Live training monitor - updates every 5 seconds
# Shows: GPU usage, process status, latest log lines

LOG_FILE="$HOME/concert_engagement/logs/training_ddp_8gpus.log"

while true; do
    clear
    echo "════════════════════════════════════════════════════════════════"
    echo "🔥 TRAINING MONITOR - $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    # GPU Status
    echo "📊 GPU STATUS:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader | head -8
    echo ""
    
    # Process Status
    echo "🔧 TRAINING PROCESSES:"
    ps aux | grep -E 'train_action_transformer|torchrun' | grep -v grep | awk '{printf "PID: %-8s CPU: %-5s MEM: %-5s CMD: %s\n", $2, $3"%", $4"%", $11}' | head -10
    PROC_COUNT=$(ps aux | grep 'train_action_transformer_ddp' | grep -v grep | wc -l)
    echo "Total processes: $PROC_COUNT"
    echo ""
    
    # Latest Log Lines
    echo "📝 LATEST LOG (last 15 lines):"
    if [ -f "$LOG_FILE" ]; then
        tail -15 "$LOG_FILE" | sed 's/^/  /'
    else
        echo "  [Log file not found: $LOG_FILE]"
    fi
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "Press Ctrl+C to exit | Refreshing in 5 seconds..."
    
    sleep 5
done
