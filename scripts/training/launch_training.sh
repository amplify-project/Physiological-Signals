#!/bin/bash
# Launch action transformer training with nohup for VPN disconnect resilience

cd ~/concert_engagement
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/training_${TIMESTAMP}.log"

echo "🚀 Launching training on 8 GPUs with DataParallel..."
echo "📝 Log file: $LOGFILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Launch training with nohup
nohup python scripts/train_action_transformer_dataparallel.py \
    --features-dir data/processed/features_kinetics700 \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.0001 \
    > "$LOGFILE" 2>&1 &

TRAIN_PID=$!
echo "✅ Training started with PID: $TRAIN_PID"
echo "📊 Monitor with: tail -f $LOGFILE"
echo ""
echo "Showing first few lines of log..."
sleep 3
tail -20 "$LOGFILE"
