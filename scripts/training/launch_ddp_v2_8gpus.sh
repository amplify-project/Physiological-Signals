#!/bin/bash
# Launch DDP V2 training on 8 GPUs with Gloo backend
# Full 100 epoch training on Kinetics-700 (86 classes, 65K samples)

set -e

echo "========================================================================"
echo "🚀 LAUNCHING 8-GPU DDP TRAINING (Gloo Backend)"
echo "========================================================================"
echo ""

# Configuration
FEATURES_DIR="data/processed/features_kinetics700"
INDICES_DIR="data/processed"
OUTPUT_DIR="models/action_transformer_kinetics700_ddp_v2"
LOG_FILE="logs/training_ddp_v2_8gpus.log"
EPOCHS=100
BATCH_SIZE=32
LR=0.0001

# Activate virtual environment
source venv/bin/activate

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Features: $FEATURES_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Effective batch size: $((BATCH_SIZE * 8))"
echo "  Learning rate: $LR"
echo "  GPUs: 8 (CUDA 0-7)"
echo ""

# Check if indices exist
if [ ! -f "$INDICES_DIR/train_indices.npy" ]; then
    echo "⚠️  Pre-computed indices not found. Generating..."
    python3 scripts/generate_dataset_indices.py \
        --features-dir "$FEATURES_DIR" \
        --output-dir "$INDICES_DIR"
    echo ""
fi

echo "Starting training in background with nohup..."
echo "Log file: $LOG_FILE"
echo ""

# Launch training with nohup
nohup torchrun --nproc_per_node=8 \
    scripts/train_action_transformer_ddp_v2.py \
    --features-dir "$FEATURES_DIR" \
    --indices-dir "$INDICES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    > "$LOG_FILE" 2>&1 &

PID=$!

echo "✅ Training launched!"
echo ""
echo "Process ID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "📊 Monitor with:"
echo "   tail -f $LOG_FILE"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "⏹️  To stop training:"
echo "   kill $PID"
echo "   # or"
echo "   pkill -9 -f train_action_transformer_ddp_v2"
echo ""
echo "⏱️  Expected duration: ~5-6 hours for 100 epochs"
echo "📁 Model checkpoints: $OUTPUT_DIR"
echo ""
echo "========================================================================"
