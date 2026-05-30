#!/bin/bash
# Quick 8-GPU test: 1 epoch to validate all GPUs work with Gloo backend

set -e

echo "========================================================================"
echo "🧪 DDP V2 TEST: 8 GPUs, 1 Epoch"
echo "========================================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

FEATURES_DIR="data/processed/features_kinetics700"
INDICES_DIR="data/processed"
LOG_FILE="logs/test_ddp_v2_8gpus_1epoch.log"

# Create logs directory
mkdir -p logs

# Check if features exist
if [ ! -d "$FEATURES_DIR" ]; then
    echo "❌ ERROR: Features directory not found: $FEATURES_DIR"
    exit 1
fi

# Check if indices exist
if [ ! -f "$INDICES_DIR/train_indices.npy" ]; then
    echo "📂 Generating pre-computed indices..."
    python3 scripts/generate_dataset_indices.py \
        --features-dir "$FEATURES_DIR" \
        --output-dir "$INDICES_DIR"
    echo ""
fi

echo "✅ Indices ready"
echo ""

echo "Launching 8-GPU training (1 epoch test)..."
echo "Log file: $LOG_FILE"
echo ""

# Run with nohup in background
nohup torchrun --nproc_per_node=8 \
    scripts/train_action_transformer_ddp_v2.py \
    --features-dir "$FEATURES_DIR" \
    --indices-dir "$INDICES_DIR" \
    --epochs 1 \
    --batch-size 32 \
    --output-dir models/test_ddp_v2_8gpus \
    > "$LOG_FILE" 2>&1 &

PID=$!

echo "🚀 Training launched in background!"
echo "   PID: $PID"
echo "   GPUs: 8 (CUDA 0-7)"
echo "   Epochs: 1"
echo "   Batch size: 32 per GPU (256 effective)"
echo ""
echo "📊 Monitor with:"
echo "   tail -f $LOG_FILE"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "⏱️  Expected duration: ~30-40 minutes for 1 epoch"
echo ""
echo "⏹️  To stop: kill $PID"
echo ""
echo "========================================================================"
