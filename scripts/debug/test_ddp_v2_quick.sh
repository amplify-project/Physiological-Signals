#!/bin/bash
# Quick DDP V2 test with full 65K dataset
# Tests the pre-computed indices fix for DDP hang issue

set -e

echo "=================================="
echo "🧪 DDP V2 QUICK TEST (65K samples)"
echo "=================================="
echo ""

# Activate virtual environment
source venv/bin/activate

FEATURES_DIR="data/processed/features_kinetics700"
INDICES_DIR="data/processed"
LOG_FILE="logs/test_ddp_v2_quick.log"

# Create logs directory
mkdir -p logs

# Check if features exist
if [ ! -d "$FEATURES_DIR" ]; then
    echo "❌ ERROR: Features directory not found: $FEATURES_DIR"
    echo "Please run feature extraction first"
    exit 1
fi

echo "Step 1: Generate pre-computed indices (this will take 2-5 minutes)..."
echo "---------------------------------------------------------------------"

python3 scripts/generate_dataset_indices.py \
    --features-dir "$FEATURES_DIR" \
    --output-dir "$INDICES_DIR"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Index generation failed"
    exit 1
fi

echo ""
echo "✅ Indices generated successfully!"
echo ""

echo "Step 2: Test DDP training with 2 GPUs, 2 epochs..."
echo "---------------------------------------------------"
echo "Log file: $LOG_FILE"
echo ""

# Run with nohup in background
nohup torchrun --nproc_per_node=2 \
    scripts/train_action_transformer_ddp_v2.py \
    --features-dir "$FEATURES_DIR" \
    --indices-dir "$INDICES_DIR" \
    --epochs 2 \
    --batch-size 32 \
    --output-dir models/test_ddp_v2 \
    > "$LOG_FILE" 2>&1 &

PID=$!

echo "🚀 Training launched in background!"
echo "   PID: $PID"
echo "   GPUs: 2 (CUDA_VISIBLE_DEVICES will use first 2 available)"
echo "   Epochs: 2"
echo "   Batch size: 32 per GPU (64 effective)"
echo ""
echo "📊 Monitor with:"
echo "   tail -f $LOG_FILE"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "🔍 Check if it hangs at first batch (the bug we're fixing)"
echo "   Success = Training proceeds past first batch"
echo "   Failure = Hangs with 100% GPU compute, 0% memory"
echo ""
echo "⏹️  To stop: kill $PID"
