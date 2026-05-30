#!/bin/bash
# Launch binary classification training on ALL 12 GPUs with Gloo backend
# Binary: engagement (1) vs disengagement (0)
# Target: >75% validation accuracy

set -e

echo "========================================================================"
echo "🚀 12-GPU TRAINING: Action Recognition (86 classes)"
echo "========================================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

FEATURES_DIR="data/processed/features_kinetics700"
INDICES_DIR="data/processed"
OUTPUT_DIR="models/action_transformer_12gpus"
LOG_FILE="logs/training_12gpus.log"
EPOCHS=100
BATCH_SIZE=32
LR=0.0001

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Task: Action Recognition (86 fine-grained classes)"
echo "  Features: $FEATURES_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Effective batch size: $((BATCH_SIZE * 12)) (32 × 12 = 384)"
echo "  Learning rate: $LR"
echo "  GPUs: 12 (ALL Tesla T4s)"
echo "  Backend: Gloo"
echo ""

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

echo "Starting binary training on 12 GPUs..."
echo "Log file: $LOG_FILE"
echo ""

# Launch training with nohup - using ALL 12 GPUs
# Note: Using train_action_transformer_ddp_v2.py which loads hierarchical labels
# The hierarchical_labels.json includes binary, macro, and fine mappings
nohup torchrun --nproc_per_node=12 \
    scripts/train_action_transformer_ddp_v2.py \
    --features-dir "$FEATURES_DIR" \
    --indices-dir "$INDICES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    > "$LOG_FILE" 2>&1 &

PID=$!

echo "🚀 Training launched in background!"
echo ""
echo "Process ID: $PID"
echo "GPUs: 12 (ALL available)"
echo "Classes: 86 action classes"
echo "Dataset: ~65K training samples"
echo "Log file: $LOG_FILE"
echo ""
echo "📊 Monitor with:"
echo "   tail -f $LOG_FILE"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "⏱️  Expected duration:"
echo "   - With 12 GPUs: ~3-4 hours for 100 epochs"
echo "   - Batch size 384 = 12× faster than single GPU"
echo ""
echo "⏹️  To stop: kill $PID"
echo ""
echo "========================================================================"
