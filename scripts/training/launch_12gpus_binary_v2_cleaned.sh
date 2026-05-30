#!/bin/bash
# Launch Binary Engagement Classification (V2 CLEANED)
# Uses cleaned labels with 32 quarantined ambiguous classes
# Increased dropout 0.1→0.3 for better regularization
# Training on 12x Tesla T4 GPUs with DDP

set -e

cd ~/concert_engagement

# Activate venv
source venv/bin/activate

# Configuration
FEATURES_DIR="data/processed/features_kinetics700"
INDICES_DIR="data/processed"
OUTPUT_DIR="models/action_transformer_12gpus_binary_v2_cleaned"
LABEL_LEVEL="binary"
BATCH_SIZE=32   # Per GPU: 32 * 12 = 384 total
EPOCHS=50       # Reduced from 100 - binary peaked at epoch 14, macro at 22
LR=0.0001
SEQUENCE_LENGTH=64

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "================================"
echo "🚀 BINARY V2 CLEANED TRAINING"
echo "================================"
echo "Output: $OUTPUT_DIR"
echo "Labels: Cleaned v2 (32 classes quarantined)"
echo "Samples: ~44k train (vs 65k original)"
echo "Dropout: 0.3 (increased for regularization)"
echo "GPUs: 12x Tesla T4"
echo "Batch size: $BATCH_SIZE/gpu = 384 total"
echo "Epochs: $EPOCHS"
echo "================================"
echo ""

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/train_binary_v2_cleaned_${TIMESTAMP}.log"

echo "📝 Logging to: $LOGFILE"
echo ""

# Run with torchrun (12 GPUs)
torchrun \
    --nproc_per_node=12 \
    --master_port=29500 \
    scripts/train_action_transformer_ddp_v2.py \
    --features-dir "$FEATURES_DIR" \
    --indices-dir "$INDICES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --label-level "$LABEL_LEVEL" \
    --use-cleaned-labels \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --sequence-length "$SEQUENCE_LENGTH" \
    2>&1 | tee "$LOGFILE"

echo ""
echo "✅ Training complete! Results in $OUTPUT_DIR"
echo "📊 Best model: $OUTPUT_DIR/best_model.pth"
echo "📈 Metrics: $OUTPUT_DIR/metrics.json"
