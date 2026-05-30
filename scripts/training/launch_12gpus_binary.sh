#!/bin/bash
#==================================================================================
# 12-GPU DDP Training - BINARY Classification (Engagement vs Disengagement)
#==================================================================================
# Purpose: Train on 2 binary classes (simplest classification)
# Expected: >75% validation accuracy (easiest level, target performance)
# Duration: ~3-4 hours for 100 epochs on 12 Tesla T4 GPUs
#==================================================================================

#----------------------------------------------------------------------------------
# Configuration
#----------------------------------------------------------------------------------
FEATURES_DIR="data/processed/features_kinetics700"
INDICES_DIR="data/processed"
OUTPUT_DIR="models/action_transformer_12gpus_binary"
LOG_FILE="logs/training_12gpus_binary.log"
NUM_GPUS=12

# Training parameters
BATCH_SIZE=32        # Per GPU
EPOCHS=100
LEARNING_RATE=1e-4
SEQUENCE_LENGTH=300

# Label level
LABEL_LEVEL="binary"  # 2 classes: engagement / disengagement

#----------------------------------------------------------------------------------
# Process Info
#----------------------------------------------------------------------------------
# Task:               Binary Classification (Engagement Detection)
# Classes:            2 classes:
#                     - engagement (1): applause, dancing, cheering, singing,
#                                       playing_instrument, recording
#                     - disengagement (0): phone_distraction, passive, fidgeting,
#                                          negative_body_language, smoking, eating, etc.
# Expected Accuracy:  >75% (target)
# 
# Training:
#   - Batch size:     32 × 12 = 384 (effective)
#   - Epochs:         100
#   - Learning rate:  1e-4
#   - Sequence:       300 frames
#   - Backend:        Gloo (CPU-based, works on PCIe GPUs)
#
# Hardware:
#   - GPUs:           12× Tesla T4
#   - Memory:         ~15 GB VRAM per GPU
#   - Duration:       ~3-4 hours for 100 epochs
#
# Purpose:
#   Test if pose features can reliably detect audience engagement.
#   This is the simplest and most important classification level.
#   Target: >75% accuracy to validate pose-based engagement detection.
#
# Use Case:
#   Real-time engagement monitoring at concerts and live events.
#   Binary classification is sufficient for practical applications.
#
#----------------------------------------------------------------------------------

# Create output directories
mkdir -p models logs "$(dirname "$OUTPUT_DIR")"

# Show configuration
echo "================================================================================"
echo "12-GPU DDP TRAINING - BINARY CLASSIFICATION (ENGAGEMENT DETECTION)"
echo "================================================================================"
echo ""
echo "📊 Configuration:"
echo "   Label Level:       $LABEL_LEVEL (2 classes: engagement/disengagement)"
echo "   Features:          $FEATURES_DIR"
echo "   Output:            $OUTPUT_DIR"
echo "   Log:               $LOG_FILE"
echo ""
echo "🎯 Training:"
echo "   GPUs:              $NUM_GPUS"
echo "   Batch size:        $BATCH_SIZE × $NUM_GPUS = $(($BATCH_SIZE * $NUM_GPUS))"
echo "   Epochs:            $EPOCHS"
echo "   Learning rate:     $LEARNING_RATE"
echo "   Sequence length:   $SEQUENCE_LENGTH"
echo ""
echo "⏱️  Expected duration:  ~3-4 hours for 100 epochs"
echo "🎯 Target accuracy:     >75%"
echo ""
echo "================================================================================"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Activate virtual environment
source venv/bin/activate

# Launch with torchrun
nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    scripts/train_action_transformer_ddp_v2.py \
    --features-dir "$FEATURES_DIR" \
    --indices-dir "$INDICES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --label-level "$LABEL_LEVEL" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --sequence-length $SEQUENCE_LENGTH \
    > "$LOG_FILE" 2>&1 &

# Get process ID
PID=$!
echo "✅ Training launched!"
echo ""
echo "📋 Process ID:     $PID"
echo "📂 Log file:       $LOG_FILE"
echo "📁 Output dir:     $OUTPUT_DIR"
echo ""
echo "💡 Monitor training:"
echo "   tail -f $LOG_FILE"
echo ""
echo "💡 Check GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "💡 Stop training:"
echo "   kill $PID"
echo ""
echo "================================================================================"
