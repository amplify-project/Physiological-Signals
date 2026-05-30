#!/bin/bash
#==================================================================================
# 12-GPU DDP Training - FINE Classification (86 Action Classes)
#==================================================================================
# Purpose: Train on 86 fine-grained action classes to establish baseline
# Expected: ~17-20% validation accuracy (known to be challenging for pose features)
# Duration: ~3-4 hours for 100 epochs on 12 Tesla T4 GPUs
#==================================================================================

#----------------------------------------------------------------------------------
# Configuration
#----------------------------------------------------------------------------------
FEATURES_DIR="data/processed/features_kinetics700"
INDICES_DIR="data/processed"
OUTPUT_DIR="models/action_transformer_12gpus_fine"
LOG_FILE="logs/training_12gpus_fine.log"
NUM_GPUS=12

# Training parameters
BATCH_SIZE=32        # Per GPU
EPOCHS=100
LEARNING_RATE=1e-4
SEQUENCE_LENGTH=300

# Label level
LABEL_LEVEL="fine"   # 86 fine-grained action classes

#----------------------------------------------------------------------------------
# Process Info
#----------------------------------------------------------------------------------
# Task:               Action Recognition (86 fine-grained classes)
# Classes:            86 action classes (applauding, dancing, playing_instrument, etc.)
# Expected Accuracy:  17-20% (baseline from previous training)
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
#   Establish baseline performance for fine-grained classification.
#   This is the hardest level and known to have lower accuracy (~17%).
#   Will be compared against macro (13 classes) and binary (2 classes).
#
#----------------------------------------------------------------------------------

# Create output directories
mkdir -p models logs "$(dirname "$OUTPUT_DIR")"

# Show configuration
echo "================================================================================"
echo "12-GPU DDP TRAINING - FINE CLASSIFICATION (86 CLASSES)"
echo "================================================================================"
echo ""
echo "📊 Configuration:"
echo "   Label Level:       $LABEL_LEVEL (86 fine-grained classes)"
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
echo "🎯 Expected accuracy:   ~17-20% (baseline)"
echo ""
echo "================================================================================"
echo ""
echo "Starting in 3 seconds..."
sleep 3

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
