#!/bin/bash
#==================================================================================
# 12-GPU DDP Training - MACRO Classification (13 Classes)
#==================================================================================
# Purpose: Train on 13 macro classes (6 engagement + 7 disengagement)
# Expected: ~50-60% validation accuracy (medium difficulty)
# Duration: ~4-5 hours for 100 epochs on 12 Tesla T4 GPUs
#==================================================================================

# Enable verbose debugging
set -x
set -e

echo "=========================================="
echo "DEBUG: Script started at $(date)"
echo "DEBUG: Current directory: $(pwd)"
echo "DEBUG: User: $(whoami)"
echo "=========================================="

#----------------------------------------------------------------------------------
# Configuration
#----------------------------------------------------------------------------------
FEATURES_DIR="data/processed/features_kinetics700"
INDICES_DIR="data/processed"
OUTPUT_DIR="models/action_transformer_12gpus_macro"
LOG_FILE="logs/training_12gpus_macro.log"
NUM_GPUS=12

# Training parameters
BATCH_SIZE=32        # Per GPU
EPOCHS=100
LEARNING_RATE=1e-4
SEQUENCE_LENGTH=300

# Label level
LABEL_LEVEL="macro"  # 13 macro category classes

#----------------------------------------------------------------------------------
# Process Info
#----------------------------------------------------------------------------------
# Task:               Action Recognition (13 macro categories)
# Classes:            13 categories:
#                     - ENGAGEMENT: applause, dancing, cheering, singing, 
#                                   playing_instrument, recording
#                     - DISENGAGEMENT: phone_distraction, passive, fidgeting,
#                                      negative_body_language, smoking, checking_time,
#                                      eating_drinking, tired_uncomfortable, negative_reactions
# Expected Accuracy:  50-60% (estimated)
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
#   Test if grouping fine classes into semantic categories improves accuracy.
#   This is the middle level between fine (86) and binary (2).
#   Should show better performance than fine-grained classification.
#
#----------------------------------------------------------------------------------

# Create output directories
mkdir -p models logs "$(dirname "$OUTPUT_DIR")"

echo "DEBUG: Created directories"
echo "DEBUG: Features dir exists: $(test -d "$FEATURES_DIR" && echo YES || echo NO)"
echo "DEBUG: Indices dir exists: $(test -d "$INDICES_DIR" && echo YES || echo NO)"
echo ""

# Show configuration
echo "================================================================================"
echo "12-GPU DDP TRAINING - MACRO CLASSIFICATION (13 CATEGORIES)"
echo "================================================================================"
echo ""
echo "📊 Configuration:"
echo "   Label Level:       $LABEL_LEVEL (13 macro categories)"
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
echo "🎯 Expected accuracy:   ~50-60% (estimated)"
echo ""
echo "================================================================================"
echo ""
echo "Starting in 3 seconds..."
sleep 3

echo "DEBUG: Activating venv..."
echo "DEBUG: venv path exists: $(test -d venv/bin && echo YES || echo NO)"

# Activate virtual environment
source venv/bin/activate

echo "DEBUG: venv activated"
echo "DEBUG: Python path: $(which python)"
echo "DEBUG: Torchrun path: $(which torchrun)"
echo ""

# Launch with torchrun
echo "DEBUG: Launching torchrun..."
echo "DEBUG: Command will be executed with nohup"
echo ""

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
echo "DEBUG: Torchrun launched with PID: $PID"
echo "DEBUG: Log file: $LOG_FILE"
echo "DEBUG: Sleeping 2 seconds to let process initialize..."
sleep 2
echo "DEBUG: Checking if process is still running..."
ps -p $PID > /dev/null 2>&1 && echo "DEBUG: Process $PID is RUNNING" || echo "DEBUG: Process $PID is NOT RUNNING"
echo ""
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
