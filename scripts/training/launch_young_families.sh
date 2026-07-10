#!/bin/bash
# Launch Young Families v0 engagement training DETACHED (non-blocking).
# Uses only the two genuinely-free T4s (physical GPUs 0 and 8) via
# CUDA_VISIBLE_DEVICES so we never disturb other users' running jobs.
#
# Run from ~/concert_engagement on the server:
#   bash scripts/launch_young_families.sh
#
# It prints the PID + log path and returns immediately. Poll with:
#   bash scripts/poll_gpus.sh

set -e
cd ~/concert_engagement
source venv/bin/activate

# Regenerate the trainer from the patch (idempotent)
python3 scripts/patch_young_families_trainer.py

FEATURES_DIR="data/processed/features_kinetics700"
INDICES_DIR="data/processed"
OUTPUT_DIR="models/action_transformer_young_families_v0"
HIER_FILE="models/action_transformer_kinetics700/hierarchical_labels_young_families.json"
BATCH_SIZE=32          # per GPU; 32 * 2 = 64 effective
EPOCHS=50
LR=0.0001
SEQUENCE_LENGTH=64
ENGAGEMENT_WEIGHT=1.5  # presume-engaged tilt

mkdir -p "$OUTPUT_DIR" logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/train_young_families_${TIMESTAMP}.log"

# Only expose the two free GPUs; DDP sees them as cuda:0 and cuda:1
export CUDA_VISIBLE_DEVICES=0,8

nohup python3 scripts/train_young_families.py \
    --features-dir "$FEATURES_DIR" \
    --indices-dir "$INDICES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --label-level binary \
    --hierarchical-file "$HIER_FILE" \
    --engagement-weight "$ENGAGEMENT_WEIGHT" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --sequence-length "$SEQUENCE_LENGTH" \
    > "$LOGFILE" 2>&1 &

PID=$!
echo "$PID" > logs/young_families.pid
echo "================================================================"
echo "Young Families v0 training launched (DETACHED)"
echo "  PID:     $PID"
echo "  GPUs:    physical 0 and 8 (CUDA_VISIBLE_DEVICES=0,8)"
echo "  Log:     $LOGFILE"
echo "  Output:  $OUTPUT_DIR"
echo "================================================================"
echo "Tail log:  tail -f $LOGFILE"
echo "Poll GPUs: bash scripts/poll_gpus.sh"
