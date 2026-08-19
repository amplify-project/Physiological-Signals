#!/bin/bash
# Launch Family-Concert v1 engagement training DETACHED (non-blocking).
#
# This is the attention-first model for the seated young-families concert context.
# It reads the SAME feature files + indices as the online-learning (original)
# models -- READ-ONLY -- so the original data is never modified. The trainer
# filters those samples down to the 14-class family-concert allow-list at load
# time (see scripts/patch_family_concert_trainer.py) and writes a brand-new
# checkpoint under its own output dir. Nothing about the original model is touched.
#
# Uses only the two genuinely-free T4s (physical GPUs 0 and 8) via
# CUDA_VISIBLE_DEVICES so we never disturb other users' running jobs.
#
# Run from ~/concert_engagement on the server:
#   bash scripts/launch_family_concert_v1.sh
#
# It prints the PID + log path and returns immediately. Poll with:
#   bash scripts/poll_gpus.sh

set -e
cd ~/concert_engagement
source venv/bin/activate

# Regenerate the trainer from the patch (idempotent)
python3 scripts/patch_family_concert_trainer.py

FEATURES_DIR="data/processed/features_kinetics700"   # read-only; shared with the original models
INDICES_DIR="data/processed"
OUTPUT_DIR="${OUTPUT_DIR:-models/action_transformer_family_concert_v1}"
HIER_FILE="models/action_transformer_kinetics700/hierarchical_labels_family_concert.json"
BATCH_SIZE=32          # per GPU; 32 * 2 = 64 effective
EPOCHS=50
LR=0.0001
SEQUENCE_LENGTH=64
ENGAGEMENT_WEIGHT="${ENGAGEMENT_WEIGHT:-1.5}"  # presume-engaged tilt; override for class imbalance
NUM_WORKERS=8          # background .npz prefetchers per rank

mkdir -p "$OUTPUT_DIR" logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/train_family_concert_${TIMESTAMP}.log"

# Only expose the two free GPUs; DDP sees them as cuda:0 and cuda:1.
# Pick genuinely-free GPUs at launch time (nvidia-smi). As of 2026-08-19, free = 3,4,5,6.
# Override from the CLI: `GPUS=3,4 bash scripts/launch_family_concert_v1.sh`
export CUDA_VISIBLE_DEVICES="${GPUS:-3,4}"

# NOTE: do NOT pass --use-cleaned-labels. The datasets must load the full
# 86-class standard mapping first; the trainer then filters to the allow-list
# and internally flips to the resize-safe sampler.
nohup python3 scripts/train_family_concert.py \
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
    --num-workers "$NUM_WORKERS" \
    > "$LOGFILE" 2>&1 &

PID=$!
echo "$PID" > logs/family_concert.pid
echo "================================================================"
echo "Family-Concert v1 training launched (DETACHED)"
echo "  PID:     $PID"
echo "  GPUs:    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Log:     $LOGFILE"
echo "  Output:  $OUTPUT_DIR"
echo "  Labels:  $HIER_FILE"
echo "================================================================"
echo "Tail log:  tail -f $LOGFILE"
echo "Poll GPUs: bash scripts/poll_gpus.sh"
