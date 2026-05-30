#!/bin/bash
# Launch 2 DDP training processes to utilize all 12 GPUs
# Process 1: GPUs 0-5 (6 GPUs)
# Process 2: GPUs 6-11 (6 GPUs)

echo "🚀 Starting 2 DDP training processes to use all 12 GPUs"
echo "================================================================"
echo "Process 1: GPUs 0-5 (port 29500)"
echo "Process 2: GPUs 6-11 (port 29501)"
echo "================================================================"

# Start first training process on GPUs 0-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
MASTER_PORT=29500 \
python scripts/train_action_transformer_ddp.py \
    --features-dir data/processed/features_kinetics700 \
    --output-dir models/action_transformer_kinetics700_group1 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0001 \
    2>&1 | tee logs/training_ddp_group1.log &

PID1=$!
echo "Started process 1 (PID: $PID1) on GPUs 0-5"

# Wait a bit to avoid race conditions
sleep 5

# Start second training process on GPUs 6-11
CUDA_VISIBLE_DEVICES=6,7,8,9,10,11 \
MASTER_PORT=29501 \
python scripts/train_action_transformer_ddp.py \
    --features-dir data/processed/features_kinetics700 \
    --output-dir models/action_transformer_kinetics700_group2 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0001 \
    2>&1 | tee logs/training_ddp_group2.log &

PID2=$!
echo "Started process 2 (PID: $PID2) on GPUs 6-11"

echo ""
echo "================================================================"
echo "Both training processes started!"
echo "Monitor logs:"
echo "  - Group 1 (GPUs 0-5): logs/training_ddp_group1.log"
echo "  - Group 2 (GPUs 6-11): logs/training_ddp_group2.log"
echo "================================================================"
echo ""
echo "Press Ctrl+C to stop both processes"

# Wait for both processes
wait $PID1 $PID2

echo ""
echo "🎉 Both training processes completed!"
