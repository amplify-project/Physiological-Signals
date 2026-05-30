#!/bin/bash
# Start 100-epoch single-GPU training with proper environment

cd ~/concert_engagement

# Kill any existing training
pkill -f train_action_transformer

# Activate venv
source venv/bin/activate

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Start training
nohup python scripts/train_action_transformer_dataparallel.py \
    --features-dir data/processed/features_kinetics700 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0001 \
    > logs/training_100epochs.log 2>&1 &

sleep 2
echo "Training started with PID: $!"
echo "Log file: logs/training_100epochs.log"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -1
