#!/bin/bash
# Launch DDP training with optimal GPU grouping for T4s
# T4 GPUs work best in groups of 8 or fewer due to peer mapping limits

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # First 8 GPUs

echo "🚀 Starting DDP training on 8 GPUs: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

torchrun --nproc_per_node=8 --nnodes=1 \
    scripts/train_action_transformer_ddp.py \
    --features-dir data/processed/features_kinetics700 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0001 \
    2>&1 | tee logs/training_ddp_8gpus.log
