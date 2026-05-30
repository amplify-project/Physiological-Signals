#!/bin/bash
# Test DDP V2 with pre-computed indices
# Run this to test the fix for large dataset hang

echo "========================================="
echo "DDP V2 TEST - Pre-computed Indices Fix"
echo "========================================="
echo ""

# Step 1: Generate indices (run once)
echo "[1/2] Generating pre-computed indices..."
echo "This takes ~2-5 minutes but only needs to run ONCE"
python scripts/generate_dataset_indices.py \
    --features-dir data/processed/features_kinetics700 \
    --output-dir data/processed

if [ $? -ne 0 ]; then
    echo "❌ Index generation failed!"
    exit 1
fi

echo ""
echo "[2/2] Testing DDP training with 2 GPUs..."
echo ""

# Step 2: Test with 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/train_action_transformer_ddp_v2.py \
    --features-dir data/processed/features_kinetics700 \
    --indices-dir data/processed \
    --output-dir models/action_transformer_test_v2 \
    --batch-size 32 \
    --epochs 2 \
    --lr 0.0001

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ SUCCESS! DDP V2 works!"
    echo "========================================="
    echo ""
    echo "Key changes:"
    echo "  ✓ Dataset loads instantly (no 2-5 min scan)"
    echo "  ✓ No hang at first batch"
    echo "  ✓ Training proceeds normally"
    echo ""
    echo "To run full training:"
    echo "  CUDA_VISIBLE_DEVICES=0-7 torchrun --nproc_per_node=8 \\"
    echo "    scripts/train_action_transformer_ddp_v2.py \\"
    echo "    --features-dir data/processed/features_kinetics700 \\"
    echo "    --epochs 100 --batch-size 32"
else
    echo ""
    echo "========================================="
    echo "❌ Test failed"
    echo "========================================="
    exit 1
fi
