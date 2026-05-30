#!/bin/bash
# Test cleaned dataset loader on server

cd ~/concert_engagement

# Activate venv
source venv/bin/activate

echo "================================"
echo "TESTING DATASET WITH CLEANED LABELS"
echo "================================"

python3 << 'PYTHON_EOF'
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from scripts.ddp_precomputed_dataset import PrecomputedActionDataset

# Test original labels
print("\n" + "="*80)
print("ORIGINAL LABELS (use_cleaned_labels=False)")
print("="*80)

dataset_original = PrecomputedActionDataset(
    features_dir='data/processed/mediapipe_features',
    indices_dir='data/processed',
    split='train',
    label_level='binary',
    use_cleaned_labels=False,
    sequence_length=64,
    verbose=True,
    rank=0
)

print(f"\n✅ Original: {len(dataset_original):,} samples, {dataset_original.num_classes} classes")

# Test cleaned labels
print("\n" + "="*80)
print("CLEANED LABELS (use_cleaned_labels=True)")
print("="*80)

dataset_cleaned = PrecomputedActionDataset(
    features_dir='data/processed/mediapipe_features',
    indices_dir='data/processed',
    split='train',
    label_level='binary',
    use_cleaned_labels=True,
    sequence_length=64,
    verbose=True,
    rank=0
)

print(f"\n✅ Cleaned: {len(dataset_cleaned):,} samples, {dataset_cleaned.num_classes} classes")
print(f"   Quarantined: {len(dataset_cleaned.quarantined_classes)} classes")

# Summary
reduction = len(dataset_original) - len(dataset_cleaned)
reduction_pct = (reduction / len(dataset_original)) * 100

print(f"\n📊 SUMMARY:")
print(f"   Original: {len(dataset_original):,} samples")
print(f"   Cleaned:  {len(dataset_cleaned):,} samples")
print(f"   Removed:  {reduction:,} samples ({reduction_pct:.1f}%)")

# Test sample loading
print(f"\n🔷 Loading test sample...")
features, label = dataset_cleaned[0]
print(f"✅ Sample loaded: features.shape={features.shape}, label={label}")

print("\n✅ Dataset test complete!")
PYTHON_EOF
