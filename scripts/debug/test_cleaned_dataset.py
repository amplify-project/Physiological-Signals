"""
Test the cleaned dataset loader with quarantine filtering.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.ddp_precomputed_dataset import PrecomputedActionDataset

# Test both versions
print("\n" + "="*80)
print("TESTING ORIGINAL LABELS (use_cleaned_labels=False)")
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

print(f"\n✅ Original dataset: {len(dataset_original):,} samples")
print(f"   Num classes: {dataset_original.num_classes}")

print("\n" + "="*80)
print("TESTING CLEANED LABELS (use_cleaned_labels=True)")
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

print(f"\n✅ Cleaned dataset: {len(dataset_cleaned):,} samples")
print(f"   Num classes: {dataset_cleaned.num_classes}")
print(f"   Quarantined classes: {len(dataset_cleaned.quarantined_classes)}")

# Calculate reduction
reduction = len(dataset_original) - len(dataset_cleaned)
reduction_pct = (reduction / len(dataset_original)) * 100

print(f"\n📊 COMPARISON:")
print(f"   Original: {len(dataset_original):,} samples")
print(f"   Cleaned:  {len(dataset_cleaned):,} samples")
print(f"   Removed:  {reduction:,} samples ({reduction_pct:.1f}%)")

# Test loading a sample
print("\n🔷 Testing sample loading from cleaned dataset...")
features, label = dataset_cleaned[0]
print(f"✅ Sample loaded: features.shape={features.shape}, label={label}")

print("\n✅ Dataset test complete!")
