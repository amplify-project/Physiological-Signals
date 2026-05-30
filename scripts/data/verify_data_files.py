"""
FINAL VERIFICATION: Check actual data files to confirm disengagement data is OK.
Sample random files from both categories and verify they load correctly.
"""

import numpy as np
from pathlib import Path
import random

features_dir = Path("data/processed/features_kinetics700")

# Sample engagement actions
engagement_samples = {
    'playing cello': [],
    'dancing ballet': [],
    'breakdancing': []
}

# Sample disengagement actions
disengagement_samples = {
    'stretching leg': [],
    'sleeping': [],
    'reading book': [],
    'winking': []  # This one has F1=0, let's check it
}

print("=" * 80)
print("DATA FILE VERIFICATION")
print("=" * 80)

# Collect sample files
for split in ['train', 'val']:
    split_dir = features_dir / split
    if not split_dir.exists():
        print(f"⚠️ Directory not found: {split_dir}")
        continue
    
    for action in engagement_samples:
        files = list(split_dir.glob(f"{action.replace(' ', '_')}*.npz"))
        engagement_samples[action].extend(files[:2])
    
    for action in disengagement_samples:
        files = list(split_dir.glob(f"{action.replace(' ', '_')}*.npz"))
        disengagement_samples[action].extend(files[:2])

# Check engagement files
print("\n🎉 ENGAGEMENT FILES:")
print("-" * 80)
for action, files in engagement_samples.items():
    print(f"\n{action.upper()}:")
    for f in files[:3]:
        if f.exists():
            try:
                data = np.load(f)
                kp = data['keypoints']
                print(f"  ✅ {f.name}")
                print(f"     Shape: {kp.shape}, Min: {kp.min():.3f}, Max: {kp.max():.3f}, Mean: {kp.mean():.3f}")
                print(f"     NaN: {np.isnan(kp).any()}, Inf: {np.isinf(kp).any()}")
            except Exception as e:
                print(f"  ❌ {f.name}: ERROR - {e}")

# Check disengagement files
print("\n😴 DISENGAGEMENT FILES:")
print("-" * 80)
for action, files in disengagement_samples.items():
    print(f"\n{action.upper()}:")
    for f in files[:3]:
        if f.exists():
            try:
                data = np.load(f)
                kp = data['keypoints']
                print(f"  ✅ {f.name}")
                print(f"     Shape: {kp.shape}, Min: {kp.min():.3f}, Max: {kp.max():.3f}, Mean: {kp.mean():.3f}")
                print(f"     NaN: {np.isnan(kp).any()}, Inf: {np.isinf(kp).any()}")
            except Exception as e:
                print(f"  ❌ {f.name}: ERROR - {e}")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
print("""
If ALL files load successfully with proper shapes and values:
  ✅ Data is FINE - The poor performance is due to:
     1. Subtle movements that pose keypoints can't capture
     2. High similarity between disengagement actions
     3. Model overfitting

If disengagement files show errors:
  ❌ Data PROBLEM - Need to investigate file corruption or format issues
""")
