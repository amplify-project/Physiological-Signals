#!/usr/bin/env python3
"""Test loading feature files to diagnose the hang issue."""
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import sys

class SimpleDataset(Dataset):
    def __init__(self, features_dir):
        self.samples = list(Path(features_dir).rglob('*.npz'))  # ALL files
        print(f"Found {len(self.samples)} files to test")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npz_path = self.samples[idx]
        print(f"Loading {idx}: {npz_path.name}", flush=True)
        
        # Try loading exactly as the training script does
        with np.load(npz_path, allow_pickle=False) as data:
            features = data['features'].copy()
        
        # Flatten and process
        features = features.reshape(features.shape[0], -1)
        
        # Pad/truncate to 300
        if features.shape[0] < 300:
            padding = np.zeros((300 - features.shape[0], features.shape[1]))
            features = np.vstack([features, padding])
        elif features.shape[0] > 300:
            indices = np.linspace(0, features.shape[0] - 1, 300, dtype=int)
            features = features[indices]
        
        print(f"  ✓ Loaded {npz_path.name}: {features.shape}", flush=True)
        return torch.FloatTensor(features), 0

# Test 1: Load files directly
print("\n=== TEST 1: Direct file loading ===")
features_dir = Path('data/processed/features_kinetics700/train')
files = list(features_dir.rglob('*.npz'))[:10]
print(f"Testing first 10 files from {len(files)} total...")

for i, f in enumerate(files):
    try:
        print(f"{i+1}. Loading {f.name}...", flush=True)
        with np.load(f, allow_pickle=False) as data:
            shape = data['features'].shape
        print(f"   ✓ {shape}", flush=True)
    except Exception as e:
        print(f"   ✗ ERROR: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)

print("\n✅ All direct loads successful!\n")

# Test 2: DataLoader with batch_size=1
print("\n=== TEST 2: DataLoader batch_size=1 ===")
dataset = SimpleDataset('data/processed/features_kinetics700/train')
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

print("Attempting to load first batch...")
for i, (features, label) in enumerate(loader):
    print(f"Batch {i}: {features.shape}")
    if i >= 4:  # Test first 5 batches
        break

print("\n✅ DataLoader test successful!")

# Test 3: DataLoader with batch_size=32
print("\n=== TEST 3: DataLoader batch_size=32 ===")
loader32 = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

print("Attempting to load first batch of 32...")
for i, (features, label) in enumerate(loader32):
    print(f"Batch {i}: {features.shape}")
    if i >= 0:  # Just test first batch
        break

print("\n✅ Batch 32 test passed!")

# Test 4: DataLoader with SHUFFLE (like training) - FULL DATASET
print("\n=== TEST 4: DataLoader with SHUFFLE=True (FULL DATASET) ===")
dataset_full = SimpleDataset('data/processed/features_kinetics700/train')
loader_shuffle = DataLoader(dataset_full, batch_size=32, shuffle=True, num_workers=0)

print("Attempting to load 10 shuffled batches from full dataset...")
for i, (features, label) in enumerate(loader_shuffle):
    print(f"Shuffled Batch {i}: {features.shape}", flush=True)
    if i >= 9:  # Test 10 shuffled batches
        break

print("\n✅ All tests passed! File loading works correctly.")
