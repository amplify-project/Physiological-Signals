#!/usr/bin/env python3
"""
Debug script to test DataLoader step by step to find where it hangs.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time

print("="*60)
print("STEP 1: Import successful")
print("="*60)

class SimpleDataset(Dataset):
    """Minimal dataset that mimics ActionDataset."""
    
    def __init__(self, features_dir, split='train'):
        self.features_dir = Path(features_dir) / split
        self.samples = []
        
        print(f"\nSTEP 2: Scanning {self.features_dir}")
        action_dirs = sorted([d for d in self.features_dir.iterdir() if d.is_dir()])
        print(f"Found {len(action_dirs)} action classes")
        
        for idx, action_dir in enumerate(action_dirs[:3]):  # Only first 3 classes for speed
            npz_files = list(action_dir.glob('*.npz'))[:5]  # Only first 5 files per class
            for npz_file in npz_files:
                self.samples.append((npz_file, idx))
            print(f"  Class {idx}: {action_dir.name} - {len(npz_files)} files")
        
        print(f"Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npz_path, label = self.samples[idx]
        
        print(f"\n__getitem__ called for idx={idx}: {npz_path.name}", flush=True)
        
        try:
            print(f"  Opening file with np.load()...", flush=True)
            start = time.time()
            with np.load(npz_path, allow_pickle=False) as data:
                print(f"  File opened in {time.time()-start:.3f}s", flush=True)
                features = data['features'].copy()
                print(f"  Loaded features: {features.shape}", flush=True)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            features = np.zeros((300, 543, 3))
        
        # Flatten
        features = features.reshape(features.shape[0], -1)
        
        # Pad/truncate to 300 frames
        num_frames = features.shape[0]
        if num_frames < 300:
            padding = np.zeros((300 - num_frames, features.shape[1]))
            features = np.vstack([features, padding])
        elif num_frames > 300:
            indices = np.linspace(0, num_frames - 1, 300, dtype=int)
            features = features[indices]
        
        print(f"  Returning tensor: {features.shape}", flush=True)
        return torch.FloatTensor(features), label


print("\n" + "="*60)
print("STEP 3: Creating dataset")
print("="*60)
dataset = SimpleDataset('data/processed/features_kinetics700', split='train')

print("\n" + "="*60)
print("STEP 4: Testing direct dataset access (no DataLoader)")
print("="*60)
print("Calling dataset[0]...")
features, label = dataset[0]
print(f"✓ Success! Got tensor {features.shape}, label={label}")

print("\n" + "="*60)
print("STEP 5: Creating DataLoader with num_workers=0")
print("="*60)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
print(f"DataLoader created: {len(dataloader)} batches")

print("\n" + "="*60)
print("STEP 6: Iterating DataLoader (num_workers=0)")
print("="*60)
print("Starting iteration...")
for batch_idx, (features, labels) in enumerate(dataloader):
    print(f"\nBatch {batch_idx}: features={features.shape}, labels={labels.shape}")
    if batch_idx >= 2:  # Only test first 3 batches
        break

print("\n" + "="*60)
print("STEP 7: Creating DataLoader with num_workers=1")
print("="*60)
dataloader_mp = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, timeout=10)
print(f"DataLoader created with 1 worker")

print("\n" + "="*60)
print("STEP 8: Iterating DataLoader (num_workers=1)")
print("="*60)
print("Starting iteration with multiprocessing...")
try:
    for batch_idx, (features, labels) in enumerate(dataloader_mp):
        print(f"\nBatch {batch_idx}: features={features.shape}, labels={labels.shape}")
        if batch_idx >= 2:
            break
    print("\n✓ SUCCESS: DataLoader with workers completed!")
except Exception as e:
    print(f"\n✗ FAILED: {type(e).__name__}: {e}")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)
