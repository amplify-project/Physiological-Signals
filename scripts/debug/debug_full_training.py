#!/usr/bin/env python3
"""
Complete debug test of training loop with CUDA fix.
Tests: DataLoader + Model + GPU + DataParallel + Training iteration
"""

import os
import sys

# CRITICAL FIX: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
print("="*60)
print("STEP 1: Setting CUDA_VISIBLE_DEVICES before torch import")
print("="*60)
max_gpus = 8
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(max_gpus))
print(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

print("\n" + "="*60)
print("STEP 2: Check CUDA devices after import")
print("="*60)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

class SimpleDataset(Dataset):
    """Minimal dataset."""
    def __init__(self, features_dir, split='train'):
        self.features_dir = Path(features_dir) / split
        self.samples = []
        
        # Only load first 3 classes, 10 files each for speed
        action_dirs = sorted([d for d in self.features_dir.iterdir() if d.is_dir()])[:3]
        for idx, action_dir in enumerate(action_dirs):
            npz_files = list(action_dir.glob('*.npz'))[:10]
            for npz_file in npz_files:
                self.samples.append((npz_file, idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npz_path, label = self.samples[idx]
        
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                features = data['features'].copy()
        except Exception as e:
            print(f"Error loading {npz_path.name}: {e}")
            features = np.zeros((300, 543, 3))
        
        features = features.reshape(features.shape[0], -1)
        num_frames = features.shape[0]
        
        if num_frames < 300:
            padding = np.zeros((300 - num_frames, features.shape[1]))
            features = np.vstack([features, padding])
        elif num_frames > 300:
            indices = np.linspace(0, num_frames - 1, 300, dtype=int)
            features = features[indices]
        
        return torch.FloatTensor(features), label


class TemporalTransformer(nn.Module):
    """Minimal transformer model."""
    def __init__(self, input_dim=543*3, num_classes=3, d_model=256, nhead=8, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.fc(x)
        return logits


print("\n" + "="*60)
print("STEP 3: Create dataset")
print("="*60)
dataset = SimpleDataset('data/processed/features_kinetics700', split='train')
print(f"Dataset size: {len(dataset)} samples")

print("\n" + "="*60)
print("STEP 4: Create DataLoader with workers")
print("="*60)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, timeout=10)
print(f"DataLoader created: {len(dataloader)} batches")

print("\n" + "="*60)
print("STEP 5: Initialize model")
print("="*60)
model = TemporalTransformer(input_dim=543*3, num_classes=3, d_model=128, nhead=4, num_layers=2)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

print("\n" + "="*60)
print("STEP 6: Move model to GPU and wrap with DataParallel")
print("="*60)
device = torch.device('cuda:0')
model = model.to(device)
print(f"Model on device: {device}")

num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    print(f"Wrapped with DataParallel: {num_gpus} GPUs")

print("\n" + "="*60)
print("STEP 7: Test training iteration")
print("="*60)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
print("Starting training loop...")

start_time = time.time()
for batch_idx, (features, labels) in enumerate(tqdm(dataloader, desc="Training")):
    features = features.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    logits = model(features)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    
    if batch_idx == 0:
        print(f"\nFirst batch:")
        print(f"  Features: {features.shape}, device: {features.device}")
        print(f"  Logits: {logits.shape}, device: {logits.device}")
        print(f"  Loss: {loss.item():.4f}")
    
    if batch_idx >= 2:  # Only run 3 batches
        break

elapsed = time.time() - start_time
print(f"\n✓ Completed 3 batches in {elapsed:.2f} seconds")

print("\n" + "="*60)
print("STEP 8: Check GPU memory usage")
print("="*60)
for i in range(num_gpus):
    mem_allocated = torch.cuda.memory_allocated(i) / 1024**2
    mem_reserved = torch.cuda.memory_reserved(i) / 1024**2
    print(f"GPU {i}: {mem_allocated:.1f} MB allocated, {mem_reserved:.1f} MB reserved")

print("\n" + "="*60)
print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
print("Training loop works correctly with CUDA fix")
print("="*60)
