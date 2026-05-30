#!/usr/bin/env python3
"""
Test DataParallel with explicit CUDA context initialization.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("CUDA devices:", torch.cuda.device_count())

print("\n⚡ CRITICAL: Initializing CUDA context on ALL GPUs before DataParallel...")
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        # Force CUDA context creation on this GPU
        _ = torch.tensor([1.0], device=f'cuda:{i}')
        print(f"  GPU {i}: Context initialized")

class SimpleDataset(Dataset):
    def __init__(self, features_dir, split='train'):
        self.features_dir = Path(features_dir) / split
        self.samples = []
        action_dirs = sorted([d for d in self.features_dir.iterdir() if d.is_dir()])[:2]
        for idx, action_dir in enumerate(action_dirs):
            npz_files = list(action_dir.glob('*.npz'))[:5]
            for npz_file in npz_files:
                self.samples.append((npz_file, idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npz_path, label = self.samples[idx]
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                features = data['features'].copy()
        except:
            features = np.zeros((300, 543, 3))
        
        features = features.reshape(features.shape[0], -1)
        if features.shape[0] < 300:
            padding = np.zeros((300 - features.shape[0], features.shape[1]))
            features = np.vstack([features, padding])
        elif features.shape[0] > 300:
            indices = np.linspace(0, features.shape[0] - 1, 300, dtype=int)
            features = features[indices]
        
        return torch.FloatTensor(features), label

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(543*3, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = x.mean(dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

print("\nCreating dataset...")
dataset = SimpleDataset('data/processed/features_kinetics700', split='train')
print(f"Dataset: {len(dataset)} samples")

print("Creating DataLoader...")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

print("Creating model...")
model = SimpleModel()
device = torch.device('cuda:0')
model = model.to(device)

print("Wrapping with DataParallel...")
num_gpus = torch.cuda.device_count()
model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
print(f"DataParallel with {num_gpus} GPUs")

print("\n🚀 Starting training with DataParallel...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

model.train()
for batch_idx, (features, labels) in enumerate(tqdm(dataloader, desc="Training")):
    features = features.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    logits = model(features)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    
    print(f"Batch {batch_idx}: loss={loss.item():.4f}")
    
    if batch_idx >= 1:
        break

print("\n✓✓✓ DATAPARALLEL TRAINING WORKS WITH CUDA INIT! ✓✓✓")
