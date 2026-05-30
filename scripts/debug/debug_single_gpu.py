#!/usr/bin/env python3
"""
Test training on SINGLE GPU (no DataParallel).
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Only GPU 0

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

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
        self.fc1 = nn.Linear(543*3, 256)
        self.fc2 = nn.Linear(256, 2)
    
    def forward(self, x):
        x = x.mean(dim=1)  # Average over time
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

print("Creating dataset...")
dataset = SimpleDataset('data/processed/features_kinetics700', split='train')
print(f"Dataset: {len(dataset)} samples")

print("Creating DataLoader...")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

print("Creating model...")
model = SimpleModel()
device = torch.device('cuda:0')
model = model.to(device)
print(f"Model on {device} - NO DataParallel")

print("Training...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

model.train()
for batch_idx, (features, labels) in enumerate(tqdm(dataloader)):
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

print("\n✓ SINGLE GPU TRAINING WORKS!")
