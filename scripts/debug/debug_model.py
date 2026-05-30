#!/usr/bin/env python3
"""
Debug script to test model initialization and forward pass.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

print("="*60)
print("STEP 1: Define model (same as training script)")
print("="*60)

class TemporalTransformer(nn.Module):
    """Temporal Transformer for action recognition."""
    
    def __init__(self, input_dim=543*3, num_classes=86, d_model=256, nhead=8, 
                 num_layers=4, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Project input
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.fc(x)  # (batch_size, num_classes)
        
        return logits

print("Model class defined")

print("\n" + "="*60)
print("STEP 2: Initialize model")
print("="*60)
model = TemporalTransformer(
    input_dim=543*3,
    num_classes=86,
    d_model=256,
    nhead=8,
    num_layers=4,
    dropout=0.1
)
print(f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters")

print("\n" + "="*60)
print("STEP 3: Test CPU forward pass")
print("="*60)
dummy_input = torch.randn(2, 300, 1629)  # batch_size=2, seq_len=300, features=1629
print(f"Input shape: {dummy_input.shape}")
output = model(dummy_input)
print(f"Output shape: {output.shape}")
print("✓ CPU forward pass successful")

print("\n" + "="*60)
print("STEP 4: Move model to GPU")
print("="*60)
device = torch.device('cuda:0')
model = model.to(device)
print(f"Model moved to {device}")

print("\n" + "="*60)
print("STEP 5: Test GPU forward pass")
print("="*60)
dummy_input_gpu = dummy_input.to(device)
print(f"Input moved to GPU: {dummy_input_gpu.device}")
output_gpu = model(dummy_input_gpu)
print(f"Output shape: {output_gpu.shape}")
print(f"Output device: {output_gpu.device}")
print("✓ GPU forward pass successful")

print("\n" + "="*60)
print("STEP 6: Wrap with DataParallel (8 GPUs)")
print("="*60)
model_parallel = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
print("Model wrapped with DataParallel")

print("\n" + "="*60)
print("STEP 7: Test DataParallel forward pass")
print("="*60)
print("Creating larger batch to utilize all GPUs...")
dummy_input_large = torch.randn(16, 300, 1629).to(device)  # batch_size=16
print(f"Input shape: {dummy_input_large.shape}")
print("Running forward pass...")
output_parallel = model_parallel(dummy_input_large)
print(f"Output shape: {output_parallel.shape}")
print("✓ DataParallel forward pass successful")

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
