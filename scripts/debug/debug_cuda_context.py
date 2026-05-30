#!/usr/bin/env python3
"""
Test CUDA context initialization.
"""

import os
import sys

# Set CUDA devices before import
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import torch
import torch.nn as nn

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

print("\nInitializing CUDA context on all GPUs...")
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        torch.cuda.init()
        # Create a small tensor to force context creation
        dummy = torch.zeros(1).cuda()
        print(f"  GPU {i}: Context initialized, tensor device: {dummy.device}")

print("\nCreating simple model...")
model = nn.Linear(10, 5)
device = torch.device('cuda:0')
model = model.to(device)
print(f"Model on {device}")

print("\nWrapping with DataParallel...")
model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
print("DataParallel created")

print("\nTesting forward pass...")
x = torch.randn(16, 10).to(device)
print(f"Input shape: {x.shape}, device: {x.device}")

y = model(x)
print(f"Output shape: {y.shape}, device: {y.device}")

print("\n✓ SUCCESS!")
