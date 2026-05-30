#!/usr/bin/env python3
"""
Quick test if DDP (DistributedDataParallel) works on this system.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

print("Testing DDP initialization...")

# Initialize process group
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '2'  # Test with 2 GPUs
os.environ['RANK'] = '0'

dist.init_process_group(backend='nccl')

print(f"DDP initialized: rank {dist.get_rank()}/{dist.get_world_size()}")

# Simple model
model = nn.Linear(10, 5).cuda()
ddp_model = DDP(model, device_ids=[0])

print("DDP model created")

# Test forward pass
x = torch.randn(4, 10).cuda()
y = ddp_model(x)

print(f"Forward pass successful: {y.shape}")
print("✓ DDP WORKS!")

dist.destroy_process_group()
