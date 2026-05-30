#!/usr/bin/env python3
"""
Test DDP training with tiny subset to validate num_workers=0 fix.
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_action_transformer_ddp import ActionDataset, TemporalTransformer


def setup(rank, world_size):
    """Initialize DDP."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up DDP."""
    dist.destroy_process_group()


def test_dataloader(rank, world_size):
    """Test dataloader with tiny subset."""
    setup(rank, world_size)
    
    print(f"[Rank {rank}] Starting test...", flush=True)
    
    # Create tiny dataset (just load the class list, don't scan all files)
    features_dir = Path("data/processed/features_kinetics700")
    
    # Manually create a tiny dataset with just 10 samples from one class
    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, features_dir, split='train'):
            self.sequence_length = 300
            # Get just one action class
            action_dir = list((features_dir / split).iterdir())[0]
            npz_files = list(action_dir.glob('*.npz'))[:10]  # Only 10 files
            self.samples = [(f, 0) for f in npz_files]
            print(f"[Rank {rank}] Created tiny dataset with {len(self.samples)} samples from {action_dir.name}", flush=True)
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            npz_path, label = self.samples[idx]
            data = np.load(npz_path)
            features = data['features']
            features = features.reshape(features.shape[0], -1)
            
            num_frames = features.shape[0]
            if num_frames < self.sequence_length:
                padding = np.zeros((self.sequence_length - num_frames, features.shape[1]))
                features = np.vstack([features, padding])
            else:
                features = features[:self.sequence_length]
            
            return torch.FloatTensor(features), label
    
    # Create dataset
    dataset = TinyDataset(features_dir, 'train')
    
    # Create sampler and dataloader
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    print(f"[Rank {rank}] Creating DataLoader with num_workers=0...", flush=True)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0, pin_memory=True)
    
    print(f"[Rank {rank}] DataLoader created, attempting to iterate...", flush=True)
    
    # Try to iterate through first few batches
    start_time = time.time()
    for i, (features, labels) in enumerate(dataloader):
        elapsed = time.time() - start_time
        print(f"[Rank {rank}] ✅ Batch {i}: features={features.shape}, labels={labels.shape}, elapsed={elapsed:.2f}s", flush=True)
        
        if i >= 2:  # Just do 3 batches
            break
    
    total_time = time.time() - start_time
    print(f"[Rank {rank}] 🎉 SUCCESS! Loaded 3 batches in {total_time:.2f}s", flush=True)
    
    cleanup()


def main():
    """Run test on 2 GPUs."""
    world_size = 2
    
    print("=" * 60)
    print("Testing DDP DataLoader with num_workers=0")
    print("=" * 60)
    
    torch.multiprocessing.spawn(
        test_dataloader,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
    
    print("=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
