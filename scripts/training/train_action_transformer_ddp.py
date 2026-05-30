#!/usr/bin/env python3
"""
Train Temporal Transformer for Kinetics-700 action recognition with DistributedDataParallel.
Multi-GPU training across all available GPUs.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import json
import argparse
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg, rank=None):
    """Print with immediate flush for real-time visibility."""
    if rank is None or rank == 0:
        print(msg, flush=True)


class ActionDataset(Dataset):
    """Dataset for Kinetics action features."""
    
    def __init__(self, features_dir, split='train', sequence_length=300, verbose=False, rank=0):
        """
        Args:
            features_dir: Path to features_kinetics700 directory
            split: 'train' or 'val'
            sequence_length: Number of frames per sequence (default 300 = 10 sec at 30fps)
            verbose: Print progress during initialization
            rank: DDP rank for logging
        """
        self.features_dir = Path(features_dir) / split
        self.sequence_length = sequence_length
        
        if verbose and rank == 0:
            log(f"🔍 Scanning {split} dataset from {self.features_dir}")
            log(f"⏳ This may take 2-3 minutes for ~70,000 files...")
        
        start_time = time.time()
        
        # Get all feature files and create label mapping
        self.samples = []
        self.action_to_idx = {}
        self.idx_to_action = {}
        
        action_dirs = sorted([d for d in self.features_dir.iterdir() if d.is_dir()])
        if verbose and rank == 0:
            log(f"📂 Found {len(action_dirs)} action classes")
        
        for idx, action_dir in enumerate(action_dirs):
            self.action_to_idx[action_dir.name] = idx
            self.idx_to_action[idx] = action_dir.name
            
            npz_files = list(action_dir.glob('*.npz'))
            for npz_file in npz_files:
                self.samples.append((npz_file, idx))
            
            if verbose and rank == 0 and (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                log(f"  ✓ Processed {idx + 1}/{len(action_dirs)} classes | {len(self.samples):,} samples | {elapsed:.1f}s elapsed")
        
        if verbose and rank == 0:
            elapsed = time.time() - start_time
            log(f"✅ Dataset ready: {len(self.samples):,} total samples from {len(action_dirs)} classes")
            log(f"⏱️  Loading took {elapsed:.1f} seconds")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npz_path, label = self.samples[idx]
        
        # Load features with memory mapping for concurrent access
        with np.load(npz_path, mmap_mode='r', allow_pickle=False) as data:
            features = np.array(data['features'])  # Copy to avoid mmap issues
        
        # Flatten keypoints: (num_frames, 543*3)
        features = features.reshape(features.shape[0], -1)
        
        # Pad or truncate to sequence_length
        num_frames = features.shape[0]
        if num_frames < self.sequence_length:
            # Pad with zeros
            padding = np.zeros((self.sequence_length - num_frames, features.shape[1]))
            features = np.vstack([features, padding])
        elif num_frames > self.sequence_length:
            # Take evenly spaced frames
            indices = np.linspace(0, num_frames - 1, self.sequence_length, dtype=int)
            features = features[indices]
        
        return torch.FloatTensor(features), label


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
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer encoder
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits


def setup_ddp(rank, world_size):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    # Port should already be set by main process before spawning
    log(f"[Rank {rank}] Attempting to connect to MASTER_PORT={os.environ.get('MASTER_PORT', 'NOT SET')}")
    
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        log(f"[Rank {rank}] ✅ Successfully joined process group on GPU {rank}")
    except Exception as e:
        log(f"[Rank {rank}] ❌ Failed to join process group: {e}")
        raise


def cleanup_ddp():
    """Cleanup DDP process group."""
    dist.destroy_process_group()


def train_epoch(model, dataloader, criterion, optimizer, device, rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    if rank == 0:
        pbar = tqdm(dataloader, desc='Training')
    else:
        pbar = dataloader
    
    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        if rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, rank):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(dataloader, desc='Validation')
        else:
            pbar = dataloader
            
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels


def train_ddp(rank, world_size, args):
    """Main training function for each DDP process."""
    # Setup DDP
    log(f"\n[Rank {rank}] Initializing DDP process...", rank)
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        log(f"\n{'='*60}")
        log(f"🚀 DISTRIBUTED TRAINING SETUP")
        log(f"{'='*60}")
        log(f"GPUs: {world_size}")
        log(f"Batch size per GPU: {args.batch_size}")
        log(f"Effective batch size: {args.batch_size * world_size}")
        log(f"{'='*60}\n")
        log(f"📊 Loading datasets from {args.features_dir}...")
        log(f"⏳ This takes 2-5 minutes for ~71,000 feature files...")
        log(f"{'='*60}\n")
    
    # Load datasets with verbose output (only rank 0 shows progress)
    log(f"[Rank {rank}] Starting dataset loading...", rank)
    train_dataset = ActionDataset(args.features_dir, split='train', 
                                   sequence_length=args.sequence_length,
                                   verbose=(rank == 0), rank=rank)
    log(f"[Rank {rank}] Train dataset loaded", rank)
    
    val_dataset = ActionDataset(args.features_dir, split='val',
                                 sequence_length=args.sequence_length,
                                 verbose=(rank == 0), rank=rank)
    log(f"[Rank {rank}] Validation dataset loaded", rank)
    
    # Save label mapping (only rank 0)
    if rank == 0:
        log(f"\n{'='*60}")
        log(f"📋 DATASET STATISTICS")
        log(f"{'='*60}")
        log(f"Training samples: {len(train_dataset):,}")
        log(f"Validation samples: {len(val_dataset):,}")
        log(f"Number of classes: {len(train_dataset.action_to_idx)}")
        log(f"Sequence length: {args.sequence_length} frames")
        log(f"{'='*60}\n")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        label_mapping = {
            'action_to_idx': train_dataset.action_to_idx,
            'idx_to_action': train_dataset.idx_to_action
        }
        with open(output_dir / 'label_mapping.json', 'w') as f:
            json.dump(label_mapping, f, indent=2)
        log(f"💾 Saved label mapping to {output_dir / 'label_mapping.json'}\n")
    
    log(f"[Rank {rank}] Creating distributed samplers...", rank)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, 
                                        rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, 
                                      rank=rank, shuffle=False)
    
    log(f"[Rank {rank}] Creating dataloaders...", rank)
    
    # Create dataloaders (num_workers=0 to avoid hanging issues)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=0, pin_memory=True)
    
    log(f"[Rank {rank}] Dataloaders ready", rank)
    
    # Initialize model
    num_classes = len(train_dataset.action_to_idx)
    if rank == 0:
        log(f"🧠 Initializing TemporalTransformer model...")
    
    log(f"[Rank {rank}] Creating model instance...", rank)
    model = TemporalTransformer(
        input_dim=543*3,
        num_classes=num_classes,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    log(f"[Rank {rank}] Model moved to GPU {rank}", rank)
    
    # Wrap with DDP
    log(f"[Rank {rank}] Wrapping model with DDP...", rank)
    model = DDP(model, device_ids=[rank])
    log(f"[Rank {rank}] DDP wrapper complete", rank)
    
    if rank == 0:
        log(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        log(f"   - Input dim: 543×3 = {543*3} keypoint features")
        log(f"   - Hidden dim: 256")
        log(f"   - Transformer layers: 4")
        log(f"   - Attention heads: 8")
        log(f"   - Output classes: {num_classes}\n")
    
    # Loss and optimizer
    log(f"[Rank {rank}] Creating optimizer and scheduler...", rank)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=5)
    log(f"[Rank {rank}] Optimizer ready", rank)
    
    # Training loop
    best_accuracy = 0.0
    results = []
    
    if rank == 0:
        log(f"{'='*60}")
        log(f"🏋️  STARTING TRAINING")
        log(f"{'='*60}")
        log(f"Epochs: {args.epochs}")
        log(f"Learning rate: {args.lr}")
        log(f"Batches per epoch: ~{len(train_loader)} train, ~{len(val_loader)} val")
        log(f"{'='*60}\n")
    
    log(f"[Rank {rank}] Entering training loop", rank)
    
    for epoch in range(args.epochs):
        # Set epoch for sampler (important for proper shuffling)
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"📅 EPOCH {epoch+1}/{args.epochs}")
            print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device, rank)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, 
                                                             criterion, device, rank)
        
        # Synchronize metrics across all processes
        if world_size > 1:
            # Average metrics across all GPUs
            metrics = torch.tensor([train_loss, train_acc, val_loss, val_acc], 
                                   device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= world_size
            train_loss, train_acc, val_loss, val_acc = metrics.tolist()
        
        # Learning rate scheduling (only on rank 0, then broadcast)
        if rank == 0:
            scheduler.step(val_acc)
        
        # Print metrics (only rank 0)
        if rank == 0:
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save results
            results.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                output_dir = Path(args.output_dir)
                
                # Save unwrapped model (remove DDP wrapper)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'action_to_idx': train_dataset.action_to_idx,
                    'idx_to_action': train_dataset.idx_to_action
                }, output_dir / 'best_model.pth')
                print(f"✅ Saved best model (val_acc: {val_acc:.4f})")
                
                # Generate classification report
                report = classification_report(
                    val_labels, val_preds,
                    target_names=[train_dataset.idx_to_action[i] for i in range(num_classes)],
                    zero_division=0
                )
                with open(output_dir / 'classification_report.txt', 'w') as f:
                    f.write(report)
    
    # Save final results (only rank 0)
    if rank == 0:
        output_dir = Path(args.output_dir)
        with open(output_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Training complete!")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        print(f"Models saved to: {output_dir}")
    
    # Cleanup
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-dir', type=str, required=True, 
                        help='Path to features_kinetics700 directory')
    parser.add_argument('--output-dir', type=str, default='models/action_transformer_kinetics700',
                        help='Output directory for models')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sequence-length', type=int, default=300,
                        help='Number of frames per sequence (300 = 10 sec at 30fps)')
    args = parser.parse_args()
    
    # When using torchrun, rank and world_size come from environment variables
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running with torchrun - use environment variables
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        train_ddp(rank, world_size, args)
    else:
        # Fallback: manual spawn (for testing without torchrun)
        world_size = torch.cuda.device_count()
        
        if world_size == 0:
            print("No GPUs available! Exiting...")
            return
        
        import random
        master_port = str(random.randint(20000, 30000))
        os.environ['MASTER_PORT'] = master_port
        os.environ['MASTER_ADDR'] = 'localhost'
        
        print(f"🚀 Starting DDP training on {world_size} GPUs")
        print(f"📡 Master port: {master_port}")
        print(f"{'='*60}\n")
        
        torch.multiprocessing.spawn(
            train_ddp,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )


if __name__ == '__main__':
    main()
