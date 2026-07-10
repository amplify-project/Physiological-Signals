#!/usr/bin/env python3
"""
Train Temporal Transformer with DDP using PRE-COMPUTED indices (v2).
This version fixes the large dataset hang by loading pre-shuffled indices.

CHANGES FROM V1:
- Uses PrecomputedActionDataset instead of ActionDataset
- Uses PrecomputedDistributedSampler instead of DistributedSampler
- Dataset loads instantly (no 2-5 minute scan)
- Eliminates race condition that causes DDP hang

PREREQUISITES:
Run once before training:
  python scripts/generate_dataset_indices.py --features-dir data/processed/features_kinetics700
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import json
import argparse

# Import precomputed dataset and sampler
sys.path.insert(0, str(Path(__file__).parent))
from ddp_precomputed_dataset import PrecomputedActionDataset, PrecomputedDistributedSampler

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg, rank=None):
    """Print with immediate flush."""
    if rank is None or rank == 0:
        print(msg, flush=True)


class TemporalTransformer(nn.Module):
    """Temporal Transformer for action recognition (same as v1)."""
    
    def __init__(self, input_dim=543*3, num_classes=86, d_model=256, nhead=8, 
                 num_layers=4, dropout=0.3):  # Increased from 0.1 to 0.3 for better regularization
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


def setup_ddp(rank, world_size):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    log(f"[Rank {rank}] Connecting to MASTER_PORT={os.environ.get('MASTER_PORT', 'NOT SET')}")
    
    try:
        # Try GLOO backend instead of NCCL - may fix .to(device) hang
        log(f"[Rank {rank}] 🔷 Using GLOO backend (CPU-based communication)", rank)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        log(f"[Rank {rank}] ✅ Joined process group with GLOO backend on GPU {rank}")
    except Exception as e:
        log(f"[Rank {rank}] ❌ Failed: {e}")
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
    
    log(f"[Rank {rank}] 🔷 train_epoch: About to create progress bar and iterate dataloader (len={len(dataloader)})", rank)
    
    if rank == 0:
        pbar = tqdm(dataloader, desc='Training')
    else:
        pbar = dataloader
    
    log(f"[Rank {rank}] 🔷 train_epoch: Starting batch enumeration loop...", rank)
    
    for batch_idx, (features, labels) in enumerate(pbar):
        if batch_idx == 0:
            log(f"[Rank {rank}] 🎉🎉 FIRST BATCH LOADED! batch_idx=0, features.shape={features.shape}, labels.shape={labels.shape}", rank)
            log(f"[Rank {rank}] 🔷 Moving batch to device {device}...", rank)
        
        features = features.to(device)
        labels = labels.to(device)
        
        if batch_idx == 0:
            log(f"[Rank {rank}] ✅ First batch on device. features on {features.device}, labels on {labels.device}", rank)
            log(f"[Rank {rank}] 🔷 Running forward pass...", rank)
        
        optimizer.zero_grad()
        logits = model(features)
        
        if batch_idx == 0:
            log(f"[Rank {rank}] ✅ Forward pass complete! logits.shape={logits.shape}", rank)
            log(f"[Rank {rank}] 🔷 Computing loss...", rank)
        
        loss = criterion(logits, labels)
        
        if batch_idx == 0:
            log(f"[Rank {rank}] ✅ Loss computed: {loss.item():.4f}", rank)
            log(f"[Rank {rank}] 🔷 Running backward pass...", rank)
        
        loss.backward()
        
        if batch_idx == 0:
            log(f"[Rank {rank}] ✅ Backward pass complete!", rank)
            log(f"[Rank {rank}] 🔷 Running optimizer step...", rank)
        
        optimizer.step()
        
        if batch_idx == 0:
            log(f"[Rank {rank}] 🎉🎉🎉 FIRST BATCH COMPLETE! Training is WORKING!", rank)
        
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
    log(f"[Rank {rank}] 🔷 train_ddp() function called", rank)
    log(f"[Rank {rank}] 🔷 Arguments: world_size={world_size}, batch_size={args.batch_size}", rank)
    
    log(f"[Rank {rank}] 🔷 Calling setup_ddp()...", rank)
    setup_ddp(rank, world_size)
    log(f"[Rank {rank}] ✅ setup_ddp() completed", rank)
    
    device = torch.device(f'cuda:{rank}')
    log(f"[Rank {rank}] 🔷 Device set to: {device}", rank)
    
    if rank == 0:
        log(f"\n{'='*80}")
        log(f"🚀 DDP TRAINING V2 (Pre-computed Indices)")
        log(f"{'='*80}")
        log(f"GPUs: {world_size}")
        log(f"Batch size per GPU: {args.batch_size}")
        log(f"Effective batch size: {args.batch_size * world_size}")
        log(f"Indices directory: {args.indices_dir}")
        log(f"Features directory: {args.features_dir}")
        log(f"Output directory: {args.output_dir}")
        log(f"Epochs: {args.epochs}")
        log(f"Learning rate: {args.lr}")
        log(f"{'='*80}\n")
    
    # Load datasets using PRECOMPUTED indices
    log(f"[Rank {rank}] 🔷 About to load train dataset...", rank)
    import time
    start_time = time.time()
    log(f"[Rank {rank}] 🔷 Creating PrecomputedActionDataset for TRAIN split...", rank)
    
    try:
        train_dataset = PrecomputedActionDataset(
            args.features_dir, 
            split='train',
            indices_dir=args.indices_dir,
            sequence_length=args.sequence_length,
            verbose=(rank == 0),
            rank=rank,
            label_level=args.label_level,
            use_cleaned_labels=args.use_cleaned_labels
        )
        elapsed = time.time() - start_time
        log(f"[Rank {rank}] ✅ Train dataset loaded in {elapsed:.2f}s", rank)
        log(f"[Rank {rank}] 🔷 Train dataset size: {len(train_dataset):,} samples", rank)
        log(f"[Rank {rank}] 🔷 Num classes ({args.label_level}): {train_dataset.num_classes}", rank)
        
        log(f"[Rank {rank}] 🔷 Creating PrecomputedActionDataset for VAL split...", rank)
        val_start = time.time()
        val_dataset = PrecomputedActionDataset(
            args.features_dir,
            split='val',
            indices_dir=args.indices_dir,
            sequence_length=args.sequence_length,
            verbose=(rank == 0),
            rank=rank,
            label_level=args.label_level,
            use_cleaned_labels=args.use_cleaned_labels
        )
        val_elapsed = time.time() - val_start
        log(f"[Rank {rank}] ✅ Val dataset loaded in {val_elapsed:.2f}s", rank)
        log(f"[Rank {rank}] 🔷 Val dataset size: {len(val_dataset):,} samples", rank)
    except FileNotFoundError as e:
        log(f"[Rank {rank}] ❌ FileNotFoundError: {e}", rank)
        if rank == 0:
            log(f"\n❌ ERROR: {e}")
            log(f"\nPlease run first:")
            log(f"  python scripts/generate_dataset_indices.py --features-dir {args.features_dir}")
        raise
    except Exception as e:
        log(f"[Rank {rank}] ❌ Unexpected error loading datasets: {e}", rank)
        import traceback
        log(f"[Rank {rank}] Traceback:\n{traceback.format_exc()}", rank)
        raise
    
    total_load_time = time.time() - start_time
    log(f"[Rank {rank}] ✅ All datasets loaded in {total_load_time:.2f}s total", rank)
    
    if rank == 0:
        log(f"\n{'='*80}")
        log(f"📋 DATASET STATISTICS")
        log(f"{'='*80}")
        log(f"Training samples: {len(train_dataset):,}")
        log(f"Validation samples: {len(val_dataset):,}")
        log(f"Classes: {len(train_dataset.action_to_idx)}")
        log(f"{'='*80}\n")
    
    # Create samplers (use regular DistributedSampler for cleaned labels since indices don't match)
    log(f"[Rank {rank}] 🔷 Creating distributed samplers...", rank)
    sampler_start = time.time()
    
    if args.use_cleaned_labels:
        # Use regular DistributedSampler for cleaned labels (dataset size changed)
        from torch.utils.data.distributed import DistributedSampler
        log(f"[Rank {rank}] 🔷 Using DistributedSampler (cleaned labels - dataset size changed)", rank)
        
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=42
        )
    else:
        # Use precomputed samplers for original dataset
        log(f"[Rank {rank}] 🔷 Creating PrecomputedDistributedSampler for TRAIN...", rank)
        train_sampler = PrecomputedDistributedSampler(
            dataset_size=len(train_dataset),
            split='train',
            indices_dir=args.indices_dir,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )
        log(f"[Rank {rank}] 🔷 Train sampler created. Num samples for this rank: {len(train_sampler)}", rank)
        
        log(f"[Rank {rank}] 🔷 Creating PrecomputedDistributedSampler for VAL...", rank)
        val_sampler = PrecomputedDistributedSampler(
            dataset_size=len(val_dataset),
            split='val',
            indices_dir=args.indices_dir,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=42
        )
        log(f"[Rank {rank}] 🔷 Val sampler created. Num samples for this rank: {len(val_sampler)}", rank)
    
    sampler_elapsed = time.time() - sampler_start
    log(f"[Rank {rank}] ✅ Samplers created in {sampler_elapsed:.2f}s", rank)
    
    # Create dataloaders
    log(f"[Rank {rank}] 🔷 Creating DataLoader for TRAIN...", rank)
    loader_start = time.time()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              sampler=train_sampler, num_workers=0, pin_memory=True)
    log(f"[Rank {rank}] 🔷 Train loader created. Batches: {len(train_loader)}", rank)
    
    log(f"[Rank {rank}] 🔷 Creating DataLoader for VAL...", rank)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=0, pin_memory=True)
    log(f"[Rank {rank}] 🔷 Val loader created. Batches: {len(val_loader)}", rank)
    
    loader_elapsed = time.time() - loader_start
    log(f"[Rank {rank}] ✅ Dataloaders created in {loader_elapsed:.2f}s", rank)
    
    # Initialize model
    num_classes = train_dataset.num_classes
    log(f"[Rank {rank}] 🔷 Number of classes: {num_classes}", rank)
    
    log(f"[Rank {rank}] 🔷 Creating TemporalTransformer model...", rank)
    model_start = time.time()
    model = TemporalTransformer(
        input_dim=543*3,
        num_classes=num_classes,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.1
    ).to(device)
    model_elapsed = time.time() - model_start
    log(f"[Rank {rank}] ✅ Model created and moved to GPU in {model_elapsed:.2f}s", rank)
    
    log(f"[Rank {rank}] 🔷 Wrapping model with DDP...", rank)
    ddp_start = time.time()
    model = DDP(model, device_ids=[rank])
    ddp_elapsed = time.time() - ddp_start
    log(f"[Rank {rank}] ✅ DDP wrapper applied in {ddp_elapsed:.2f}s", rank)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        log(f"✅ Model initialized: {total_params:,} params")
        log(f"   Architecture: 4 layers, 8 heads, 256d, dropout=0.1\n")
    
    # Optimizer
    log(f"[Rank {rank}] 🔷 Creating optimizer and scheduler...", rank)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=5)
    log(f"[Rank {rank}] ✅ Optimizer ready: AdamW, lr={args.lr}, weight_decay=0.01", rank)
    
    # Training loop
    best_accuracy = 0.0
    results = []
    
    if rank == 0:
        log(f"{'='*80}")
        log(f"🏋️  STARTING TRAINING")
        log(f"{'='*80}")
        log(f"Epochs: {args.epochs}")
        log(f"Batches/epoch: ~{len(train_loader)} train, ~{len(val_loader)} val")
        log(f"{'='*80}\n")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        log(f"[Rank {rank}] 🔷 EPOCH {epoch+1}/{args.epochs} - Setting sampler epoch...", rank)
        train_sampler.set_epoch(epoch)  # Important for shuffling
        log(f"[Rank {rank}] ✅ Sampler epoch set to {epoch}", rank)
        
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"📅 EPOCH {epoch+1}/{args.epochs}")
            print(f"{'='*80}")
        
        log(f"[Rank {rank}] 🔷 Starting training epoch...", rank)
        train_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device, rank)
        train_elapsed = time.time() - train_start
        log(f"[Rank {rank}] ✅ Training completed in {train_elapsed:.1f}s", rank)
        
        log(f"[Rank {rank}] 🔷 Starting validation...", rank)
        val_start = time.time()
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, 
                                                             criterion, device, rank)
        val_elapsed = time.time() - val_start
        log(f"[Rank {rank}] ✅ Validation completed in {val_elapsed:.1f}s", rank)
        
        # Synchronize metrics across GPUs
        log(f"[Rank {rank}] 🔷 Synchronizing metrics across {world_size} GPUs...", rank)
        if world_size > 1:
            metrics = torch.tensor([train_loss, train_acc, val_loss, val_acc], 
                                   device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= world_size
            train_loss, train_acc, val_loss, val_acc = metrics.tolist()
            log(f"[Rank {rank}] ✅ Metrics synchronized", rank)
        else:
            log(f"[Rank {rank}] ⚠️  Single GPU - no metric sync needed", rank)
        
        epoch_elapsed = time.time() - epoch_start
        log(f"[Rank {rank}] 🔷 Epoch {epoch+1} total time: {epoch_elapsed:.1f}s", rank)
        
        if rank == 0:
            scheduler.step(val_acc)
            
            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
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
                output_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'action_to_idx': train_dataset.action_to_idx,
                    'idx_to_action': train_dataset.idx_to_action
                }, output_dir / 'best_model.pth')
                print(f"✅ Saved best model (acc: {val_acc:.4f})")
                
                report = classification_report(
                    val_labels, val_preds,
                    target_names=[train_dataset.idx_to_action[i] for i in range(num_classes)],
                    zero_division=0
                )
                with open(output_dir / 'classification_report.txt', 'w') as f:
                    f.write(report)
    
    # Save results
    if rank == 0:
        output_dir = Path(args.output_dir)
        with open(output_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Training complete!")
        print(f"Best val acc: {best_accuracy:.4f}")
    
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='DDP Training V2 (Pre-computed Indices)')
    parser.add_argument('--features-dir', type=str, required=True,
                        help='Path to features_kinetics700 directory')
    parser.add_argument('--indices-dir', type=str, default='data/processed',
                        help='Directory with pre-computed indices (default: data/processed)')
    parser.add_argument('--output-dir', type=str, default='models/action_transformer_kinetics700_v2',
                        help='Output directory')
    parser.add_argument('--label-level', type=str, default='fine', choices=['fine', 'macro', 'binary'],
                        help='Label level: fine (86 classes), macro (13 classes), binary (2 classes)')
    parser.add_argument('--use-cleaned-labels', action='store_true',
                        help='Use cleaned labels v2 with quarantined ambiguous classes (only for binary/macro)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sequence-length', type=int, default=300)
    args = parser.parse_args()
    
    # Check if running with torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        train_ddp(rank, world_size, args)
    else:
        # Fallback: manual spawn
        world_size = torch.cuda.device_count()
        
        if world_size == 0:
            print("❌ No GPUs available!")
            return
        
        import random
        master_port = str(random.randint(20000, 30000))
        os.environ['MASTER_PORT'] = master_port
        os.environ['MASTER_ADDR'] = 'localhost'
        
        print(f"🚀 Starting DDP V2 on {world_size} GPUs")
        print(f"📡 Master port: {master_port}\n")
        
        torch.multiprocessing.spawn(
            train_ddp,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )


if __name__ == '__main__':
    main()
