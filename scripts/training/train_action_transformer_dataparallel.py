#!/usr/bin/env python3
"""
Train Temporal Transformer for Kinetics-700 using SINGLE GPU.
DataParallel has CUDA context issues on this system - using single GPU for now.
"""

import os
import sys

# Use only GPU 0 for now (DataParallel hangs on this system)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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


def log(msg):
    """Print with immediate flush for real-time visibility."""
    print(msg, flush=True)


class ActionDataset(Dataset):
    """Dataset for Kinetics action features."""
    
    # Class-level tracking for corrupted files across all dataset instances
    corrupted_files = []
    
    def __init__(self, features_dir, split='train', sequence_length=300, verbose=False):
        """
        Args:
            features_dir: Path to features_kinetics700 directory
            split: 'train' or 'val'
            sequence_length: Number of frames per sequence (default 300 = 10 sec at 30fps)
            verbose: Print progress during initialization
        """
        self.features_dir = Path(features_dir) / split
        self.sequence_length = sequence_length
        self.split = split  # Track which split this is
        
        if verbose:
            log(f"🔍 Scanning {split} dataset from {self.features_dir}")
            log(f"⏳ This may take 2-3 minutes for ~70,000 files...")
        
        start_time = time.time()
        
        # Get all feature files and create label mapping
        self.samples = []
        self.action_to_idx = {}
        self.idx_to_action = {}
        
        action_dirs = sorted([d for d in self.features_dir.iterdir() if d.is_dir()])
        if verbose:
            log(f"📂 Found {len(action_dirs)} action classes")
        
        for idx, action_dir in enumerate(action_dirs):
            self.action_to_idx[action_dir.name] = idx
            self.idx_to_action[idx] = action_dir.name
            
            npz_files = list(action_dir.glob('*.npz'))
            for npz_file in npz_files:
                self.samples.append((npz_file, idx))
            
            if verbose and (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                log(f"  ✓ Processed {idx + 1}/{len(action_dirs)} classes | {len(self.samples):,} samples | {elapsed:.1f}s elapsed")
        
        if verbose:
            elapsed = time.time() - start_time
            log(f"✅ Dataset ready: {len(self.samples):,} total samples from {len(action_dirs)} classes")
            log(f"⏱️  Loading took {elapsed:.1f} seconds")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npz_path, label = self.samples[idx]
        
        # Load features - SKIP CRC CHECK, just try to load quickly
        features = None
        
        try:
            # Load directly with np.load - if it hangs, DataLoader timeout will catch it
            with np.load(npz_path, allow_pickle=False) as data:
                features = data['features'].copy()  # (num_frames, 543, 3)
                
        except Exception as e:
            # File is corrupted or unreadable
            corrupted_entry = {
                'file': str(npz_path),
                'split': self.split,
                'action': self.idx_to_action[label],
                'error': str(e)
            }
            ActionDataset.corrupted_files.append(corrupted_entry)
            print(f"⚠️  SKIPPING CORRUPTED FILE: {npz_path.name} - {type(e).__name__}: {str(e)[:100]}", flush=True)
            
            # Return zeros to avoid crashing training
            features = np.zeros((self.sequence_length, 543, 3))
        
        # Ensure features is not None (shouldn't happen, but for type safety)
        if features is None:
            features = np.zeros((self.sequence_length, 543, 3))
        
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


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
            
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-dir', type=str, required=True, 
                        help='Path to features_kinetics700 directory')
    parser.add_argument('--output-dir', type=str, default='models/action_transformer_kinetics700',
                        help='Output directory for models')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (will be split across GPUs)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sequence-length', type=int, default=300,
                        help='Number of frames per sequence (300 = 10 sec at 30fps)')
    args = parser.parse_args()
    
    # Check GPU
    if not torch.cuda.is_available():
        log("❌ No GPU available! Exiting...")
        return
    
    device = torch.device('cuda:0')
    log(f"🚀 Using single GPU: cuda:0 (DataParallel has issues on this system)")
    log(f"📊 Batch size: {args.batch_size}\n")
    
    log(f"{'='*60}")
    log(f"📊 Loading datasets from {args.features_dir}...")
    log(f"⏳ This takes 2-5 minutes for ~71,000 feature files...")
    log(f"{'='*60}\n")
    
    # Load datasets
    train_dataset = ActionDataset(args.features_dir, split='train', 
                                   sequence_length=args.sequence_length, verbose=True)
    val_dataset = ActionDataset(args.features_dir, split='val',
                                 sequence_length=args.sequence_length, verbose=True)
    
    log(f"\n{'='*60}")
    log(f"📋 DATASET STATISTICS")
    log(f"{'='*60}")
    log(f"Training samples: {len(train_dataset):,}")
    log(f"Validation samples: {len(val_dataset):,}")
    log(f"Number of classes: {len(train_dataset.action_to_idx)}")
    log(f"Sequence length: {args.sequence_length} frames")
    log(f"{'='*60}\n")
    
    # Save label mapping
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    label_mapping = {
        'action_to_idx': train_dataset.action_to_idx,
        'idx_to_action': train_dataset.idx_to_action
    }
    with open(output_dir / 'label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    log(f"💾 Saved label mapping to {output_dir / 'label_mapping.json'}\n")
    
    # Create dataloaders with num_workers=4 and timeout to catch hangs
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True,
                              timeout=30)  # 30 second timeout per batch
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True,
                            timeout=30)
    
    # Initialize model
    num_classes = len(train_dataset.action_to_idx)
    log(f"🧠 Initializing TemporalTransformer model...")
    
    model = TemporalTransformer(
        input_dim=543*3,
        num_classes=num_classes,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.1
    )
    
    # Move to GPU (no DataParallel - it hangs on this system)
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    log(f"✅ Model initialized with {param_count:,} parameters")
    log(f"   - Input dim: 543×3 = {543*3} keypoint features")
    log(f"   - Hidden dim: 256")
    log(f"   - Transformer layers: 4")
    log(f"   - Attention heads: 8")
    log(f"   - Output classes: {num_classes}")
    log(f"   - Device: {device}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=5)
    
    # Training loop
    best_accuracy = 0.0
    results = []
    
    log(f"{'='*60}")
    log(f"🏋️  STARTING TRAINING")
    log(f"{'='*60}")
    log(f"Epochs: {args.epochs}")
    log(f"Batch size: {args.batch_size}")
    log(f"Learning rate: {args.lr}")
    log(f"Batches per epoch: ~{len(train_loader)} train, ~{len(val_loader)} val")
    log(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        log(f"\n{'='*60}")
        log(f"📅 EPOCH {epoch+1}/{args.epochs}")
        log(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, 
                                                             criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Print metrics
        log(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        log(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        log(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
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
            
            # Unwrap DataParallel if needed
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'action_to_idx': train_dataset.action_to_idx,
                'idx_to_action': train_dataset.idx_to_action
            }, output_dir / 'best_model.pth')
            log(f"✅ Saved best model (val_acc: {val_acc:.4f})")
            
            # Generate classification report
            report = classification_report(
                val_labels, val_preds,
                target_names=[train_dataset.idx_to_action[i] for i in range(num_classes)],
                zero_division=0
            )
            with open(output_dir / 'classification_report.txt', 'w') as f:
                f.write(report)
    
    # Save final results
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Post-training summary: corrupted files
    log("\n" + "=" * 60)
    log("📊 TRAINING SUMMARY")
    log("=" * 60)
    log(f"🎉 Training complete!")
    log(f"✅ Best validation accuracy: {best_accuracy:.4f}")
    log(f"💾 Models saved to: {output_dir}")
    
    if ActionDataset.corrupted_files:
        log(f"\n⚠️  CORRUPTED FILES DETECTED: {len(ActionDataset.corrupted_files)}")
        log("=" * 60)
        log("These .npz files had bad CRC-32 checksums (corrupted zip archives):")
        log("They were replaced with zeros during training.\n")
        
        # Group by split
        train_corrupted = [f for f in ActionDataset.corrupted_files if f['split'] == 'train']
        val_corrupted = [f for f in ActionDataset.corrupted_files if f['split'] == 'val']
        
        if train_corrupted:
            log(f"📁 TRAIN SET ({len(train_corrupted)} corrupted):")
            for entry in train_corrupted[:20]:  # Show first 20
                log(f"  • {Path(entry['file']).name} ({entry['action']})")
            if len(train_corrupted) > 20:
                log(f"  ... and {len(train_corrupted) - 20} more")
        
        if val_corrupted:
            log(f"\n📁 VAL SET ({len(val_corrupted)} corrupted):")
            for entry in val_corrupted[:20]:  # Show first 20
                log(f"  • {Path(entry['file']).name} ({entry['action']})")
            if len(val_corrupted) > 20:
                log(f"  ... and {len(val_corrupted) - 20} more")
        
        # Save full list to file
        corrupted_log = output_dir / 'corrupted_files.json'
        with open(corrupted_log, 'w') as f:
            json.dump(ActionDataset.corrupted_files, f, indent=2)
        log(f"\n📄 Full list saved to: {corrupted_log}")
        log("\n💡 Recommendation: Re-extract these videos and regenerate features")
        log("=" * 60)
    else:
        log("✅ No corrupted files detected!")
    
    log(f"\n{'=' * 60}\n")


if __name__ == '__main__':
    main()
