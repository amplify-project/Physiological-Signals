#!/usr/bin/env python3
"""
PHASE 1 Training: Reduced model with regularization for quick validation.

Changes from original:
- Model: 2 layers, 4 heads, 128d (was 4 layers, 8 heads, 256d)
- Parameters: ~937K (was ~3.75M - 75% reduction)
- Optimizer: AdamW with weight_decay=0.01 (was Adam)
- Loss: Label smoothing 0.1 (was none)
- Scheduler: ReduceLROnPlateau (was none)
- Epochs: 30 (was 100) for quick validation
"""

import os
import sys

# Use only GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


def log(msg):
    """Print with immediate flush."""
    print(msg, flush=True)


class ActionDataset(Dataset):
    """Dataset for Kinetics action features."""
    
    corrupted_files = []
    
    def __init__(self, features_dir, split='train', sequence_length=300, verbose=False):
        self.features_dir = Path(features_dir) / split
        self.sequence_length = sequence_length
        self.split = split
        
        if verbose:
            log(f"🔍 Scanning {split} dataset from {self.features_dir}")
        
        start_time = time.time()
        
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
                log(f"  ✓ Processed {idx + 1}/{len(action_dirs)} classes | {len(self.samples):,} samples")
        
        if verbose:
            elapsed = time.time() - start_time
            log(f"✅ Dataset ready: {len(self.samples):,} samples from {len(action_dirs)} classes ({elapsed:.1f}s)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npz_path, label = self.samples[idx]
        
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                features = data['features'].copy()
        except Exception as e:
            corrupted_entry = {
                'file': str(npz_path),
                'split': self.split,
                'action': self.idx_to_action[label],
                'error': str(e)
            }
            ActionDataset.corrupted_files.append(corrupted_entry)
            print(f"⚠️  SKIPPING CORRUPTED: {npz_path.name}", flush=True)
            features = np.zeros((self.sequence_length, 543, 3))
        
        if features is None:
            features = np.zeros((self.sequence_length, 543, 3))
        
        # Flatten keypoints: (num_frames, 543*3)
        features = features.reshape(features.shape[0], -1)
        
        # Pad or truncate
        num_frames = features.shape[0]
        if num_frames < self.sequence_length:
            padding = np.zeros((self.sequence_length - num_frames, features.shape[1]))
            features = np.vstack([features, padding])
        elif num_frames > self.sequence_length:
            indices = np.linspace(0, num_frames - 1, self.sequence_length, dtype=int)
            features = features[indices]
        
        return torch.FloatTensor(features), label


class TemporalTransformer(nn.Module):
    """PHASE 1: Reduced Temporal Transformer with dropout."""
    
    def __init__(self, input_dim=543*3, num_classes=86, d_model=128, nhead=4, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        
        # Input projection with dropout
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model))
        
        # Transformer encoder (reduced from 4 layers to 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # 512
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head with dropout
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        log(f"✅ Model initialized: {d_model}d, {nhead} heads, {num_layers} layers, dropout={dropout}")
        total_params = sum(p.numel() for p in self.parameters())
        log(f"📊 Total parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
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
        
        optimizer.zero_grad()
        
        logits = model(features)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
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
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Phase 1 Training: Reduced model for validation')
    parser.add_argument('--features-dir', type=str, required=True, help='Path to features directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--output-dir', type=str, default='models/action_transformer_phase1', 
                       help='Output directory for model')
    
    args = parser.parse_args()
    
    log("=" * 80)
    log("PHASE 1 TRAINING: Quick Validation with Reduced Model")
    log("=" * 80)
    log(f"Configuration:")
    log(f"  Model: 2 layers, 4 heads, 128d (~937K params)")
    log(f"  Regularization: Dropout 0.3, Weight decay 0.01, Label smoothing 0.1")
    log(f"  Training: {args.epochs} epochs, batch_size={args.batch_size}")
    log(f"  Learning rate: {args.lr} with ReduceLROnPlateau scheduler")
    log(f"  Expected duration: ~2 hours")
    log(f"  Target: Validation acc > 20% with train/val gap < 30%")
    log("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        log(f"   GPU: {torch.cuda.get_device_name(0)}")
        log(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    log("\n📦 Loading datasets...")
    train_dataset = ActionDataset(args.features_dir, split='train', verbose=True)
    val_dataset = ActionDataset(args.features_dir, split='val', verbose=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    log(f"\n📊 Dataset stats:")
    log(f"   Training: {len(train_dataset):,} samples")
    log(f"   Validation: {len(val_dataset):,} samples")
    log(f"   Classes: {len(train_dataset.action_to_idx)}")
    
    # Create model
    log("\n🏗️  Creating model...")
    model = TemporalTransformer(
        input_dim=543*3,
        num_classes=len(train_dataset.action_to_idx),
        d_model=128,        # REDUCED from 256
        nhead=4,            # REDUCED from 8
        num_layers=2,       # REDUCED from 4
        dropout=0.3         # ADDED (was 0.1)
    ).to(device)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    log("✅ Using CrossEntropyLoss with label_smoothing=0.1")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    log(f"✅ Using AdamW optimizer (lr={args.lr}, weight_decay=0.01)")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    log("✅ Using ReduceLROnPlateau scheduler (patience=5, factor=0.5)")
    
    # Training loop
    log("\n" + "=" * 80)
    log("🚀 Starting training...")
    log("=" * 80)
    
    best_val_acc = 0.0
    results = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    for epoch in range(1, args.epochs + 1):
        log(f"\n{'=' * 80}")
        log(f"📅 EPOCH {epoch}/{args.epochs}")
        log(f"{'=' * 80}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log results
        log(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        log(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        log(f"LR: {current_lr:.6f}")
        
        gap = train_acc - val_acc
        log(f"📊 Train/Val Gap: {gap:.4f} ({gap*100:.1f}%)")
        
        # Save results
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        results['lr'].append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            log(f"💾 New best validation accuracy: {val_acc:.4f} - Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, output_dir / 'best_model.pth')
            
            # Save classification report
            report = classification_report(
                val_labels, val_preds,
                target_names=[train_dataset.idx_to_action[i] for i in range(len(train_dataset.idx_to_action))],
                zero_division=0
            )
            with open(output_dir / 'classification_report.txt', 'w') as f:
                f.write(report)
    
    # Save final results
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save label mapping
    label_mapping = {
        'action_to_idx': train_dataset.action_to_idx,
        'idx_to_action': train_dataset.idx_to_action
    }
    with open(output_dir / 'label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    log("\n" + "=" * 80)
    log("📊 PHASE 1 TRAINING SUMMARY")
    log("=" * 80)
    log(f"🎉 Training complete!")
    log(f"✅ Best validation accuracy: {best_val_acc:.4f}")
    log(f"💾 Models saved to: {output_dir}")
    
    if ActionDataset.corrupted_files:
        log(f"⚠️  {len(ActionDataset.corrupted_files)} corrupted files detected")
    else:
        log("✅ No corrupted files detected!")
    
    log("\n🎯 PHASE 1 DECISION:")
    if best_val_acc > 0.20:
        log(f"   ✅ SUCCESS: Val acc = {best_val_acc:.1%} > 20%")
        log(f"   → Proceed to Phase 2 (full training with augmentation)")
    else:
        log(f"   ❌ INSUFFICIENT: Val acc = {best_val_acc:.1%} < 20%")
        log(f"   → Consider: Binary classification OR Holistic features")
    
    log("=" * 80)


if __name__ == '__main__':
    main()
