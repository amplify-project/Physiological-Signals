#!/usr/bin/env python3
"""
Train Temporal Transformer for Kinetics-700 action recognition on single GPU.
Fallback for when DDP has issues.
"""

import os
import sys
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
        
        # Load features
        data = np.load(npz_path)
        features = data['features']  # (num_frames, 543, 3)
        
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
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sequence-length', type=int, default=300,
                        help='Number of frames per sequence (300 = 10 sec at 30fps)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    log(f"🚀 Using device: {device}")
    
    log(f"\n{'='*60}")
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
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
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
    ).to(device)
    
    log(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    log(f"   - Input dim: 543×3 = {543*3} keypoint features")
    log(f"   - Hidden dim: 256")
    log(f"   - Transformer layers: 4")
    log(f"   - Attention heads: 8")
    log(f"   - Output classes: {num_classes}\n")
    
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
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
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
    
    log(f"\n🎉 Training complete!")
    log(f"Best validation accuracy: {best_accuracy:.4f}")
    log(f"Models saved to: {output_dir}")


if __name__ == '__main__':
    main()
