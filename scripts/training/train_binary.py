#!/usr/bin/env python3
"""
Binary Classification Training: Engagement vs Disengagement

This is the simplest possible task - validates if pose features are sufficient.
If this fails (<75% accuracy), pose features alone are insufficient.
If this succeeds (>75% accuracy), we can proceed to macro (13 classes) or fine (86 classes).

Architecture: Same as Phase 1 (2 layers, 4 heads, 128d, ~683K params)
Classes: 2 (engagement, disengagement)
Target: >75% validation accuracy
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


class BinaryActionDataset(Dataset):
    """Dataset for binary engagement/disengagement classification."""
    
    corrupted_files = []
    
    def __init__(self, features_dir, split='train', sequence_length=300, hierarchical_labels_path=None, verbose=False):
        self.features_dir = Path(features_dir) / split
        self.sequence_length = sequence_length
        self.split = split
        
        # Load hierarchical labels
        if hierarchical_labels_path is None:
            hierarchical_labels_path = Path("models/action_transformer_kinetics700/hierarchical_labels.json")
        
        with open(hierarchical_labels_path) as f:
            self.label_hierarchy = json.load(f)
        
        self.fine_to_binary = self.label_hierarchy['conversions']['fine_to_binary']
        self.binary_to_idx = self.label_hierarchy['binary']['action_to_idx']
        self.idx_to_binary = {int(k): v for k, v in self.label_hierarchy['binary']['idx_to_action'].items()}
        
        if verbose:
            log(f"🔍 Scanning {split} dataset from {self.features_dir}")
            log(f"📊 Binary labels: {list(self.binary_to_idx.keys())}")
        
        start_time = time.time()
        
        self.samples = []
        
        action_dirs = sorted([d for d in self.features_dir.iterdir() if d.is_dir()])
        if verbose:
            log(f"📂 Found {len(action_dirs)} fine-grained action classes")
        
        engagement_count = 0
        disengagement_count = 0
        
        for action_dir in action_dirs:
            fine_action = action_dir.name
            
            # Map fine action → binary label
            if fine_action not in self.fine_to_binary:
                if verbose:
                    log(f"⚠️  Skipping unmapped action: {fine_action}")
                continue
            
            binary_label = self.fine_to_binary[fine_action]
            binary_idx = self.binary_to_idx[binary_label]
            
            npz_files = list(action_dir.glob('*.npz'))
            for npz_file in npz_files:
                self.samples.append((npz_file, binary_idx))
            
            if binary_label == "engagement":
                engagement_count += len(npz_files)
            else:
                disengagement_count += len(npz_files)
        
        if verbose:
            elapsed = time.time() - start_time
            log(f"✅ Dataset ready: {len(self.samples):,} samples (Engagement: {engagement_count:,}, Disengagement: {disengagement_count:,}) ({elapsed:.1f}s)")
    
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
                'error': str(e)
            }
            BinaryActionDataset.corrupted_files.append(corrupted_entry)
            print(f"⚠️  SKIPPING CORRUPTED: {npz_path.name}", flush=True)
            features = np.zeros((self.sequence_length, 543, 3))
        
        if features is None:
            features = np.zeros((self.sequence_length, 543, 3))
        
        # Pad or truncate to fixed length
        if len(features) < self.sequence_length:
            pad_len = self.sequence_length - len(features)
            features = np.pad(features, ((0, pad_len), (0, 0), (0, 0)), mode='constant')
        elif len(features) > self.sequence_length:
            features = features[:self.sequence_length]
        
        # Flatten: (T, 543, 3) → (T, 1629)
        features = features.reshape(self.sequence_length, -1)
        
        return torch.FloatTensor(features), torch.LongTensor([label])[0]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ActionTransformer(nn.Module):
    def __init__(self, input_dim=1629, d_model=128, nhead=4, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch, seq_len, d_model)
        x = x.mean(dim=1)  # Global average pooling: (batch, d_model)
        x = self.dropout(x)
        x = self.fc(x)  # (batch, num_classes)
        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training")
    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validation"):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train binary engagement classifier')
    parser.add_argument('--features-dir', type=str, required=True, help='Path to features directory')
    parser.add_argument('--output-dir', type=str, default='models/action_transformer_binary', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=300, help='Sequence length')
    args = parser.parse_args()
    
    log("="*80)
    log("BINARY CLASSIFICATION TRAINING: Engagement vs Disengagement")
    log("="*80)
    log("Configuration:")
    log("  Model: 2 layers, 4 heads, 128d (~683K params)")
    log("  Classes: 2 (engagement, disengagement)")
    log("  Regularization: Dropout 0.3, Weight decay 0.01, Label smoothing 0.1")
    log(f"  Training: {args.epochs} epochs, batch_size={args.batch_size}")
    log(f"  Learning rate: {args.lr} with ReduceLROnPlateau scheduler")
    log(f"  Target: Validation acc > 75%")
    log("="*80)
    log("")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        log(f"   GPU: {torch.cuda.get_device_name(0)}")
        log(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log("")
    
    # Load datasets
    log("📦 Loading datasets...")
    train_dataset = BinaryActionDataset(
        args.features_dir, 
        split='train', 
        sequence_length=args.sequence_length,
        verbose=True
    )
    val_dataset = BinaryActionDataset(
        args.features_dir, 
        split='val', 
        sequence_length=args.sequence_length,
        verbose=True
    )
    log("")
    
    log(f"📊 Dataset stats:")
    log(f"   Training: {len(train_dataset):,} samples")
    log(f"   Validation: {len(val_dataset):,} samples")
    log(f"   Classes: 2 (engagement, disengagement)")
    log("")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    log("🏗️  Creating model...")
    model = ActionTransformer(
        input_dim=543*3,  # 543 landmarks * 3 coords
        d_model=128,
        nhead=4,
        num_layers=2,
        num_classes=2,  # BINARY
        dropout=0.3
    ).to(device)
    
    num_params = count_parameters(model)
    log(f"✅ Model initialized: 128d, 4 heads, 2 layers, dropout=0.3")
    log(f"📊 Total parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    log("")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    log("✅ Using CrossEntropyLoss with label_smoothing=0.1")
    log(f"✅ Using AdamW optimizer (lr={args.lr}, weight_decay=0.01)")
    log("✅ Using ReduceLROnPlateau scheduler (patience=5, factor=0.5)")
    log("")
    
    # Training
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0.0
    results = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_epoch': 0,
        'best_val_acc': 0.0
    }
    
    log("="*80)
    log("🚀 Starting training...")
    log("="*80)
    log("")
    
    for epoch in range(args.epochs):
        log("="*80)
        log(f"📅 EPOCH {epoch+1}/{args.epochs}")
        log("="*80)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        
        log(f"\n📊 Results:")
        log(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        log(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        log(f"   Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            results['best_epoch'] = epoch + 1
            results['best_val_acc'] = best_val_acc
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, output_dir / 'best_model.pth')
            
            log(f"\n🌟 New best model! Val Acc: {val_acc*100:.2f}%")
            
            # Save confusion matrix
            cm = confusion_matrix(val_labels, val_preds)
            log(f"\n📊 Confusion Matrix:")
            log(f"                Predicted")
            log(f"              Eng    Dis")
            log(f"   Actual Eng {cm[0,0]:5d}  {cm[0,1]:5d}")
            log(f"          Dis {cm[1,0]:5d}  {cm[1,1]:5d}")
            
            # Save classification report
            class_names = ['engagement', 'disengagement']
            report = classification_report(val_labels, val_preds, target_names=class_names)
            with open(output_dir / 'classification_report.txt', 'w') as f:
                f.write(report)
            log(f"\n{report}")
        
        log("")
    
    # Final summary
    log("="*80)
    log("✅ TRAINING COMPLETE")
    log("="*80)
    log(f"\n🏆 Best Results:")
    log(f"   Epoch: {results['best_epoch']}")
    log(f"   Validation Accuracy: {results['best_val_acc']*100:.2f}%")
    log(f"   Target: 75.00%")
    
    if results['best_val_acc'] >= 0.75:
        log(f"\n🎉 SUCCESS! Binary classification achieved target accuracy.")
        log(f"   ✅ Pose features ARE sufficient for engagement detection")
        log(f"   ✅ Next step: Try MACRO classification (13 classes)")
    else:
        log(f"\n⚠️  Binary classification below target.")
        log(f"   ❌ Pose features may be insufficient")
        log(f"   💡 Consider: Feature re-extraction or accept binary ceiling")
    
    # Save results
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save label mapping
    label_mapping = {
        'binary_to_idx': train_dataset.binary_to_idx,
        'idx_to_binary': train_dataset.idx_to_binary
    }
    with open(output_dir / 'label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    log(f"\n📂 Outputs saved to: {output_dir}")
    log("")


if __name__ == '__main__':
    main()
