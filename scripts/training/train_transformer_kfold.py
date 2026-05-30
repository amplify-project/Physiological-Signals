"""
Train Temporal Transformer engagement model with k-fold cross-validation
Treats engagement prediction as regression problem for real-time AR feedback
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class EngagementDataset(Dataset):
    """Dataset loader for pose engagement features (DWPose or MediaPipe)"""
    
    def __init__(self, features_dir, labels, max_frames=300, num_keypoints=134):
        """
        Args:
            features_dir: Path to features directory (without split subdirs for k-fold)
            labels: Dictionary mapping video_id to engagement labels
            max_frames: Maximum number of frames to use
            num_keypoints: Number of keypoints per person (134 for DWPose, 543 for MediaPipe)
        """
        self.labels = labels
        self.max_frames = max_frames
        self.num_keypoints = num_keypoints
        
        # Load all valid feature files from all splits
        self.samples = []
        features_path = Path(features_dir)
        
        print(f"Loading dataset from {features_path}...")
        
        # Load from Train, Validation, Test subdirectories
        for split in ['Train', 'Validation', 'Test']:
            split_dir = features_path / split
            if not split_dir.exists():
                continue
                
            for npy_file in tqdm(sorted(split_dir.glob('**/*.npy')), desc=f"Loading {split}"):
                video_id = npy_file.stem + '.avi'  # Labels use .avi extension
                
                # Check if we have labels for this video
                if video_id not in labels:
                    continue
                
                # Try to load the file
                try:
                    data = np.load(npy_file, allow_pickle=True)
                    # Verify it's valid
                    # Support both DWPose format (frames, people, keypoints, coords) and MediaPipe format (frames, keypoints, coords)
                    if len(data) > 0:
                        first_frame = data[0]
                        if len(first_frame) > 0:
                            # Check if it's DWPose format (frames, people, keypoints, coords)
                            if len(first_frame.shape) == 2 and first_frame.shape[1] == self.num_keypoints:
                                is_valid = True
                            # Check if it's MediaPipe format (frames, keypoints, coords)
                            elif len(first_frame.shape) == 2 and first_frame.shape[0] == self.num_keypoints:
                                is_valid = True
                            else:
                                is_valid = False
                            
                            if is_valid:
                                self.samples.append({
                                    'video_id': video_id,
                                    'file_path': npy_file,
                                    'labels': labels[video_id]
                                })
                except Exception as e:
                    # Skip corrupted files
                    continue
        
        print(f"Loaded {len(self.samples)} valid samples total")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load pose data
        data = np.load(sample['file_path'], allow_pickle=True)
        
        # Extract features: supports both formats
        # DWPose format: (frames, people, keypoints, coords)
        # MediaPipe format: (frames, keypoints, coords)
        features = []
        for frame in data:
            if len(frame) > 0:
                # Check format: if first dimension has shape matching num_keypoints, it's MediaPipe
                if len(frame.shape) == 2 and frame.shape[0] == self.num_keypoints:
                    # MediaPipe format: (keypoints, coords)
                    frame_features = frame.flatten()
                else:
                    # DWPose format: (people, keypoints, coords) - use first person
                    frame_features = frame[0].flatten()
                features.append(frame_features)
            else:
                # Empty frame - use zeros
                features.append(np.zeros(self.num_keypoints * 3))
        
        features = np.array(features)
        
        # Pad or truncate to max_frames
        if len(features) < self.max_frames:
            padding = np.zeros((self.max_frames - len(features), self.num_keypoints * 3))
            features = np.vstack([features, padding])
        else:
            features = features[:self.max_frames]
        
        # Get labels (4 values: boredom, engagement, confusion, frustration)
        labels = np.array([
            sample['labels']['boredom'],
            sample['labels']['engagement'],
            sample['labels']['confusion'],
            sample['labels']['frustration']
        ], dtype=np.float32)
        
        # Normalize labels to [0, 1] for regression
        labels = labels / 3.0
        
        return torch.FloatTensor(features), torch.FloatTensor(labels)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformerModel(nn.Module):
    """Temporal Transformer for engagement prediction"""
    
    def __init__(self, input_size=402, d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.3, max_frames=300):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_frames, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 4)  # 4 engagement dimensions
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global average pooling over time dimension
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Output layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Output in [0, 1] range
        
        return x


class WeightedMSELoss(nn.Module):
    """MSE loss with class weights to handle imbalance"""
    
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights if weights is not None else torch.ones(4)
    
    def forward(self, pred, target):
        # pred, target: (batch, 4)
        mse = (pred - target) ** 2  # (batch, 4)
        weighted_mse = mse * self.weights.to(pred.device)
        return weighted_mse.mean()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for features, labels in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model with regression metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating"):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all predictions and labels
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate regression metrics for each dimension
    metrics = {}
    dim_names = ['boredom', 'engagement', 'confusion', 'frustration']
    
    for i, name in enumerate(dim_names):
        mae = mean_absolute_error(all_labels[:, i], all_preds[:, i])
        rmse = np.sqrt(mean_squared_error(all_labels[:, i], all_preds[:, i]))
        r2 = r2_score(all_labels[:, i], all_preds[:, i])
        
        metrics[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    # Overall metrics
    overall_mae = mean_absolute_error(all_labels.flatten(), all_preds.flatten())
    overall_rmse = np.sqrt(mean_squared_error(all_labels.flatten(), all_preds.flatten()))
    overall_r2 = r2_score(all_labels.flatten(), all_preds.flatten())
    
    return total_loss / len(dataloader), metrics, overall_mae, overall_rmse, overall_r2


def plot_kfold_results(fold_results, output_dir):
    """Plot k-fold cross-validation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('10-Fold Cross-Validation Results', fontsize=16, fontweight='bold')
    
    dim_names = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
    metrics_names = ['MAE', 'RMSE', 'R²']
    
    for idx, dim_name in enumerate(dim_names):
        ax = axes[idx // 2, idx % 2]
        
        # Collect metrics across folds
        maes = [fold['val_metrics'][dim_name.lower()]['mae'] for fold in fold_results]
        rmses = [fold['val_metrics'][dim_name.lower()]['rmse'] for fold in fold_results]
        r2s = [fold['val_metrics'][dim_name.lower()]['r2'] for fold in fold_results]
        
        x = np.arange(len(fold_results))
        width = 0.25
        
        ax.bar(x - width, maes, width, label='MAE', alpha=0.8)
        ax.bar(x, rmses, width, label='RMSE', alpha=0.8)
        ax.bar(x + width, r2s, width, label='R²', alpha=0.8)
        
        ax.set_xlabel('Fold', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'{dim_name} Metrics Across Folds', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'F{i+1}' for i in range(len(fold_results))])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean lines
        ax.axhline(y=np.mean(maes), color='C0', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=np.mean(rmses), color='C1', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=np.mean(r2s), color='C2', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kfold_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot overall metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    overall_maes = [fold['val_overall_mae'] for fold in fold_results]
    overall_rmses = [fold['val_overall_rmse'] for fold in fold_results]
    overall_r2s = [fold['val_overall_r2'] for fold in fold_results]
    
    x = np.arange(len(fold_results))
    width = 0.25
    
    ax.bar(x - width, overall_maes, width, label='MAE', alpha=0.8)
    ax.bar(x, overall_rmses, width, label='RMSE', alpha=0.8)
    ax.bar(x + width, overall_r2s, width, label='R²', alpha=0.8)
    
    ax.set_xlabel('Fold', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Overall Regression Metrics Across Folds', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(len(fold_results))])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean and std annotations
    ax.text(0.02, 0.98, 
            f"Mean MAE: {np.mean(overall_maes):.4f} ± {np.std(overall_maes):.4f}\n"
            f"Mean RMSE: {np.mean(overall_rmses):.4f} ± {np.std(overall_rmses):.4f}\n"
            f"Mean R²: {np.mean(overall_r2s):.4f} ± {np.std(overall_r2s):.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kfold_overall.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Transformer engagement model with k-fold CV')
    parser.add_argument('--features-dir', type=str, required=True, help='Path to features directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for trained model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs per fold')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=256, help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--max-frames', type=int, default=300, help='Maximum frames per video')
    parser.add_argument('--n-folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--num-keypoints', type=int, default=134, help='Number of keypoints per person (134 for DWPose, 543 for MediaPipe)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    labels_file = Path(args.features_dir) / 'labels.json'
    print(f"Loading labels from {labels_file}")
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    # Create full dataset
    full_dataset = EngagementDataset(args.features_dir, labels, max_frames=args.max_frames, num_keypoints=args.num_keypoints)
    
    if len(full_dataset) == 0:
        print("ERROR: No valid samples loaded!")
        return
    
    # Calculate class weights for weighted loss (handle imbalance)
    all_labels = []
    for i in range(len(full_dataset)):
        _, label = full_dataset[i]
        all_labels.append(label.numpy())
    all_labels = np.array(all_labels)
    
    # Compute inverse frequency weights
    label_means = all_labels.mean(axis=0)
    weights = 1.0 / (label_means + 0.01)  # Add epsilon to avoid division by zero
    weights = weights / weights.sum() * 4  # Normalize to sum to number of classes
    weights = torch.FloatTensor(weights)
    print(f"Class weights: {weights.numpy()}")
    
    # K-Fold Cross Validation
    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    print(f"\n{'='*80}")
    print(f"Starting {args.n_folds}-Fold Cross Validation")
    print(f"{'='*80}\n")
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold + 1}/{args.n_folds}")
        print(f"{'='*80}")
        print(f"Train samples: {len(train_ids)}, Validation samples: {len(val_ids)}")
        
        # Create data samplers and loaders
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4)
        
        # Initialize model
        model = TemporalTransformerModel(
            input_size=args.num_keypoints * 3,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_frames=args.max_frames
        ).to(device)
        
        # Loss and optimizer
        criterion = WeightedMSELoss(weights=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        best_engagement_mae = float('inf')  # Track best engagement MAE specifically
        best_epoch = 0
        patience = 10
        patience_counter = 0
        
        fold_history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_r2': []
        }
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss, val_metrics, val_mae, val_rmse, val_r2 = evaluate(model, val_loader, criterion, device)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save history
            fold_history['train_loss'].append(train_loss)
            fold_history['val_loss'].append(val_loss)
            fold_history['val_mae'].append(val_mae)
            fold_history['val_rmse'].append(val_rmse)
            fold_history['val_r2'].append(val_r2)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")
            print("Per-dimension metrics:")
            for dim_name in ['boredom', 'engagement', 'confusion', 'frustration']:
                m = val_metrics[dim_name]
                print(f"  {dim_name.capitalize():12s}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, R²={m['r2']:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                # Save best model for this fold
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2
                }, output_dir / f'fold_{fold+1}_best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
        
        # Load best model for this fold
        checkpoint = torch.load(output_dir / f'fold_{fold+1}_best_model.pth', weights_only=False)
        
        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'best_epoch': best_epoch,
            'val_loss': checkpoint['val_loss'],
            'val_metrics': checkpoint['val_metrics'],
            'val_mae': checkpoint['val_mae'],
            'val_rmse': checkpoint['val_rmse'],
            'val_r2': checkpoint['val_r2'],
            'history': fold_history
        })
        
        print(f"\nFold {fold + 1} completed!")
        print(f"Best epoch: {best_epoch + 1}")
        print(f"Best val loss: {checkpoint['val_loss']:.4f}")
        print(f"Best MAE: {checkpoint['val_mae']:.4f} | RMSE: {checkpoint['val_rmse']:.4f} | R²: {checkpoint['val_r2']:.4f}")
    
    # Summary of k-fold results
    print(f"\n{'='*80}")
    print("K-FOLD CROSS VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    
    val_maes = [fold['val_mae'] for fold in fold_results]
    val_rmses = [fold['val_rmse'] for fold in fold_results]
    val_r2s = [fold['val_r2'] for fold in fold_results]
    
    print(f"Overall MAE:  {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")
    print(f"Overall RMSE: {np.mean(val_rmses):.4f} ± {np.std(val_rmses):.4f}")
    print(f"Overall R²:   {np.mean(val_r2s):.4f} ± {np.std(val_r2s):.4f}")
    
    print("\nPer-dimension summary:")
    for dim_name in ['boredom', 'engagement', 'confusion', 'frustration']:
        maes = [fold['val_metrics'][dim_name]['mae'] for fold in fold_results]
        rmses = [fold['val_metrics'][dim_name]['rmse'] for fold in fold_results]
        r2s = [fold['val_metrics'][dim_name]['r2'] for fold in fold_results]
        
        print(f"\n{dim_name.capitalize()}:")
        print(f"  MAE:  {np.mean(maes):.4f} ± {np.std(maes):.4f}")
        print(f"  RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
        print(f"  R²:   {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
    
    # Save results
    results_file = output_dir / 'kfold_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'n_folds': args.n_folds,
            'fold_results': fold_results,
            'summary': {
                'mean_mae': float(np.mean(val_maes)),
                'std_mae': float(np.std(val_maes)),
                'mean_rmse': float(np.mean(val_rmses)),
                'std_rmse': float(np.std(val_rmses)),
                'mean_r2': float(np.mean(val_r2s)),
                'std_r2': float(np.std(val_r2s))
            }
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    
    # Plot results
    plot_kfold_results(fold_results, output_dir)
    print(f"Plots saved to {output_dir}")
    
    # Train final model on all data with best hyperparameters
    print(f"\n{'='*80}")
    print("Training final model on full dataset...")
    print(f"{'='*80}\n")
    
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    final_model = TemporalTransformerModel(
        input_size=args.num_keypoints * 3,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_frames=args.max_frames
    ).to(device)
    
    criterion = WeightedMSELoss(weights=weights)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Train for average best epoch across folds
    avg_best_epoch = int(np.mean([fold['best_epoch'] for fold in fold_results]))
    print(f"Training for {avg_best_epoch} epochs (average best epoch from k-fold)")
    
    for epoch in range(avg_best_epoch):
        train_loss = train_epoch(final_model, full_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{avg_best_epoch} - Train Loss: {train_loss:.4f}")
    
    # Save final model
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_config': {
            'input_size': args.num_keypoints * 3,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'max_frames': args.max_frames
        },
        'kfold_summary': {
            'mean_mae': float(np.mean(val_maes)),
            'mean_rmse': float(np.mean(val_rmses)),
            'mean_r2': float(np.mean(val_r2s))
        }
    }, output_dir / 'final_model.pth')
    
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"Final model saved to: {output_dir / 'final_model.pth'}")
    print(f"K-fold results saved to: {results_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
