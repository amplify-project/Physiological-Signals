#!/usr/bin/env python3
"""
Debug script to test a single .npz file through the entire training pipeline.
This helps identify exactly where corrupted files cause issues.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass
import sys
import zipfile


# Simple dataset for testing single file
class SingleFileDataset(Dataset):
    def __init__(self, file_path, sequence_length=300):
        self.file_path = Path(file_path)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        # Load features with same logic as training
        with np.load(self.file_path, allow_pickle=False) as data:
            features = data['features'].copy()  # (num_frames, 543, 3)
        
        # Flatten to (num_frames, 1629)
        num_frames = features.shape[0]
        features = features.reshape(num_frames, -1)
        
        # Pad or truncate to sequence_length
        if num_frames < self.sequence_length:
            padding = np.zeros((self.sequence_length - num_frames, features.shape[1]))
            features = np.vstack([features, padding])
        else:
            features = features[:self.sequence_length]
        
        return torch.FloatTensor(features), 0  # dummy label


def test_file_info(file_path: Path) -> bool:
    """Test basic file information."""
    print("\n" + "="*60)
    print("STAGE 1: File Information")
    print("="*60)
    
    if not file_path.exists():
        print(f"❌ File does not exist: {file_path}")
        return False
    
    file_size = file_path.stat().st_size
    print(f"✅ File exists: {file_path}")
    print(f"   Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    # Try to check if it's a valid zip file
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            print(f"   ZIP contents: {z.namelist()}")
            for name in z.namelist():
                info = z.getinfo(name)
                print(f"     - {name}: {info.file_size:,} bytes (compressed: {info.compress_size:,})")
    except zipfile.BadZipFile as e:
        print(f"⚠️  Bad ZIP file: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Error reading ZIP: {e}")
        return False
    
    print("✅ File info check passed")
    return True


def test_numpy_load(file_path: Path) -> bool:
    """Test if file can be loaded with np.load()."""
    print("\n" + "="*60)
    print("STAGE 2: NumPy Load Test (watch for hangs...)")
    print("="*60)
    
    try:
        print("   Opening file with np.load()...")
        with np.load(file_path, allow_pickle=False) as data:
            print(f"   Keys: {list(data.keys())}")
            features = data['features']
            print(f"   Features shape: {features.shape}")
            print(f"   Features dtype: {features.dtype}")
            
            # Try to actually access the data (triggers decompression)
            print("   Accessing first frame (this will hang if file is corrupted)...")
            first_frame = features[0].copy()
            print(f"   First frame shape: {first_frame.shape}")
            print(f"   First frame range: [{first_frame.min():.3f}, {first_frame.max():.3f}]")
        
        print("✅ NumPy load successful")
        return True
    
    except zipfile.BadZipFile as e:
        print(f"❌ BadZipFile error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading with numpy: {type(e).__name__}: {e}")
        return False


def test_dataset_loading(file_path: Path) -> bool:
    """Test if file can be loaded through dataset wrapper."""
    print("\n" + "="*60)
    print("STAGE 3: Dataset Loading Test")
    print("="*60)
    
    try:
        dataset = SingleFileDataset(file_path)
        features, label = dataset[0]
        
        print(f"   Features tensor shape: {features.shape}")
        print(f"   Features dtype: {features.dtype}")
        print(f"   Label: {label}")
        
        print("✅ Dataset loading successful")
        return True
    
    except Exception as e:
        print(f"❌ Error in dataset loading: {type(e).__name__}: {e}")
        return False


def test_dataloader_iteration(file_path: Path) -> bool:
    """Test if file can be iterated through DataLoader."""
    print("\n" + "="*60)
    print("STAGE 4: DataLoader Iteration Test")
    print("="*60)
    
    try:
        dataset = SingleFileDataset(file_path)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Single-threaded for debugging
            pin_memory=False
        )
        
        print(f"   DataLoader created with {len(dataloader)} batches")
        
        for batch_idx, (features, labels) in enumerate(dataloader):
            print(f"   Batch {batch_idx}:")
            print(f"     Features shape: {features.shape}")
            print(f"     Labels shape: {labels.shape}")
            break  # Only need to test first batch
        
        print("✅ DataLoader iteration successful")
        return True
    
    except Exception as e:
        print(f"❌ Error in DataLoader iteration: {type(e).__name__}: {e}")
        return False


def test_batch_processing(file_path: Path, device: str = 'cuda:0') -> bool:
    """Test if batch can be moved to GPU and processed."""
    print("\n" + "="*60)
    print("STAGE 5: Batch Processing Test")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping GPU test")
        return True
    
    try:
        dataset = SingleFileDataset(file_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        for features, labels in dataloader:
            print(f"   CPU tensor shape: {features.shape}")
            
            # Move to GPU
            features = features.to(device)
            labels = labels.to(device)
            
            print(f"   GPU tensor device: {features.device}")
            print(f"   GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
            
            # Test basic operations
            mean = features.mean()
            std = features.std()
            print(f"   Tensor mean: {mean.item():.4f}")
            print(f"   Tensor std: {std.item():.4f}")
            
            break
        
        print("✅ Batch processing successful")
        return True
    
    except Exception as e:
        print(f"❌ Error in batch processing: {type(e).__name__}: {e}")
        return False


def main(file_path: str):
    """Run all debug tests on a single file."""
    file_path = Path(file_path)
    
    print("="*60)
    print("🔍 DEBUG SINGLE FILE PIPELINE")
    print("="*60)
    print(f"File: {file_path}")
    print(f"Absolute path: {file_path.absolute()}")
    
    # Run tests in order
    results = {}
    
    results['file_info'] = test_file_info(file_path)
    if not results['file_info']:
        print("\n❌ File info check failed - stopping here")
        return
    
    results['numpy_load'] = test_numpy_load(file_path)
    if not results['numpy_load']:
        print("\n❌ NumPy load failed - stopping here")
        return
    
    results['dataset_loading'] = test_dataset_loading(file_path)
    if not results['dataset_loading']:
        print("\n❌ Dataset loading failed - stopping here")
        return
    
    results['dataloader_iteration'] = test_dataloader_iteration(file_path)
    if not results['dataloader_iteration']:
        print("\n❌ DataLoader iteration failed - stopping here")
        return
    
    results['batch_processing'] = test_batch_processing(file_path)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    for stage, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{stage:25} {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! File is valid.")
    else:
        print("\n⚠️  Some tests failed - see details above")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_single_file.py <path_to_npz_file>")
        sys.exit(1)
    
    main(sys.argv[1])
