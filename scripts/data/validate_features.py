#!/usr/bin/env python3
"""
Quickly validate all .npz feature files and remove corrupted ones.
"""
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

def validate_npz(npz_path):
    """Try to load and validate an npz file. Returns (is_valid, error_message)."""
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            # Just check if 'features' key exists
            if 'features' not in data:
                return False, "Missing 'features' key"
            features = data['features']
            # Check shape is reasonable (num_frames, 543, 3)
            # Allow 0 frames (empty videos) - that's valid!
            if len(features.shape) != 3:
                return False, f"Wrong dimensions: {features.shape}"
            if features.shape[1] != 543 or features.shape[2] != 3:
                return False, f"Wrong shape: {features.shape}, expected (*, 543, 3)"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"

def main():
    features_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/processed/features_kinetics700')
    
    print(f"Scanning {features_dir} for corrupted .npz files...")
    
    corrupted = []
    
    for split in ['train', 'val']:
        split_dir = features_dir / split
        if not split_dir.exists():
            print(f"Skipping {split} (not found)")
            continue
        
        npz_files = list(split_dir.rglob('*.npz'))
        print(f"\n{split}: Validating {len(npz_files)} files...")
        
        for npz_file in tqdm(npz_files, desc=f"Validating {split}"):
            is_valid, error_msg = validate_npz(npz_file)
            if not is_valid:
                corrupted.append((npz_file, error_msg))
                print(f"\n⚠️  CORRUPTED: {npz_file}")
                print(f"    Error: {error_msg}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total corrupted files: {len(corrupted)}")
    
    if corrupted:
        # Group by error type
        error_counts = {}
        for _, error_msg in corrupted:
            error_type = error_msg.split(':')[0] if error_msg else 'Unknown'
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        print(f"\nCorrupted files by error type:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
        
        print(f"\nFirst 20 corrupted files:")
        for f, error_msg in corrupted[:20]:
            print(f"  {f}")
            print(f"    → {error_msg}")
        
        if len(corrupted) > 20:
            print(f"  ... and {len(corrupted) - 20} more")
        
        response = input(f"\nDelete {len(corrupted)} corrupted files? (yes/no): ")
        if response.lower() == 'yes':
            for f, _ in corrupted:
                f.unlink()
                print(f"Deleted: {f}")
            print(f"\n✅ Deleted {len(corrupted)} corrupted files")
        else:
            print("Not deleting files.")
    else:
        print("✅ No corrupted files found!")

if __name__ == '__main__':
    main()
