#!/usr/bin/env python3
"""
Auto-delete corrupted .npz files without prompting.
"""
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

def validate_npz(npz_path):
    """Try to load and validate an npz file. Returns (is_valid, error_message)."""
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if 'features' not in data:
                return False, "Missing 'features' key"
            features = data['features']
            # Allow 0 frames - that's valid!
            if len(features.shape) != 3:
                return False, f"Wrong dimensions: {features.shape}"
            if features.shape[1] != 543 or features.shape[2] != 3:
                return False, f"Wrong shape: {features.shape}, expected (*, 543, 3)"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"

def main():
    features_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/processed/features_kinetics700')
    
    # Create quarantine directory
    quarantine_dir = Path('data/processed/corrupted_features_quarantine')
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {features_dir} for corrupted .npz files...")
    print(f"📦 QUARANTINE MODE: Corrupted files will be moved to {quarantine_dir}\n")
    
    corrupted = []
    moved_count = 0
    already_deleted = []
    
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
                # Move to quarantine preserving directory structure
                try:
                    relative_path = npz_file.relative_to(features_dir)
                    quarantine_path = quarantine_dir / relative_path
                    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
                    npz_file.rename(quarantine_path)
                    moved_count += 1
                except FileNotFoundError:
                    # File was already deleted in a previous run
                    already_deleted.append(npz_file)
                except Exception as e:
                    print(f"\n⚠️  Failed to move {npz_file}: {e}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total corrupted files found: {len(corrupted)}")
    print(f"Successfully moved to quarantine: {moved_count}")
    print(f"Already deleted (from previous run): {len(already_deleted)}")
    print(f"Quarantine location: {quarantine_dir.absolute()}")
    
    if already_deleted:
        print(f"\n📝 Files already deleted (first 10):")
        for f in already_deleted[:10]:
            print(f"  {f.name}")
        if len(already_deleted) > 10:
            print(f"  ... and {len(already_deleted) - 10} more")
    
    if corrupted:
        # Group by error type
        error_counts = {}
        for _, error_msg in corrupted:
            error_type = error_msg.split(':')[0] if error_msg else 'Unknown'
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        print(f"\nCorrupted files by error type:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
        
        print(f"\n✅ All corrupted files have been moved to quarantine!")
        print(f"   You can safely delete {quarantine_dir} later if training succeeds.")
    else:
        print("✅ No corrupted files found!")

if __name__ == '__main__':
    main()
