#!/usr/bin/env python3
"""
Generate pre-shuffled dataset indices for DDP training.
This eliminates the race condition during DistributedSampler initialization.
"""

import numpy as np
from pathlib import Path
import argparse
import time


def generate_indices(features_dir, output_dir='data/processed'):
    """
    Scan dataset and generate pre-shuffled index files for train and val splits.
    
    Args:
        features_dir: Path to features_kinetics700 directory
        output_dir: Where to save the index files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val']:
        print(f"\n{'='*80}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*80}")
        
        split_dir = Path(features_dir) / split
        
        # Collect all samples
        samples = []
        action_to_idx = {}
        idx_to_action = {}
        
        action_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        print(f"Found {len(action_dirs)} action classes")
        
        start_time = time.time()
        
        for idx, action_dir in enumerate(action_dirs):
            action_to_idx[action_dir.name] = idx
            idx_to_action[idx] = action_dir.name
            
            npz_files = list(action_dir.glob('*.npz'))
            for npz_file in npz_files:
                # Store relative path from features_dir for portability
                rel_path = npz_file.relative_to(Path(features_dir))
                samples.append((str(rel_path), idx))
            
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {idx + 1}/{len(action_dirs)} classes | "
                      f"{len(samples):,} samples | {elapsed:.1f}s elapsed")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Collected {len(samples):,} samples in {elapsed:.1f}s")
        
        # Generate shuffled indices
        print(f"🔀 Generating shuffled indices...")
        indices = np.arange(len(samples))
        np.random.shuffle(indices)
        
        # Save indices
        indices_file = output_dir / f'{split}_indices.npy'
        np.save(indices_file, indices)
        print(f"💾 Saved shuffled indices to {indices_file}")
        
        # Save samples list (paths and labels)
        samples_file = output_dir / f'{split}_samples.npy'
        np.save(samples_file, np.array(samples, dtype=object))
        print(f"💾 Saved samples list to {samples_file}")
        
        # Save label mapping (only once, same for train/val)
        if split == 'train':
            import json
            label_mapping = {
                'action_to_idx': action_to_idx,
                'idx_to_action': idx_to_action
            }
            mapping_file = output_dir / 'label_mapping.json'
            with open(mapping_file, 'w') as f:
                json.dump(label_mapping, f, indent=2)
            print(f"💾 Saved label mapping to {mapping_file}")
    
    print(f"\n{'='*80}")
    print(f"✅ Index generation complete!")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  - {output_dir}/train_indices.npy")
    print(f"  - {output_dir}/train_samples.npy")
    print(f"  - {output_dir}/val_indices.npy")
    print(f"  - {output_dir}/val_samples.npy")
    print(f"  - {output_dir}/label_mapping.json")
    print(f"\nUsage in DDP training:")
    print(f"  Pass --use-precomputed-indices to train_action_transformer_ddp_v2.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate pre-shuffled dataset indices')
    parser.add_argument('--features-dir', type=str, required=True,
                       help='Path to features_kinetics700 directory')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Where to save index files (default: data/processed)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for shuffling (default: 42)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    
    print(f"\n{'='*80}")
    print(f"DATASET INDEX GENERATOR")
    print(f"{'='*80}")
    print(f"Features directory: {args.features_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    
    generate_indices(args.features_dir, args.output_dir)
