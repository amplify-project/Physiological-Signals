#!/usr/bin/env python3
"""
Custom Dataset and Sampler using pre-computed indices to avoid DDP hang on large datasets.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path


class PrecomputedActionDataset(Dataset):
    """
    Dataset that loads from pre-computed samples and indices files.
    This avoids expensive file scanning during DDP initialization.
    """
    
    def __init__(self, features_dir, split='train', indices_dir='data/processed', 
                 sequence_length=300, verbose=False, rank=0, label_level='fine', 
                 use_cleaned_labels=False):
        """
        Args:
            features_dir: Path to features_kinetics700 directory
            split: 'train' or 'val'
            indices_dir: Directory containing pre-computed indices files
            sequence_length: Number of frames per sequence
            verbose: Print progress
            rank: DDP rank for logging
            label_level: 'fine' (86 classes), 'macro' (13 classes), or 'binary' (2 classes)
            use_cleaned_labels: Use hierarchical_labels_v2_cleaned.json (excludes ambiguous classes)
        """
        self.features_dir = Path(features_dir)
        self.sequence_length = sequence_length
        self.split = split
        self.rank = rank  # Store rank for debug logging
        self.label_level = label_level
        
        # Load pre-computed samples list
        samples_file = Path(indices_dir) / f'{split}_samples.npy'
        if not samples_file.exists():
            raise FileNotFoundError(
                f"Samples file not found: {samples_file}\n"
                f"Run: python scripts/generate_dataset_indices.py --features-dir {features_dir}"
            )
        
        if verbose and rank == 0:
            print(f"📂 Loading pre-computed {split} samples from {samples_file}", flush=True)
        
        import time
        load_start = time.time()
        
        # Load samples: array of (relative_path, label) tuples
        self.samples = np.load(samples_file, allow_pickle=True)
        
        load_elapsed = time.time() - load_start
        
        if verbose and rank == 0:
            print(f"✅ Loaded {len(self.samples):,} samples in {load_elapsed:.3f}s!", flush=True)
            print(f"🔷 Sample format check: {type(self.samples[0])}, length: {len(self.samples[0])}", flush=True)
        
        # Load label mapping
        import json
        mapping_file = Path(indices_dir) / 'label_mapping.json'
        
        if verbose and rank == 0:
            print(f"🔷 Loading label mapping from {mapping_file}", flush=True)
        
        with open(mapping_file, 'r') as f:
            label_data = json.load(f)
            self.action_to_idx = label_data['action_to_idx']
            # Convert string keys back to int for idx_to_action
            self.idx_to_action = {int(k): v for k, v in label_data['idx_to_action'].items()}
        
        # Load hierarchical label conversions if using macro/binary
        self.quarantined_classes = set()
        if label_level != 'fine':
            # Choose labels file
            if use_cleaned_labels:
                hierarchical_file = Path('models/action_transformer_kinetics700/hierarchical_labels_v2_cleaned.json')
            else:
                hierarchical_file = Path('models/action_transformer_kinetics700/hierarchical_labels.json')
                
            if not hierarchical_file.exists():
                raise FileNotFoundError(
                    f"Hierarchical labels file not found: {hierarchical_file}\n"
                    f"Run: python scripts/create_hierarchical_labels.py"
                )
            
            if verbose and rank == 0:
                print(f"📋 Loading hierarchical labels from: {hierarchical_file.name}", flush=True)
            
            with open(hierarchical_file, 'r') as f:
                hierarchical = json.load(f)
            
            # Get quarantined classes if using cleaned version
            if use_cleaned_labels and 'quarantined_classes' in hierarchical:
                self.quarantined_classes = set(hierarchical['quarantined_classes']['fine_classes'])
                if verbose and rank == 0:
                    print(f"🔒 Quarantining {len(self.quarantined_classes)} ambiguous classes", flush=True)
                
                # Filter out samples from quarantined classes
                original_count = len(self.samples)
                filtered_samples = []
                for rel_path, label_idx in self.samples:
                    fine_class = self.idx_to_action[label_idx]
                    if fine_class not in self.quarantined_classes:
                        filtered_samples.append((rel_path, label_idx))
                
                self.samples = np.array(filtered_samples, dtype=object)
                filtered_count = len(self.samples)
                
                if verbose and rank == 0:
                    removed = original_count - filtered_count
                    print(f"🗑️  Filtered samples: {original_count:,} → {filtered_count:,} ({removed:,} removed, {removed/original_count*100:.1f}%)", flush=True)
            
            if label_level == 'macro':
                # Map fine class names → macro indices
                self.label_conversion = {}
                
                if 'conversions' in hierarchical:
                    # Original format with conversions section
                    for fine_class, fine_idx in self.action_to_idx.items():
                        macro_class = hierarchical['conversions']['fine_to_macro'].get(fine_class)
                        if macro_class:
                            macro_idx = hierarchical['macro']['action_to_idx'][macro_class]
                            self.label_conversion[fine_idx] = macro_idx
                else:
                    # Cleaned format with hierarchy
                    for macro_class, macro_data in hierarchical['hierarchy'].items():
                        macro_idx = hierarchical['macro']['action_to_idx'][macro_class]
                        for fine_class in macro_data['fine_classes']:
                            if fine_class in self.action_to_idx:
                                fine_idx = self.action_to_idx[fine_class]
                                self.label_conversion[fine_idx] = macro_idx
                
                self.num_classes = hierarchical['macro']['num_classes']
                if verbose and rank == 0:
                    print(f"✅ Using MACRO labels: {self.num_classes} classes", flush=True)
            
            elif label_level == 'binary':
                # Map fine class names → binary indices
                self.label_conversion = {}
                
                if 'conversions' in hierarchical:
                    # Original format with conversions section
                    for fine_class, fine_idx in self.action_to_idx.items():
                        binary_class = hierarchical['conversions']['fine_to_binary'].get(fine_class)
                        if binary_class:
                            binary_idx = hierarchical['binary']['action_to_idx'][binary_class]
                            self.label_conversion[fine_idx] = binary_idx
                else:
                    # Cleaned format with hierarchy
                    for macro_class, macro_data in hierarchical['hierarchy'].items():
                        binary_class = macro_data['binary']
                        binary_idx = hierarchical['binary']['action_to_idx'][binary_class]
                        for fine_class in macro_data['fine_classes']:
                            if fine_class in self.action_to_idx:
                                fine_idx = self.action_to_idx[fine_class]
                                self.label_conversion[fine_idx] = binary_idx
                
                self.num_classes = hierarchical['binary']['num_classes']
                if verbose and rank == 0:
                    print(f"✅ Using BINARY labels: {self.num_classes} classes (engagement/disengagement)", flush=True)
        else:
            self.label_conversion = None
            self.num_classes = len(self.action_to_idx)
            if verbose and rank == 0:
                print(f"✅ Using FINE labels: {self.num_classes} classes", flush=True)
        
        if verbose and rank == 0:
            print(f"✅ Label mapping loaded: {len(self.action_to_idx)} fine classes → {self.num_classes} {label_level} classes", flush=True)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx == 0:
            print(f"[Rank {self.rank}] 🔷 __getitem__ called for idx=0 (FIRST SAMPLE)", flush=True)
        
        rel_path, label = self.samples[idx]
        npz_path = self.features_dir / rel_path
        
        if idx == 0:
            print(f"[Rank {self.rank}] 🔷 Loading first file: {npz_path}", flush=True)
        
        # Load features
        with np.load(npz_path, allow_pickle=False) as data:
            features = np.array(data['features'])
        
        if idx == 0:
            print(f"[Rank {self.rank}] ✅ First file loaded! features.shape={features.shape}", flush=True)
        
        # Flatten keypoints: (num_frames, 543*3)
        features = features.reshape(features.shape[0], -1)
        
        # Pad or truncate to sequence_length
        num_frames = features.shape[0]
        if num_frames < self.sequence_length:
            padding = np.zeros((self.sequence_length - num_frames, features.shape[1]))
            features = np.vstack([features, padding])
        elif num_frames > self.sequence_length:
            indices = np.linspace(0, num_frames - 1, self.sequence_length, dtype=int)
            features = features[indices]
        
        # Convert label if using macro/binary classification
        if self.label_conversion is not None:
            label = self.label_conversion.get(label, label)  # Fallback to original if not in mapping
        
        return torch.FloatTensor(features), label


class PrecomputedDistributedSampler(Sampler):
    """
    DistributedSampler that uses pre-shuffled indices.
    This eliminates the expensive __iter__() initialization that causes DDP hangs.
    """
    
    def __init__(self, dataset_size, split='train', indices_dir='data/processed',
                 num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        """
        Args:
            dataset_size: Total number of samples in dataset
            split: 'train' or 'val'
            indices_dir: Directory containing pre-computed indices files
            num_replicas: Number of DDP processes
            rank: Current DDP rank
            shuffle: Whether to shuffle (if False, uses indices as-is)
            seed: Random seed for per-epoch shuffling
            drop_last: Whether to drop last incomplete batch
        """
        if num_replicas is None:
            import torch.distributed as dist
            if not dist.is_available():
                raise RuntimeError("Requires distributed package")
            num_replicas = dist.get_world_size()
        if rank is None:
            import torch.distributed as dist
            if not dist.is_available():
                raise RuntimeError("Requires distributed package")
            rank = dist.get_rank()
        
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        # Load pre-computed shuffled indices
        indices_file = Path(indices_dir) / f'{split}_indices.npy'
        if not indices_file.exists():
            raise FileNotFoundError(
                f"Indices file not found: {indices_file}\n"
                f"Run: python scripts/generate_dataset_indices.py --features-dir <path>"
            )
        
        print(f"[Rank {rank}] 🔷 PrecomputedDistributedSampler: Loading indices from {indices_file}", flush=True)
        self.base_indices = np.load(indices_file)
        print(f"[Rank {rank}] ✅ Loaded {len(self.base_indices):,} pre-shuffled indices", flush=True)
        
        if len(self.base_indices) != dataset_size:
            raise ValueError(
                f"Indices file has {len(self.base_indices)} entries "
                f"but dataset has {dataset_size} samples"
            )
        
        # Calculate subset size for this rank
        if self.drop_last and len(self.base_indices) % self.num_replicas != 0:
            self.num_samples = len(self.base_indices) // self.num_replicas
        else:
            self.num_samples = (len(self.base_indices) + self.num_replicas - 1) // self.num_replicas
        
        self.total_size = self.num_samples * self.num_replicas
        
        print(f"[Rank {rank}] 🔷 PrecomputedDistributedSampler initialized:", flush=True)
        print(f"[Rank {rank}]    - Dataset size: {dataset_size:,}", flush=True)
        print(f"[Rank {rank}]    - Num replicas: {num_replicas}", flush=True)
        print(f"[Rank {rank}]    - Rank: {rank}", flush=True)
        print(f"[Rank {rank}]    - Samples for this rank: {self.num_samples:,}", flush=True)
        print(f"[Rank {rank}]    - Total size (padded): {self.total_size:,}", flush=True)
    
    def __iter__(self):
        """
        Generate indices for this rank.
        Uses pre-computed shuffled indices, avoiding expensive on-the-fly generation.
        """
        # Start with pre-shuffled base indices
        indices = self.base_indices.copy()
        
        # Optionally re-shuffle per epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(indices), generator=g).numpy()
            indices = indices[perm]
        
        # Pad if needed (to make evenly divisible)
        if not self.drop_last and len(indices) < self.total_size:
            padding_size = self.total_size - len(indices)
            indices = np.concatenate([indices, indices[:padding_size]])
        else:
            # Truncate if drop_last
            indices = indices[:self.total_size]
        
        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        """Set epoch for per-epoch shuffling."""
        self.epoch = epoch
