#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import sys

# Test loading first few files to see if any hang
files = list(Path('data/processed/features_kinetics700/train').rglob('*.npz'))[:10]
print(f"Testing first 10 training files...")
for f in files:
    try:
        with np.load(f) as data:
            shape = data['features'].shape
        print(f"✓ {f.name}: {shape}")
    except Exception as e:
        print(f"✗ {f.name}: {type(e).__name__} - {e}")
        sys.exit(1)
print("\n✅ All test files loaded successfully!")
