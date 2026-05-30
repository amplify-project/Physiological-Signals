#!/usr/bin/env python3
"""
Scan all .npz feature files and attempt to load them with a per-file timeout.
Writes results to logs/scan_npz_results.txt with format: OK | SLOW | ERROR | PATH | seconds
"""
import os
import sys
import time
import signal
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / 'data' / 'processed' / 'features_kinetics700'
OUT = ROOT / 'logs' / 'scan_npz_results.txt'
TIMEOUT = 5  # seconds per file

if not FEATURES.exists():
    print(f"Features dir not found: {FEATURES}")
    sys.exit(1)

OUT.parent.mkdir(parents=True, exist_ok=True)

bad_count = 0
total = 0

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, handler)

with OUT.open('w') as fout:
    for split in ('train','val'):
        split_dir = FEATURES / split
        if not split_dir.exists():
            continue
        for action_dir in sorted(split_dir.iterdir()):
            if not action_dir.is_dir():
                continue
            for npz in sorted(action_dir.glob('*.npz')):
                total += 1
                path = str(npz)
                try:
                    start = time.time()
                    signal.alarm(TIMEOUT)
                    # use mmap_mode read-only to avoid full load; still triggers errors if corrupted
                    data = np.load(path, mmap_mode='r')
                    # touch the array to ensure it's readable (copy small slice)
                    _ = data['features'][0:1]
                    signal.alarm(0)
                    elapsed = time.time() - start
                    tag = 'OK' if elapsed < 1.0 else 'SLOW'
                    fout.write(f"{tag} | {elapsed:.3f} | {path}\n")
                except TimeoutException:
                    fout.write(f"TIMEOUT | >{TIMEOUT} | {path}\n")
                    bad_count += 1
                except Exception as e:
                    fout.write(f"ERROR | {repr(e)} | {path}\n")
                    bad_count += 1
                if total % 1000 == 0:
                    print(f"Scanned {total} files, bad so far: {bad_count}")

print(f"Scan complete. Total: {total}, bad: {bad_count}. Results in {OUT}")
