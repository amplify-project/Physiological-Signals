#!/usr/bin/env python3
"""
Download disengagement action classes from Kinetics-700-2020
Maps action labels to alphabetically-ordered tar file numbers
"""

import requests
import subprocess
from pathlib import Path
from collections import defaultdict

# Base paths
BASE_DIR = Path.home() / "concert_engagement/data/raw/Kinetics700"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

# Disengagement actions to download (45 actions)
DISENGAGEMENT_ACTIONS = {
    # Phone/Tech distraction (4)
    "looking at phone", "texting", "talking on cell phone", "listening with headphones",
    
    # Passive/Bored (7)
    "sleeping", "yawning", "staring", "watching tv", "reading book", 
    "reading newspaper", "waiting in line",
    
    # Fidgeting/Restlessness (5)
    "fidgeting", "twiddling fingers", "drumming fingers", "tapping pen", "winking",
    
    # Negative body language (4)
    "rolling eyes", "shaking head", "crossing eyes", "arguing",
    
    # Smoking/Distracted (3)
    "smoking", "smoking hookah", "smoking pipe",
    
    # Checking time (1)
    "checking watch",
    
    # Eating/Drinking (9)
    "eating burger", "eating chips", "eating doughnuts", "eating hotdog", 
    "eating ice cream", "eating nachos", "eating spaghetti", "drinking shots", "sipping cup",
    
    # Tired/Uncomfortable (6)
    "waking up", "stretching arm", "stretching leg", "coughing", "sneezing", "blowing nose",
    
    # Negative reactions (6)
    "crying", "burping", "drooling", "chewing gum", "falling off chair", "falling off bike"
}

def download_annotations():
    """Download train and val CSV annotations"""
    print("📥 Downloading annotations...")

    train_csv = BASE_DIR / "train.csv"
    val_csv = BASE_DIR / "val.csv"

    if not train_csv.exists():
        print("  Downloading train.csv...")
        r = requests.get("https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv")
        train_csv.write_text(r.text)
        print(f"  ✓ Saved: {train_csv}")
    else:
        print(f"  ✓ Already exists: {train_csv}")

    if not val_csv.exists():
        print("  Downloading val.csv...")
        r = requests.get("https://s3.amazonaws.com/kinetics/700_2020/annotations/val.csv")
        val_csv.write_text(r.text)
        print(f"  ✓ Saved: {val_csv}")
    else:
        print(f"  ✓ Already exists: {val_csv}")

    return train_csv, val_csv

def map_actions_to_tars(csv_file, split):
    """Map action classes to their tar file numbers"""
    print(f"\n📊 Analyzing {split} split...")

    # Read CSV and group by action
    action_videos = defaultdict(list)

    with open(csv_file) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                action = parts[0]
                video_id = parts[1]
                action_videos[action].append(video_id)

    # Get all unique actions
    all_actions = sorted(action_videos.keys())
    print(f"  Total actions: {len(all_actions)}")

    # Map actions to tar numbers (alphabetical order, 1-indexed)
    action_to_tar = {}
    for idx, action in enumerate(all_actions, start=1):
        action_to_tar[action] = idx

    # Find disengagement actions
    relevant_tars = {}
    for action in DISENGAGEMENT_ACTIONS:
        if action in action_to_tar:
            tar_num = action_to_tar[action]
            video_count = len(action_videos[action])
            relevant_tars[action] = {
                'tar_num': tar_num,
                'video_count': video_count
            }
            print(f"  ✓ {action}: tar #{tar_num:03d} ({video_count} videos)")
        else:
            print(f"  ✗ {action}: NOT FOUND in dataset")

    return relevant_tars

def download_tar_files(relevant_tars, split):
    """Download relevant tar files from S3"""
    base_url = f"https://s3.amazonaws.com/kinetics/700_2020/{split}"
    output_dir = TRAIN_DIR if split == "train" else VAL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Downloading {split} tar files...")
    print(f"  Target: {output_dir}")
    print(f"  Actions to download: {len(relevant_tars)}")
    print()

    total_videos = 0
    success_count = 0
    failed = []

    for idx, (action, info) in enumerate(sorted(relevant_tars.items()), start=1):
        tar_num = info['tar_num']
        video_count = info['video_count']

        # Construct tar filename
        tar_file = f"k700_{split}_{tar_num:03d}.tar.gz"
        url = f"{base_url}/{tar_file}"
        output_file = output_dir / tar_file

        print(f"[{idx}/{len(relevant_tars)}] {action}")
        print(f"  URL: {url}")
        print(f"  Videos: {video_count}")

        # Check if already downloaded
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ SKIP: Already exists ({size_mb:.1f} MB)")
            success_count += 1
            total_videos += video_count
            continue

        # Download with wget
        try:
            print(f"  → Downloading...")
            result = subprocess.run(
                ["wget", "-q", "--show-progress", "-O", str(output_file), url],
                check=True,
                capture_output=True
            )

            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ Downloaded: {size_mb:.1f} MB")

            success_count += 1
            total_videos += video_count

        except subprocess.CalledProcessError as e:
            print(f"  ✗ FAILED: {e}")
            failed.append(action)
            if output_file.exists():
                output_file.unlink()

        print()

    print("=" * 60)
    print(f"✅ {split.upper()} DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Success: {success_count} / {len(relevant_tars)}")
    print(f"Failed:  {len(failed)}")
    print(f"Videos:  ~{total_videos}")

    if failed:
        print(f"\nFailed actions:")
        for action in failed:
            print(f"  - {action}")

    return success_count, total_videos

def main():
    # Download annotations
    train_csv, val_csv = download_annotations()

    # Map actions to tar numbers for both splits
    train_tars = map_actions_to_tars(train_csv, "train")
    val_tars = map_actions_to_tars(val_csv, "val")

    # Download train split
    train_success, train_videos = download_tar_files(train_tars, "train")

    # Download val split
    val_success, val_videos = download_tar_files(val_tars, "val")

    # Final summary
    print(f"\n{'=' * 60}")
    print("🎉 DOWNLOAD COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nTRAIN SPLIT:")
    print(f"  Actions: {train_success}/{len(train_tars)}")
    print(f"  Videos:  ~{train_videos}")
    
    print(f"\nVAL SPLIT:")
    print(f"  Actions: {val_success}/{len(val_tars)}")
    print(f"  Videos:  ~{val_videos}")
    
    print(f"\nTOTAL: ~{train_videos + val_videos} videos")
    print(f"\n📦 Next: Extract tar files")

if __name__ == "__main__":
    main()
