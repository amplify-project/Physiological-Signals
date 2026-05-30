#!/usr/bin/env python3
"""
Count samples for cleaned binary classification (v2)
Shows how many samples remain after quarantining noisy classes
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def load_labels(labels_file):
    with open(labels_file, 'r') as f:
        return json.load(f)

def count_samples(features_dir, labels_data):
    """Count samples per class and binary category"""
    
    # Get quarantined classes
    quarantined = set(labels_data['quarantined_classes']['fine_classes'])
    
    # Build mapping from fine class to binary
    fine_to_binary = {}
    for macro_name, macro_data in labels_data['hierarchy'].items():
        binary_label = macro_data['binary']
        for fine_class in macro_data['fine_classes']:
            fine_to_binary[fine_class] = binary_label
    
    # Count samples
    counts = {
        'engagement': defaultdict(int),
        'disengagement': defaultdict(int),
        'quarantined': defaultdict(int)
    }
    
    total_active = 0
    total_quarantined = 0
    
    for split in ['train', 'val']:
        split_dir = Path(features_dir) / split
        if not split_dir.exists():
            continue
            
        for action_dir in split_dir.iterdir():
            if not action_dir.is_dir():
                continue
                
            action_name = action_dir.name
            num_files = len(list(action_dir.glob('*.npz')))
            
            if action_name in quarantined:
                counts['quarantined'][action_name] += num_files
                total_quarantined += num_files
            elif action_name in fine_to_binary:
                binary_label = fine_to_binary[action_name]
                counts[binary_label][action_name] += num_files
                total_active += num_files
            else:
                print(f"WARNING: Unknown action '{action_name}' with {num_files} samples")
    
    return counts, total_active, total_quarantined

def main():
    # Paths
    features_dir = Path('data/processed/features_kinetics700')
    labels_file = Path('models/action_transformer_kinetics700/hierarchical_labels_v2_cleaned.json')
    
    print("="*80)
    print("CLEANED BINARY DATASET ANALYSIS (v2)")
    print("="*80)
    print()
    
    # Load labels
    labels_data = load_labels(labels_file)
    
    # Count samples
    counts, total_active, total_quarantined = count_samples(features_dir, labels_data)
    
    # Print engagement classes
    print("✅ ENGAGEMENT CLASSES:")
    print("-" * 80)
    engagement_total = 0
    for action, count in sorted(counts['engagement'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {action:40s} {count:5d} samples")
        engagement_total += count
    print(f"  {'TOTAL ENGAGEMENT':40s} {engagement_total:5d} samples")
    print()
    
    # Print disengagement classes
    print("❌ DISENGAGEMENT CLASSES:")
    print("-" * 80)
    disengagement_total = 0
    for action, count in sorted(counts['disengagement'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {action:40s} {count:5d} samples")
        disengagement_total += count
    print(f"  {'TOTAL DISENGAGEMENT':40s} {disengagement_total:5d} samples")
    print()
    
    # Print quarantined classes
    print("🔒 QUARANTINED CLASSES (excluded from training):")
    print("-" * 80)
    for action, count in sorted(counts['quarantined'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {action:40s} {count:5d} samples")
    print(f"  {'TOTAL QUARANTINED':40s} {total_quarantined:5d} samples")
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"  Active classes:       {len(counts['engagement']) + len(counts['disengagement'])}")
    print(f"    - Engagement:       {len(counts['engagement'])} classes, {engagement_total:,} samples ({engagement_total/(engagement_total+disengagement_total)*100:.1f}%)")
    print(f"    - Disengagement:    {len(counts['disengagement'])} classes, {disengagement_total:,} samples ({disengagement_total/(engagement_total+disengagement_total)*100:.1f}%)")
    print(f"  Quarantined classes:  {len(counts['quarantined'])} classes, {total_quarantined:,} samples")
    print(f"  Total dataset:        {total_active + total_quarantined:,} samples")
    print(f"  Samples used:         {total_active:,} ({total_active/(total_active+total_quarantined)*100:.1f}%)")
    print(f"  Samples excluded:     {total_quarantined:,} ({total_quarantined/(total_active+total_quarantined)*100:.1f}%)")
    print()
    print(f"  Class balance:        {engagement_total/disengagement_total:.2f}:1 (engagement:disengagement)")
    print("="*80)

if __name__ == '__main__':
    main()
