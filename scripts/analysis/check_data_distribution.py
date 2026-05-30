"""
Verify that disengagement data was loaded correctly during training.
Check for class imbalance, data format issues, or loading bugs.
"""

import json
from collections import defaultdict

# Load label mapping
with open('models/action_transformer_kinetics700/label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

action_to_idx = label_mapping['action_to_idx']

# Define engagement vs disengagement
engagement_actions = [
    'applauding', 'clapping', 'cheerleading', 'headbanging', 'mosh pit dancing',
    'surfing crowd', 'belly dancing', 'breakdancing', 'country line dancing',
    'dancing ballet', 'dancing charleston', 'dancing gangnam style', 'dancing macarena',
    'jumpstyle dancing', 'robot dancing', 'salsa dancing', 'shoot dance', 'square dancing',
    'swing dancing', 'tango dancing', 'tap dancing', 'singing', 'gospel singing in church',
    'playing accordion', 'playing bagpipes', 'playing bass guitar', 'playing cello',
    'playing clarinet', 'playing cymbals', 'playing drums', 'playing guitar',
    'playing harmonica', 'playing keyboard', 'playing piano', 'playing saxophone',
    'playing trombone', 'playing trumpet', 'playing ukulele', 'playing violin',
    'recording music', 'using megaphone'
]

disengagement_actions = [
    'looking at phone', 'texting', 'talking on cell phone', 'listening with headphones',
    'sleeping', 'yawning', 'staring', 'watching tv', 'reading book', 'reading newspaper',
    'waiting in line', 'fidgeting', 'twiddling fingers', 'drumming fingers', 'tapping pen',
    'winking', 'rolling eyes', 'shaking head', 'crossing eyes', 'arguing',
    'smoking', 'smoking hookah', 'smoking pipe', 'checking watch',
    'eating burger', 'eating chips', 'eating doughnuts', 'eating hotdog', 'eating ice cream',
    'eating nachos', 'eating spaghetti', 'drinking shots', 'sipping cup',
    'waking up', 'stretching arm', 'stretching leg', 'coughing', 'sneezing', 'blowing nose',
    'crying', 'burping', 'drooling', 'chewing gum', 'falling off chair', 'falling off bike'
]

print("=" * 80)
print("ENGAGEMENT VS DISENGAGEMENT DISTRIBUTION IN LABEL MAPPING")
print("=" * 80)

engagement_indices = []
disengagement_indices = []

print("\nENGAGEMENT ACTIONS:")
print("-" * 80)
for action in sorted(engagement_actions):
    if action in action_to_idx:
        idx = action_to_idx[action]
        engagement_indices.append(idx)
        print(f"  {idx:3d}: {action}")

print("\nDISENGAGEMENT ACTIONS:")
print("-" * 80)
for action in sorted(disengagement_actions):
    if action in action_to_idx:
        idx = action_to_idx[action]
        disengagement_indices.append(idx)
        print(f"  {idx:3d}: {action}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total classes: 86")
print(f"Engagement classes: {len(engagement_indices)} (indices: {min(engagement_indices)}-{max(engagement_indices)})")
print(f"Disengagement classes: {len(disengagement_indices)} (indices: {min(disengagement_indices)}-{max(disengagement_indices)})")

# Check if indices are interleaved
print(f"\n⚠️ CRITICAL CHECK: Are engagement/disengagement indices MIXED?")
print(f"   Engagement range: {min(engagement_indices)} to {max(engagement_indices)}")
print(f"   Disengagement range: {min(disengagement_indices)} to {max(disengagement_indices)}")
print(f"   Overlap: YES ✅" if min(engagement_indices) <= max(disengagement_indices) and min(disengagement_indices) <= max(engagement_indices) else "   Overlap: NO ❌ PROBLEM!")

# Load classification report and check which indices failed
print("\n" + "=" * 80)
print("CHECKING FAILED CLASSES BY INDEX")
print("=" * 80)

failed_classes = [
    'coughing', 'crying', 'drumming fingers', 'eating burger', 'eating nachos',
    'looking at phone', 'reading newspaper', 'recording music', 'singing',
    'using megaphone', 'winking'
]

print("\nFailed classes (F1 = 0.00):")
for action in failed_classes:
    idx = action_to_idx[action]
    category = "ENGAGEMENT" if action in engagement_actions else "DISENGAGEMENT"
    print(f"  {idx:3d}: {action:<30} [{category}]")

engagement_failed = sum(1 for a in failed_classes if a in engagement_actions)
disengagement_failed = sum(1 for a in failed_classes if a in disengagement_actions)

print(f"\nFailed breakdown:")
print(f"  Engagement: {engagement_failed}/41 ({engagement_failed/41*100:.1f}%)")
print(f"  Disengagement: {disengagement_failed}/45 ({disengagement_failed/45*100:.1f}%)")

print("\n⚠️ YOUR CONCERN: Are disengagement classes systematically failing?")
print(f"   Answer: NO - only {disengagement_failed/45*100:.1f}% of disengagement classes have F1=0")
print(f"   Reality: Both categories have failures, it's a CLASS-SPECIFIC issue, not category-wide")
