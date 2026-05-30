"""
Analyze Action Transformer Model Performance
Identifies failing classes, correlates with deleted files, and suggests improvements.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# Load classification report
report_path = Path("models/action_transformer_kinetics700/classification_report.txt")
with open(report_path, 'r') as f:
    report_text = f.read()

# Load label mapping
with open("models/action_transformer_kinetics700/label_mapping.json", 'r') as f:
    label_mapping = json.load(f)

# Load deleted files data
deleted_files_doc = Path("docs/kinetics700_deleted_features.md")
with open(deleted_files_doc, 'r', encoding='utf-8') as f:
    deleted_doc_text = f.read()

print("=" * 80)
print("ACTION TRANSFORMER PERFORMANCE ANALYSIS")
print("=" * 80)

# Parse classification report
class_metrics = {}
lines = report_text.split('\n')
for line in lines:
    if line.strip() and not line.startswith('precision') and not line.startswith('accuracy') and not line.startswith('macro') and not line.startswith('weighted'):
        parts = line.split()
        if len(parts) >= 5:
            class_name = ' '.join(parts[:-4])
            if class_name:
                try:
                    precision = float(parts[-4])
                    recall = float(parts[-3])
                    f1 = float(parts[-2])
                    support = int(parts[-1])
                    class_metrics[class_name] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'support': support
                    }
                except ValueError:
                    continue

print(f"\nTotal classes analyzed: {len(class_metrics)}")
print(f"Expected classes: 86")

# 1. Analyze per-class performance
print("\n" + "=" * 80)
print("1. PER-CLASS PERFORMANCE ANALYSIS")
print("=" * 80)

# Sort by F1 score
sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)

print("\n🏆 TOP 10 PERFORMING CLASSES (by F1-score):")
print("-" * 80)
for i, (cls, metrics) in enumerate(sorted_classes[:10], 1):
    print(f"{i:2d}. {cls:30s} | F1: {metrics['f1']:.3f} | P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | Support: {metrics['support']}")

print("\n💀 BOTTOM 10 PERFORMING CLASSES (by F1-score):")
print("-" * 80)
for i, (cls, metrics) in enumerate(sorted_classes[-10:], 1):
    print(f"{i:2d}. {cls:30s} | F1: {metrics['f1']:.3f} | P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | Support: {metrics['support']}")

# 2. Check for completely failing classes
print("\n" + "=" * 80)
print("2. COMPLETELY FAILING CLASSES (F1 = 0.00)")
print("=" * 80)

zero_f1_classes = [cls for cls, metrics in class_metrics.items() if metrics['f1'] == 0.0]
print(f"\nTotal classes with 0% F1-score: {len(zero_f1_classes)}")
print("-" * 80)

# Categorize by type
engagement_zero = []
disengagement_zero = []

engagement_keywords = ['applaud', 'clap', 'cheer', 'headbang', 'dance', 'sing', 'play', 'record', 'megaphone']
disengagement_keywords = ['phone', 'text', 'sleep', 'yawn', 'star', 'watch', 'read', 'wait', 'fidget', 'twiddle', 
                          'drum fingers', 'tap', 'roll', 'shake', 'cross', 'argu', 'smoke', 'check', 'eat', 'drink',
                          'sip', 'wake', 'stretch', 'cough', 'sneez', 'blow', 'cry', 'burp', 'drool', 'chew', 'fall', 'wink']

for cls in zero_f1_classes:
    if any(kw in cls.lower() for kw in engagement_keywords):
        engagement_zero.append(cls)
    elif any(kw in cls.lower() for kw in disengagement_keywords):
        disengagement_zero.append(cls)
    else:
        disengagement_zero.append(cls)  # Default to disengagement

print(f"\n🎉 Engagement actions failing (0% F1): {len(engagement_zero)}")
for cls in engagement_zero:
    print(f"   - {cls}")

print(f"\n😴 Disengagement actions failing (0% F1): {len(disengagement_zero)}")
for cls in disengagement_zero:
    print(f"   - {cls}")

# 3. Performance patterns
print("\n" + "=" * 80)
print("3. PERFORMANCE PATTERNS BY ACTION CATEGORY")
print("=" * 80)

# Musical instruments
instrument_classes = [cls for cls in class_metrics.keys() if 'playing' in cls.lower()]
instrument_f1_avg = sum(class_metrics[cls]['f1'] for cls in instrument_classes) / len(instrument_classes) if instrument_classes else 0

# Dancing
dancing_classes = [cls for cls in class_metrics.keys() if 'danc' in cls.lower()]
dancing_f1_avg = sum(class_metrics[cls]['f1'] for cls in dancing_classes) / len(dancing_classes) if dancing_classes else 0

# Eating
eating_classes = [cls for cls in class_metrics.keys() if 'eating' in cls.lower()]
eating_f1_avg = sum(class_metrics[cls]['f1'] for cls in eating_classes) / len(eating_classes) if eating_classes else 0

# Smoking
smoking_classes = [cls for cls in class_metrics.keys() if 'smok' in cls.lower()]
smoking_f1_avg = sum(class_metrics[cls]['f1'] for cls in smoking_classes) / len(smoking_classes) if smoking_classes else 0

# Stretching
stretching_classes = [cls for cls in class_metrics.keys() if 'stretch' in cls.lower()]
stretching_f1_avg = sum(class_metrics[cls]['f1'] for cls in stretching_classes) / len(stretching_classes) if stretching_classes else 0

print(f"\n🎸 Playing Instruments ({len(instrument_classes)} classes): Avg F1 = {instrument_f1_avg:.3f}")
print(f"💃 Dancing ({len(dancing_classes)} classes): Avg F1 = {dancing_f1_avg:.3f}")
print(f"🍕 Eating ({len(eating_classes)} classes): Avg F1 = {eating_f1_avg:.3f}")
print(f"🚬 Smoking ({len(smoking_classes)} classes): Avg F1 = {smoking_f1_avg:.3f}")
print(f"🤸 Stretching ({len(stretching_classes)} classes): Avg F1 = {stretching_f1_avg:.3f}")

# Overall engagement vs disengagement
engagement_classes = []
disengagement_classes = []

for cls in class_metrics.keys():
    if any(kw in cls.lower() for kw in engagement_keywords):
        engagement_classes.append(cls)
    else:
        disengagement_classes.append(cls)

engagement_f1_avg = sum(class_metrics[cls]['f1'] for cls in engagement_classes) / len(engagement_classes) if engagement_classes else 0
disengagement_f1_avg = sum(class_metrics[cls]['f1'] for cls in disengagement_classes) / len(disengagement_classes) if disengagement_classes else 0

print(f"\n🎉 Overall ENGAGEMENT actions ({len(engagement_classes)} classes): Avg F1 = {engagement_f1_avg:.3f}")
print(f"😴 Overall DISENGAGEMENT actions ({len(disengagement_classes)} classes): Avg F1 = {disengagement_f1_avg:.3f}")

# 4. Correlate with deleted files
print("\n" + "=" * 80)
print("4. CORRELATION WITH DELETED FILES")
print("=" * 80)

# Extract deleted file counts from documentation
deleted_counts = {}
# Pattern: `action_name` - XX files
pattern = r'`([^`]+)` - (\d+) files'
matches = re.findall(pattern, deleted_doc_text)
for action, count in matches:
    deleted_counts[action] = int(count)

print(f"\nTotal action classes with deletions tracked: {len(deleted_counts)}")

# Find classes with deletions and check their performance
affected_classes = []
for cls in class_metrics.keys():
    if cls in deleted_counts:
        affected_classes.append({
            'class': cls,
            'deleted': deleted_counts[cls],
            'f1': class_metrics[cls]['f1'],
            'support': class_metrics[cls]['support']
        })

affected_classes.sort(key=lambda x: x['deleted'], reverse=True)

print(f"\nClasses affected by deletions: {len(affected_classes)}")
print("\nTOP 15 MOST AFFECTED (by number of deleted files):")
print("-" * 80)
print(f"{'Class':<30} {'Deleted':>8} {'F1-Score':>10} {'Support':>8}")
print("-" * 80)
for cls_info in affected_classes[:15]:
    print(f"{cls_info['class']:<30} {cls_info['deleted']:>8} {cls_info['f1']:>10.3f} {cls_info['support']:>8}")

# Check if there's correlation between deletions and poor performance
high_deletion = [c for c in affected_classes if c['deleted'] > 30]
avg_f1_high_deletion = sum(c['f1'] for c in high_deletion) / len(high_deletion) if high_deletion else 0

low_deletion = [c for c in affected_classes if c['deleted'] <= 10]
avg_f1_low_deletion = sum(c['f1'] for c in low_deletion) / len(low_deletion) if low_deletion else 0

print(f"\n📊 Correlation Analysis:")
print(f"   Classes with >30 deletions ({len(high_deletion)}): Avg F1 = {avg_f1_high_deletion:.3f}")
print(f"   Classes with ≤10 deletions ({len(low_deletion)}): Avg F1 = {avg_f1_low_deletion:.3f}")
if avg_f1_high_deletion < avg_f1_low_deletion:
    print(f"   ⚠️  HIGH DELETIONS CORRELATE WITH WORSE PERFORMANCE")
else:
    print(f"   ✅ No strong correlation between deletions and performance")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
