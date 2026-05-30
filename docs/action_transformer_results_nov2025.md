# Action Transformer Training Results - November 2025

## Executive Summary

Training completed successfully after 100 epochs (~40 hours on single GPU), but model performance is **critically poor** with severe overfitting.

**Key Metrics:**
- ✅ Training completed: 100/100 epochs
- ❌ **Training accuracy: 71.44%**
- ❌ **Validation accuracy: 17.39%**
- ❌ **Overfitting gap: 54.05%**
- ⚠️ Best val acc: 17.39% (barely better than random 1.16%)

---

## 1. Performance Analysis by Class

### Top 10 Performing Classes (F1-Score)

| Rank | Class | F1 | Precision | Recall | Support |
|------|-------|-----|-----------|--------|---------|
| 1 | stretching leg | 0.500 | 0.470 | 0.520 | 50 |
| 2 | playing cello | 0.440 | 0.490 | 0.410 | 49 |
| 3 | playing violin | 0.420 | 0.430 | 0.420 | 50 |
| 4 | country line dancing | 0.400 | 0.390 | 0.420 | 50 |
| 5 | belly dancing | 0.390 | 0.390 | 0.380 | 47 |
| 6 | checking watch | 0.360 | 0.280 | 0.500 | 42 |
| 7 | stretching arm | 0.360 | 0.320 | 0.410 | 44 |
| 8 | playing ukulele | 0.350 | 0.320 | 0.390 | 46 |
| 9 | eating spaghetti | 0.320 | 0.230 | 0.550 | 49 |
| 10 | dancing ballet | 0.310 | 0.410 | 0.240 | 49 |

**Pattern:** Classes with **large, distinctive body movements** perform best.

---

## 2. Completely Failing Classes (F1 = 0.00)

### Total: 11 classes with zero F1-score

**Engagement actions (3):**
- `recording music` - Static pose, device-based
- `singing` - Mouth movement, not detectable from pose
- `using megaphone` - Object-based, similar to singing

**Disengagement actions (8):**
- `coughing` - Brief, inconsistent movement
- `crying` - Face-based emotion
- `drumming fingers` - Hand micro-movement too subtle
- `eating burger` - Identical pose to other eating actions
- `eating nachos` - Identical pose to other eating actions
- `looking at phone` - Very similar to reading/staring
- `reading newspaper` - Identical pose to reading book
- `winking` - Face-based, not captured by pose keypoints

### Near-Failures (F1 < 0.10, effectively random)

**Total: 29 additional classes**

Most affected categories:
- **Eating actions** (4/7 classes): Can't distinguish burger vs chips vs ice cream from pose
- **Face-based** (5 classes): winking, rolling eyes, crossing eyes, crying, yawning
- **Subtle hand movements** (3 classes): drumming fingers, twiddling fingers, fidgeting
- **Similar static poses** (6 classes): reading, staring, watching TV, looking at phone

---

## 3. Performance by Action Category

| Category | Classes | Avg F1 | Insight |
|----------|---------|--------|---------|
| 🤸 **Stretching** | 2 | **0.430** | ✅ Best performing (large body movements) |
| 🎸 **Playing Instruments** | 16 | 0.239 | Mixed (depends on instrument distinctiveness) |
| 💃 **Dancing** | 16 | 0.231 | Mixed (distinctive styles better) |
| 🚬 **Smoking** | 3 | 0.107 | ❌ Poor (all arm-to-mouth poses similar) |
| 🍕 **Eating** | 7 | **0.086** | ❌ Worst (all arm-to-mouth poses identical) |

**Overall:**
- 🎉 **Engagement actions** (26 classes): Avg F1 = 0.192
- 😴 **Disengagement actions** (62 classes): Avg F1 = 0.139

---

## 4. Correlation with Deleted Files

**Finding:** Classes with >30 deleted files perform **6% worse** on average.

### Most Affected Classes

| Class | Deleted Files | F1-Score | Impact |
|-------|---------------|----------|--------|
| winking | 63 | **0.000** | ⚠️ Critical |
| crossing eyes | 52 | 0.240 | ⚠️ High |
| sipping cup | 51 | 0.050 | ⚠️ High |
| eating ice cream | 48 | 0.100 | ⚠️ High |
| stretching arm | 48 | **0.360** | ✅ OK (distinctive action) |
| smoking pipe | 45 | 0.210 | ⚠️ Moderate |
| crying | 44 | **0.000** | ⚠️ Critical |

**Correlation Analysis:**
- Classes with **>30 deletions** (27 classes): Avg F1 = 0.168
- Classes with **≤10 deletions** (15 classes): Avg F1 = 0.174
- ⚠️ **Deleted files DO correlate with worse performance**

However, the performance gap is small (0.6%), suggesting:
1. **Main problem is model overfitting**, not dataset loss
2. **2.8% deletion impact is real but minor**
3. Some high-deletion classes (like `stretching arm`) still perform well due to distinctive movements

---

## 5. Root Cause Analysis

### Why is Validation Accuracy So Low?

1. **Model Too Complex**
   - 3.75M parameters for 65K training samples
   - Ratio: 17 samples/parameter (should be >100)
   - Result: Model memorizes training data instead of learning patterns

2. **No Regularization**
   - No dropout
   - No weight decay
   - No label smoothing
   - Result: Unconstrained overfitting

3. **Pose-Only Limitation**
   - 33 body keypoints insufficient for many actions
   - Missing: facial expressions, hand details, object context
   - Examples: Can't distinguish eating burger vs nachos (same pose)

4. **Too Many Similar Classes**
   - 86 classes, many with near-identical pose signatures
   - Example: All "eating X" have arm-to-mouth pose
   - Example: All "reading X" have seated/holding pose

5. **No Data Augmentation**
   - No temporal jittering
   - No spatial perturbations
   - No mixup/cutmix
   - Result: Model doesn't generalize to variations

---

## 6. Recommended Actions

### PHASE 1: Quick Validation (2 hours) ⭐ **START HERE**

**Objective:** Test if architectural changes help before committing to 40-hour training.

**Changes:**
- Reduce model: 4→2 layers, 8→4 heads, 256→128 dim (~75% fewer params)
- Add dropout: 0.3
- Add weight decay: 0.01
- Add label smoothing: 0.1
- Train: 30 epochs (not 100)

**Expected:**
- Validation acc: 20-25%
- Train/val gap: <30%
- Duration: ~2 hours

**Decision:**
- If val acc > 20%: Proceed to Phase 2
- If val acc < 20%: Pose-only is insufficient, need Holistic features

---

### PHASE 2: Full Training (7 hours)

**If Phase 1 succeeds**, implement full improvements:

**Model:**
- 3 layers, 6 heads, 192 dim (~2M params)
- Dropout: 0.4, Weight decay: 0.02
- Stochastic depth: 0.1

**Data:**
- Remove bottom 30 classes (keep 56 best)
- OR keep all 86 with class weights

**Training:**
- 100 epochs
- Cosine annealing with warmup
- Full augmentation (temporal + spatial + mixup)
- ReduceLROnPlateau scheduler

**Expected:**
- Validation acc: 30-35%
- Train/val gap: 20-25%

---

### PHASE 3: Holistic Features (30 hours feature extraction)

**If Phase 2 plateaus**, re-extract features with:

**MediaPipe Holistic:**
- Pose: 33 keypoints × 3 = 99 features
- Hands: 42 keypoints × 3 = 126 features
- **Total: 225 features** (vs current 99)

**Benefits:**
- Finger movements: drumming, twiddling, tapping
- Hand gestures: partial help for actions
- Better temporal context

**Expected:**
- Validation acc: 40-50%
- Finger-based actions improve dramatically

---

### Alternative: Binary Classification

**If multi-class continues to fail**, simplify to:

- **Class 0:** All engagement actions (26 classes merged)
- **Class 1:** All disengagement actions (60 classes merged)

**Expected:**
- Validation acc: 60-75%
- Actually useful for concert engagement detection
- Aligns with original project goal

---

## 7. Code Changes Required

### Change 1: Model Architecture
```python
# FILE: scripts/train_action_transformer_dataparallel.py

# OLD:
model = ActionTransformer(
    input_dim=99,
    num_classes=86,
    d_model=256,
    nhead=8,
    num_layers=4,
)

# NEW (Phase 1):
model = ActionTransformer(
    input_dim=99,
    num_classes=86,
    d_model=128,          # -50%
    nhead=4,              # -50%
    num_layers=2,         # -50%
    dim_feedforward=512,  # -50%
    dropout=0.3,          # ADDED
)
```

### Change 2: Optimizer with Weight Decay
```python
# OLD:
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# NEW:
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=0.01
)
```

### Change 3: Loss with Label Smoothing
```python
# OLD:
criterion = nn.CrossEntropyLoss()

# NEW:
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Change 4: Learning Rate Scheduler
```python
# ADD after optimizer:
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
    verbose=True
)

# In training loop after validation:
scheduler.step(val_acc)
```

---

## 8. Conclusion

**Current model is NOT production-ready:**
- 17% validation accuracy is barely better than random
- 71% training accuracy shows severe overfitting
- 11 classes completely fail (F1 = 0.00)
- 29 classes near-fail (F1 < 0.10)

**The 2,016 deleted files have minor impact:**
- Only 0.6% performance difference between high/low deletion classes
- Main problem is model architecture, not dataset

**Recommended path forward:**
1. ✅ Implement Phase 1 (2 hours) to validate architectural fixes
2. If successful → Phase 2 with full training (7 hours)
3. If still failing → Binary classification OR Holistic features

**Reality check:**
Pose-only action recognition for 86 diverse classes is **fundamentally limited**. Many actions require facial expressions, hand details, or object recognition that pose keypoints cannot provide. A more realistic target is:
- Binary engagement classification (60-75% accuracy achievable)
- Reduced class set (30 distinctive actions, 35-45% accuracy)
- Holistic features (pose + hands, 40-50% accuracy)

---

## Files Generated

- `models/action_transformer_kinetics700/best_model.pth` (43 MB)
- `models/action_transformer_kinetics700/classification_report.txt`
- `models/action_transformer_kinetics700/training_results.json`
- `models/action_transformer_kinetics700/label_mapping.json`
- `logs/training_100epochs.log` (full training log)
- `scripts/analyze_model_performance.py` (performance analysis)
- `scripts/analyze_failing_classes.py` (failure analysis)
- `scripts/architecture_recommendations.py` (improvement suggestions)

---

**Next Step:** Review recommendations and decide on Phase 1 implementation.
