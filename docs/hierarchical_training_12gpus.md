# Hierarchical Training Strategy - 12 GPUs

## Overview
Train the same model at three different label granularities to determine the optimal classification level for pose-based engagement detection.

## Hierarchical Levels

### 1. FINE (86 classes) - Hardest
- **Classes**: 86 fine-grained action classes
- **Examples**: applauding, clapping, belly dancing, breakdancing, texting, etc.
- **Expected Accuracy**: ~17-20% (baseline from previous training)
- **Purpose**: Establish baseline, show that fine-grained is too hard
- **Script**: `scripts/launch_12gpus_fine.sh`
- **Output**: `models/action_transformer_12gpus_fine/`
- **Log**: `logs/training_12gpus_fine.log`

### 2. MACRO (13 classes) - Medium
- **Classes**: 13 semantic categories
  - **Engagement**: applause, dancing, cheering, singing, playing_instrument, recording
  - **Disengagement**: phone_distraction, passive, fidgeting, negative_body_language, smoking, checking_time, eating_drinking, tired_uncomfortable, negative_reactions
- **Expected Accuracy**: ~50-60%
- **Purpose**: Test if semantic grouping improves accuracy
- **Script**: `scripts/launch_12gpus_macro.sh`
- **Output**: `models/action_transformer_12gpus_macro/`
- **Log**: `logs/training_12gpus_macro.log`

### 3. BINARY (2 classes) - Easiest ⭐
- **Classes**: 2 binary classes
  - **engagement (1)**: All engagement actions
  - **disengagement (0)**: All disengagement actions
- **Expected Accuracy**: **>75%** (target)
- **Purpose**: Validate pose features for engagement detection
- **Script**: `scripts/launch_12gpus_binary.sh`
- **Output**: `models/action_transformer_12gpus_binary/`
- **Log**: `logs/training_12gpus_binary.log`

## Training Configuration

All three levels use the same configuration:
- **GPUs**: 12× Tesla T4
- **Batch Size**: 32 per GPU (384 effective)
- **Epochs**: 100
- **Learning Rate**: 1e-4
- **Sequence Length**: 300 frames
- **Backend**: Gloo (CPU-based, works on PCIe GPUs)
- **Duration**: ~3-4 hours per level

## How to Run

### From Linux Server (SSH):
```bash
# 1. Fine-grained (86 classes)
bash scripts/launch_12gpus_fine.sh

# 2. Macro categories (13 classes)
bash scripts/launch_12gpus_macro.sh

# 3. Binary engagement (2 classes) ⭐ RECOMMENDED
bash scripts/launch_12gpus_binary.sh
```

### From Windows (PowerShell):
```powershell
# Connect to server
ssh eoghan@10.0.0.119

# Navigate to workspace
cd /mnt/data/concert_engagement

# Launch training (choose one)
bash scripts/launch_12gpus_binary.sh   # Binary (recommended first)
bash scripts/launch_12gpus_macro.sh    # Macro categories
bash scripts/launch_12gpus_fine.sh     # Fine-grained
```

## Monitoring

### In a NEW PowerShell Window:
```powershell
# Open new PowerShell window
ssh eoghan@10.0.0.119

# Monitor training log
tail -f logs/training_12gpus_binary.log

# Watch GPU usage
watch -n 1 nvidia-smi
```

### Alternative: VS Code Integrated Terminal
- Open a new terminal in VS Code
- Run monitoring commands there

## Expected Results

### Success Criteria:
- **Binary**: >75% validation accuracy → **SUCCESS** ✅
  - If achieved: Pose features are sufficient for engagement detection
  - Ready for real-time deployment

- **Macro**: >50% validation accuracy → **GOOD** 👍
  - Shows semantic grouping helps
  - 13 categories still useful for detailed analytics

- **Fine**: ~17-20% validation accuracy → **BASELINE** 📊
  - Confirms fine-grained is too hard with pose features alone
  - May need additional visual features (RGB, optical flow)

### Decision Tree:
```
Binary >75%?
├── YES ✅ → Deploy for real-time engagement detection
│           → Use macro (13) for detailed analytics
│           → Fine (86) needs more features
│
└── NO ❌  → Need additional features:
            - RGB features (ResNet, ViT)
            - Optical flow
            - Audio features
            - Multi-modal fusion
```

## Technical Notes

### Label Conversion
- Dataset loads fine-grained labels (86 classes)
- Converted on-the-fly during training:
  - `fine → macro`: Using `fine_to_macro` mapping
  - `fine → binary`: Using `fine_to_binary` mapping
- Conversion defined in: `models/action_transformer_kinetics700/hierarchical_labels.json`

### Model Differences
- **Same architecture** for all three levels
- **Different output layers**:
  - Fine: 86 classes in final layer
  - Macro: 13 classes in final layer
  - Binary: 2 classes in final layer

### Why This Approach?
1. **Validate pose features**: Can they detect engagement at all?
2. **Find optimal granularity**: What level works best?
3. **Inform next steps**: Need more features? Or pose is enough?
4. **Practical deployment**: Binary is simplest for real-time use

## Next Steps After Training

1. **Check Results**: Compare validation accuracies across all three levels
2. **Analyze Errors**: Look at confusion matrices for each level
3. **Select Best Model**: 
   - If binary >75% → Deploy for real-time use
   - If macro >50% → Use for detailed analytics
   - If both fail → Need additional features
4. **Real-time Testing**: Test best model with live webcam
5. **Deployment**: Package for AR/VR integration

## Files Modified

### Code Changes:
- `scripts/ddp_precomputed_dataset.py`: Added `label_level` parameter
- `scripts/train_action_transformer_ddp_v2.py`: Added `--label-level` argument

### New Scripts:
- `scripts/launch_12gpus_fine.sh`: Fine-grained training (86 classes)
- `scripts/launch_12gpus_macro.sh`: Macro category training (13 classes)
- `scripts/launch_12gpus_binary.sh`: Binary training (2 classes)

### Existing Resources:
- `models/action_transformer_kinetics700/hierarchical_labels.json`: Label mappings
- `scripts/create_hierarchical_labels.py`: Generates hierarchical mappings
- `data/processed/train_samples.npy`: Pre-computed training samples
- `data/processed/val_samples.npy`: Pre-computed validation samples

## Recommendation

**Start with BINARY training** (`launch_12gpus_binary.sh`):
- Simplest to validate
- Most important for practical use
- Fastest to see if pose features work
- Target >75% accuracy

If successful, the other levels provide:
- **Macro**: More detailed analytics (which type of engagement?)
- **Fine**: Reference baseline (shows pose limitations)

Good luck! 🚀
