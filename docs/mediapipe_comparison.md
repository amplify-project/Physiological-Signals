# MediaPipe vs DWPose Performance Comparison

## Setup

### 1. Install MediaPipe

```bash
pip install mediapipe
```

MediaPipe is Google's lightweight pose estimation solution with:
- **Holistic model**: Body (33) + Face (468) + Hands (21×2) = 543 keypoints
- **Single person only**: Cannot detect multiple people
- **CPU/GPU support**: Optimized for mobile and edge devices
- **Fast inference**: ~30-60 FPS on CPU

## Quick Comparison (Local Test)

Test both models on sample videos:

```bash
# Compare on DAiSEE Test videos
python scripts/compare_mediapipe_dwpose.py \
    --video-dir "path/to/daisee/Test" \
    --output-dir results/comparison \
    --max-videos 10 \
    --max-frames 100
```

This will:
- Process 10 videos, 100 frames each
- Measure FPS, keypoint count, detection rate
- Generate comparison plots
- Save results to JSON

**Expected Results**:
- MediaPipe: ~30-60 FPS (CPU), ~100-150 FPS (GPU)
- DWPose: ~9-10 FPS (RTX 3080 Ti), ~20-30 FPS (A100)
- MediaPipe: 543 total keypoints (but single person only)
- DWPose: 134 keypoints per person (multi-person capable)

## Full Dataset Extraction (Remote Server)

Extract features from entire DAiSEE dataset with MediaPipe:

```bash
# On server 192.168.200.205
python3 scripts/extract_mediapipe_features.py \
    --dataset-dir ~/datasets/daisee \
    --output-dir ~/datasets/daisee/features_mediapipe \
    --split all \
    --max-frames 300
```

**Expected Performance**:
- CPU: ~2-3 FPS → ~1-2 hours per 1000 videos
- GPU: ~30-50 FPS → ~5-10 minutes per 1000 videos
- **Total time**: 6-12 hours for 8,232 videos (vs 7-8 hours for DWPose)

## Comparison Metrics

### Speed
- **MediaPipe**: Faster on CPU, optimized for mobile
- **DWPose**: Requires GPU, but handles multiple people

### Keypoints
- **MediaPipe**: 543 keypoints (33 body + 468 face + 42 hands)
- **DWPose**: 134 keypoints (17 body + 68 face + 42 hands + 6 feet + 1 root)
- MediaPipe has more face detail, DWPose has feet

### Multi-Person
- **MediaPipe**: ❌ Single person only
- **DWPose**: ✅ Multiple people per frame

### Robustness
- **MediaPipe**: Good on frontal, struggles with occlusion
- **DWPose**: Better with occlusion, multiple poses

### Use Case for Concert
- **MediaPipe**: Good for single-person demos, fast prototyping
- **DWPose**: Better for concert crowds (multi-person essential)

## Training Comparison

After extracting MediaPipe features:

```bash
# Train Transformer on MediaPipe features
python3 scripts/train_transformer_kfold.py \
    --features-dir ~/datasets/daisee/features_mediapipe \
    --output-dir ~/models/engagement_mediapipe \
    --batch-size 32 \
    --epochs 30 \
    --lr 0.0001 \
    --n-folds 10 \
    --device cuda
```

Compare model performance:
- Same architecture, different input features
- MediaPipe: 142 keypoints × 3 = 426 input features
- DWPose: 134 keypoints × 3 = 402 input features

**Expected R² comparison**:
- DWPose: R²=0.584 (current baseline)
- MediaPipe: R²=? (to be determined)

## Decision Criteria

**Choose MediaPipe if**:
- ✅ Single person detection is sufficient
- ✅ Need fast CPU inference
- ✅ Deploying to mobile/edge devices
- ✅ Want detailed face keypoints (468 points)

**Choose DWPose if**:
- ✅ Need multi-person detection (concerts!)
- ✅ Have GPU available
- ✅ Need better occlusion handling
- ✅ Want foot keypoints for full-body analysis

## Next Steps

1. **Quick test** (5 minutes):
   ```bash
   python scripts/compare_mediapipe_dwpose.py \
       --video-dir data/test_videos \
       --max-videos 5
   ```

2. **Full extraction** (6-12 hours):
   ```bash
   ssh -i Enter eoghan@192.168.200.205
   python3 scripts/extract_mediapipe_features.py \
       --dataset-dir ~/datasets/daisee \
       --output-dir ~/datasets/daisee/features_mediapipe \
       --split all
   ```

3. **Train & compare models**:
   - Train Transformer on MediaPipe features
   - Compare R² scores
   - Analyze which keypoints are most predictive

4. **Decision**:
   - If MediaPipe R² ≥ DWPose R², consider switching (faster)
   - If DWPose R² > MediaPipe R², keep DWPose (better quality)
   - For concerts, DWPose likely better due to multi-person

## File Sizes

- DWPose features: ~6,527 files × ~500KB = ~3.3 GB
- MediaPipe features: ~6,527 files × ~500KB = ~3.3 GB (similar)

## Cost-Benefit Analysis

| Aspect | MediaPipe | DWPose |
|--------|-----------|--------|
| Speed (CPU) | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Speed (GPU) | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Multi-person | ❌ | ✅ |
| Keypoint detail | Face: ⭐⭐⭐⭐⭐ | Body: ⭐⭐⭐⭐ |
| Occlusion robustness | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Mobile deployment | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Concert suitability | ⭐⭐ | ⭐⭐⭐⭐⭐ |

**Recommendation for concert engagement**: Stick with **DWPose** due to multi-person detection requirement, but good to benchmark MediaPipe for comparison.
