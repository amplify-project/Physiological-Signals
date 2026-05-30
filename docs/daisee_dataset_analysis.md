# DAiSEE Dataset Analysis

## Dataset Overview

**Status**: ✅ Downloaded and extracted on remote server (`~/datasets/daisee/`)
**Size**: 15GB (18,268 files)
**Format**: Raw videos (.avi) + CSV labels - **NO pre-extracted features**

## Data Structure

### Videos
- **Format**: .avi files (10-second clips)
- **Size**: ~2MB per video
- **Organization**: `DataSet/{Train,Validation,Test}/{SubjectID}/{VideoID}/{VideoID}.avi`
- **Total Videos**: 8,925 labeled clips

### Labels (CSV Format)

```csv
ClipID,Boredom,Engagement,Confusion,Frustration
1100011002.avi,0,2,0,0
1100011003.avi,0,2,0,0
```

**Engagement Levels**: 0 (Very Low), 1 (Low), 2 (High), 3 (Very High)

### Data Splits

| Split | Samples | File |
|-------|---------|------|
| **Train** | 5,358 | `TrainLabels.csv` |
| **Validation** | 1,429 | `ValidationLabels.csv` |
| **Test** | 1,784 | `TestLabels.csv` |
| **Total** | 8,925 | `AllLabels.csv` |

### Engagement Distribution (Training Set)

| Level | Count | Percentage |
|-------|-------|------------|
| 0 (Very Low) | 34 | 0.6% |
| 1 (Low) | 213 | 4.0% |
| 2 (High) | 2,617 | 48.8% |
| 3 (Very High) | 2,494 | 46.6% |

⚠️ **Note**: Dataset is heavily imbalanced toward high engagement (levels 2-3 = 95.4%)

## What We Need to Do

### ✅ What DAiSEE Provides
- Raw video files (.avi, 10-second clips)
- Engagement labels (0-3) in CSV format
- Video-to-label mapping via ClipID

### ❌ What DAiSEE Does NOT Provide
- No DWPose keypoints
- No pre-extracted pose features
- Only helper scripts for HOG features (not compatible with our approach)

## Our Workflow

### Step 1: Extract DWPose Features (On Remote Server)
```bash
# We need to:
1. Upload DWPose inference code to remote server
2. Upload ONNX models (556MB)
3. Process all 8,925 videos → extract 134 keypoints per frame
4. Save as: {VideoID}_keypoints.npy + engagement_label
```

### Step 2: Train Engagement Model
```python
# Input: DWPose keypoints (134 points/person)
# Output: Engagement prediction (0-3)
# Models: XGBoost, LSTM, or hybrid
```

### Step 3: Deploy to Concert Footage
```python
# Real-time inference:
# Webcam → DWPose → Trained Model → Engagement Score
```

## Required Sync to Remote Server

Since DAiSEE has **raw videos only**, we need:

### ✅ Upload to Remote
- `src/dwpose_engagement/wholebody.py` - DWPose inference
- `src/dwpose_engagement/onnxdet.py` - Detection
- `src/dwpose_engagement/onnxpose.py` - Pose estimation
- `src/dwpose_engagement/util.py` - Utilities
- `models/dwpose/*.onnx` - ONNX models (556MB)
- `scripts/extract_daisee_features.py` - Feature extraction script (to create)
- `src/dwpose_engagement/models.py` - Training models
- `src/dwpose_engagement/data_loader.py` - Data loading
- `configs/` - Configuration files

### Processing Plan

**Estimated Time**: 
- 8,925 videos × 10 seconds each = ~25 hours of video
- DWPose @ 14 FPS → ~30 frames/video
- On 4×A100 GPUs with parallelization: **~2-4 hours** for full extraction

**Storage**:
- Keypoints: 134 points × 2 coords × 30 frames × 8925 videos × 4 bytes = ~300MB
- Very manageable!

## Next Steps

1. **Create feature extraction script** (`scripts/extract_daisee_features.py`)
2. **Sync DWPose code + models to remote server**
3. **Run feature extraction on all videos** (parallel processing on 4×A100)
4. **Train engagement model** on extracted features
5. **Download trained model** for local inference

## Key Insight

✅ **Good News**: Labels are clean and well-structured
⚠️ **Challenge**: Need to run DWPose on 8,925 videos on remote server
✅ **Advantage**: 4×A100 GPUs make this feasible in a few hours
