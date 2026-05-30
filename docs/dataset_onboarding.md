# HBCU Multi-Modal Engagement Dataset Onboarding

## Dataset Overview

The HBCU Multi-Modal Engagement dataset is a comprehensive collection of multimodal data designed for analyzing audience engagement in educational and performance settings. This dataset includes video recordings of audiences during various events, with annotations for engagement levels.

## Dataset Components

### Video Data
- **Format**: MP4 video files
- **Resolution**: Typically 1080p or 720p
- **Frame Rate**: 30 FPS
- **Duration**: Variable (typically 30 seconds to 5 minutes per clip)
- **Content**: Audience members during concerts, performances, and educational sessions

### Annotations
- **Engagement Levels**: Categorical labels (Low, Medium, High) or continuous scores (0-1)
- **Temporal Annotations**: Frame-level or segment-level engagement annotations
- **Individual vs. Crowd**: Both individual person annotations and overall crowd engagement scores
- **Format**: JSON or CSV files with timestamps and engagement scores

### Metadata
- **Event Information**: Type of performance, venue, demographics
- **Camera Information**: Viewing angle, distance from audience
- **Environmental Factors**: Lighting conditions, crowd density

## Data Structure

```
data/raw/hbcu_engagement/
├── videos/
│   ├── concert_001.mp4
│   ├── concert_002.mp4
│   └── ...
├── annotations/
│   ├── engagement_labels.json
│   ├── temporal_annotations.csv
│   └── metadata.json
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## Download Instructions

### Option 1: Direct Download (if available)
```bash
# Replace with actual download URLs when available
wget https://example.com/hbcu_engagement_dataset.zip
unzip hbcu_engagement_dataset.zip -d data/raw/
```

### Option 2: Academic Access
1. Visit the official dataset website
2. Request access through academic channels
3. Download using provided credentials
4. Extract to `data/raw/hbcu_engagement/`

### Option 3: Manual Setup
If the dataset is not publicly available, you can use sample data or similar engagement datasets:
1. Create the directory structure above
2. Add your own video files of audience/crowd scenarios
3. Create engagement annotations manually or use crowd-sourcing

## Data Preprocessing Pipeline

1. **Video Preprocessing**
   - Standardize resolution and frame rate
   - Extract frames for pose estimation
   - Create video segments for analysis

2. **Annotation Processing**
   - Parse engagement labels
   - Align annotations with video frames
   - Create train/validation/test splits

3. **Quality Checks**
   - Verify video integrity
   - Check annotation completeness
   - Validate temporal alignment

## Usage in Project

The data loader (`src/dwpose_engagement/data_loader.py`) will:
- Load video files and annotations
- Handle data splits for training/validation/testing
- Provide batch processing for DWPose inference
- Manage engagement label encoding

## Data Ethics and Usage

- Ensure proper consent for all video data
- Follow privacy guidelines for audience footage
- Respect licensing terms of the dataset
- Consider bias and representation in the data

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Ensure all video processing libraries are installed
2. **File Permissions**: Check read/write permissions for data directories
3. **Memory Usage**: Large video files may require efficient loading strategies
4. **Annotation Mismatch**: Verify temporal alignment between videos and labels

### Contact
For dataset-specific issues, refer to the original dataset documentation or contact the dataset maintainers.

## Next Steps

1. Download and organize the dataset following the structure above
2. Run the data validation script (to be created)
3. Explore the data using the provided notebooks
4. Begin DWPose inference pipeline development