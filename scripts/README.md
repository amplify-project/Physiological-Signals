# Scripts Directory

This directory contains all utility scripts for the Concert Engagement Analysis system, organized by purpose.

---

> ⚠️ **IMPORTANT: Always activate the virtual environment before running scripts!**
> 
> ```powershell
> # Windows
> .\.venv\Scripts\Activate.ps1
> 
> # macOS/Linux  
> source .venv/bin/activate
> ```
> 
> You should see `(.venv)` in your terminal prompt. Running without the venv will use system Python which may be missing dependencies or have wrong PyTorch version (CPU instead of CUDA).

---

## 📁 Directory Structure

### `inference/` - Live Inference Scripts
Real-time engagement detection and streaming.

| Script | Description |
|--------|-------------|
| `live_multiperson_binary_v2.py` | **Main inference script** - Multi-person engagement with FPS-adaptive buffer |
| `live_binary_inference.py` | Single-person binary engagement inference |
| `multiperson_engagement.py` | Multi-person pose estimation (older version) |
| `console_subscriber.py` | Redis subscriber for viewing engagement scores |

### `training/` - Model Training Scripts
All training-related scripts and launch configurations.

| Script | Description |
|--------|-------------|
| `train_action_transformer.py` | Base transformer training script |
| `train_action_transformer_ddp_v2.py` | Distributed Data Parallel training (recommended) |
| `train_binary.py` | Binary engagement classifier training |
| `launch_12gpus_*.sh` | Multi-GPU launch scripts for HPC |

### `data/` - Data Processing Scripts
Feature extraction, dataset preparation, and validation.

| Script | Description |
|--------|-------------|
| `extract_kinetics_features.py` | Extract MediaPipe features from Kinetics videos |
| `create_hierarchical_labels.py` | Generate hierarchical engagement labels |
| `download_disengagement_actions.py` | Download Kinetics700 action videos |
| `validate_features.py` | Validate extracted feature files |

### `analysis/` - Analysis & Visualization
Model analysis, performance metrics, and visualizations.

| Script | Description |
|--------|-------------|
| `analyze_model_performance.py` | Generate model performance reports |
| `visualize_model_3d.py` | 3D network architecture visualization |
| `check_data_distribution.py` | Analyze dataset class distributions |
| `architecture_recommendations.py` | Model architecture suggestions |

### `debug/` - Debugging & Testing
Development and troubleshooting utilities.

| Script | Description |
|--------|-------------|
| `debug_cuda_context.py` | Debug CUDA/GPU issues |
| `test_ddp.py` | Test distributed training setup |
| `test_imports.py` | Verify all imports work correctly |
| `debug_dataloader.py` | Debug data loading issues |

### `utils/` - Utilities & Monitoring
System monitoring and helper scripts.

| Script | Description |
|--------|-------------|
| `monitor_training_dashboard.ps1` | Live training metrics dashboard |
| `check_gpu_users.ps1` | Show GPU usage across users |
| `monitor_resources.ps1` | System resource monitoring |
| `generate_pipeline_flowchart.py` | Generate system architecture diagrams |

## 🚀 Quick Start

### Run Live Inference
```bash
# From project root
python scripts/inference/live_multiperson_binary_v2.py
```

### Monitor Training
```powershell
# Windows
.\scripts\utils\monitor_training_dashboard.ps1
```

### View Engagement Scores
```bash
python scripts/inference/console_subscriber.py
```

## 📝 Notes

- All paths are relative to project root
- Activate virtual environment before running: `source .venv/bin/activate` (or `.\.venv\Scripts\Activate.ps1` on Windows)
- Most scripts expect Redis to be running on localhost:6379
