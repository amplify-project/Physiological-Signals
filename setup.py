#!/usr/bin/env python3
"""
Setup script for the Concert Engagement with DWPose project.

This script verifies the environment, installs dependencies, and sets up
the project for development or production use.
"""

import os
import sys
import subprocess
import platform
import importlib
import urllib.request
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

FACE_WEIGHTS_URL = "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
FACE_WEIGHTS_PATH = Path(__file__).parent / "models" / "20180402-114759-vggface2.pt"


class EnvironmentChecker:
    """Check and verify the development environment."""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.required_python = (3, 8)
        self.errors = []
        self.warnings = []
    
    def check_python_version(self):
        """Check if Python version meets requirements."""
        logger.info(f"Checking Python version: {self.python_version}")
        
        if self.python_version < self.required_python:
            self.errors.append(
                f"Python {self.required_python[0]}.{self.required_python[1]}+ required, "
                f"found {self.python_version[0]}.{self.python_version[1]}"
            )
        else:
            logger.info("✓ Python version check passed")
    
    def check_gpu_availability(self):
        """Check for GPU availability."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✓ CUDA available: {gpu_count} GPU(s) detected")
                logger.info(f"  Primary GPU: {gpu_name}")
            else:
                self.warnings.append("CUDA not available - will use CPU for inference")
                logger.warning("⚠ No CUDA GPUs detected")
        except ImportError:
            self.warnings.append("PyTorch not installed - cannot check GPU")
    
    def check_required_packages(self):
        """Check if required packages are available."""
        required_packages = [
            'numpy', 'pandas', 'opencv-python', 'torch', 'torchvision',
            'scikit-learn', 'matplotlib', 'seaborn', 'jupyter'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                # Handle special cases
                if package == 'opencv-python':
                    import cv2
                elif package == 'scikit-learn':
                    import sklearn
                else:
                    importlib.import_module(package)
                logger.info(f"✓ {package} available")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"✗ {package} not found")
        
        if missing_packages:
            self.errors.append(f"Missing packages: {', '.join(missing_packages)}")
    
    def check_mmpose_installation(self):
        """Check if MMPose and related packages are installed."""
        mmpose_packages = ['mmcv', 'mmdet', 'mmpose']
        
        for package in mmpose_packages:
            try:
                importlib.import_module(package)
                logger.info(f"✓ {package} available")
            except ImportError:
                self.warnings.append(f"MMPose package {package} not found")
                logger.warning(f"⚠ {package} not installed")
    
    def check_directory_structure(self):
        """Check if project directory structure is correct."""
        required_dirs = [
            'src/dwpose_engagement',
            'configs',
            'data/raw',
            'data/processed',
            'notebooks',
            'models',
            'docs'
        ]
        
        project_root = Path(__file__).parent
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            if full_path.exists():
                logger.info(f"✓ {dir_path} exists")
            else:
                self.warnings.append(f"Directory {dir_path} not found")
                logger.warning(f"⚠ {dir_path} missing")
                # Create missing directories
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"  Created {dir_path}")
    
    def check_config_files(self):
        """Check if configuration files exist."""
        config_files = ['configs/default.yaml']
        
        project_root = Path(__file__).parent
        
        for config_file in config_files:
            config_path = project_root / config_file
            if config_path.exists():
                logger.info(f"✓ {config_file} exists")
            else:
                self.warnings.append(f"Configuration file {config_file} not found")
    
    def run_all_checks(self):
        """Run all environment checks."""
        logger.info("=" * 50)
        logger.info("ENVIRONMENT VERIFICATION")
        logger.info("=" * 50)
        
        self.check_python_version()
        self.check_required_packages()
        self.check_gpu_availability()
        self.check_mmpose_installation()
        self.check_directory_structure()
        self.check_config_files()
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 50)
        
        if self.errors:
            logger.error("ERRORS FOUND:")
            for error in self.errors:
                logger.error(f"  ✗ {error}")
        
        if self.warnings:
            logger.warning("WARNINGS:")
            for warning in self.warnings:
                logger.warning(f"  ⚠ {warning}")
        
        if not self.errors and not self.warnings:
            logger.info("✓ All checks passed! Environment is ready.")
            return True
        elif not self.errors:
            logger.info("✓ Environment is functional with minor warnings.")
            return True
        else:
            logger.error("✗ Environment setup incomplete. Please address errors.")
            return False


def install_dependencies():
    """Install Python dependencies from requirements.txt."""
    logger.info("Installing Python dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found!")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "facenet-pytorch==2.6.0", "--no-deps"
        ])
        logger.info("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def ensure_face_id_weights():
    """Download FaceNet weights used by face identification if missing."""
    logger.info("Ensuring Face ID weights are available...")
    FACE_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if FACE_WEIGHTS_PATH.exists():
        logger.info("✓ Face ID weights already present")
        return True

    try:
        urllib.request.urlretrieve(FACE_WEIGHTS_URL, FACE_WEIGHTS_PATH)
        logger.info("✓ Face ID weights downloaded")
        return True
    except Exception as exc:
        logger.error(f"Failed to download Face ID weights: {exc}")
        logger.error(f"Download manually from {FACE_WEIGHTS_URL}")
        logger.error(f"Place the file at {FACE_WEIGHTS_PATH}")
        return False


def setup_data_directories():
    """Set up data directory structure."""
    logger.info("Setting up data directories...")
    
    project_root = Path(__file__).parent
    data_dirs = [
        "data/raw/hbcu_engagement/videos",
        "data/raw/hbcu_engagement/annotations",
        "data/raw/hbcu_engagement/splits",
        "data/processed/poses",
        "data/processed/features",
        "models/checkpoints",
        "models/exports",
        "results",
        "logs",
        "plots"
    ]
    
    for dir_path in data_dirs:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created {dir_path}")
    
    # Create placeholder files
    placeholders = [
        ("data/raw/hbcu_engagement/videos/.gitkeep", "# Video files go here\n"),
        ("data/raw/hbcu_engagement/annotations/.gitkeep", "# Annotation files go here\n"),
        ("data/processed/poses/.gitkeep", "# Pose keypoint files go here\n"),
        ("models/checkpoints/.gitkeep", "# Model checkpoints go here\n")
    ]
    
    for file_path, content in placeholders:
        full_path = project_root / file_path
        if not full_path.exists():
            full_path.write_text(content)
            logger.info(f"✓ Created placeholder {file_path}")


def create_sample_data():
    """Create sample data for testing."""
    logger.info("Creating sample data for testing...")
    
    project_root = Path(__file__).parent
    
    # Create sample engagement labels
    sample_labels = {
        "sample_video_001": 0.8,
        "sample_video_002": 0.3,
        "sample_video_003": 0.6
    }
    
    labels_file = project_root / "data/raw/hbcu_engagement/annotations/engagement_labels.json"
    if not labels_file.exists():
        import json
        with open(labels_file, 'w') as f:
            json.dump(sample_labels, f, indent=2)
        logger.info("✓ Created sample engagement labels")
    
    # Create sample metadata
    sample_metadata = {
        "sample_video_001": {
            "event_type": "concert",
            "venue": "university_auditorium",
            "crowd_size": 150,
            "lighting": "dim"
        },
        "sample_video_002": {
            "event_type": "lecture",
            "venue": "classroom",
            "crowd_size": 30,
            "lighting": "bright"
        }
    }
    
    metadata_file = project_root / "data/raw/hbcu_engagement/annotations/metadata.json"
    if not metadata_file.exists():
        import json
        with open(metadata_file, 'w') as f:
            json.dump(sample_metadata, f, indent=2)
        logger.info("✓ Created sample metadata")
    
    # Create sample split files
    splits = {
        "train.txt": ["sample_video_001", "sample_video_002"],
        "val.txt": ["sample_video_003"],
        "test.txt": ["sample_video_003"]
    }
    
    for split_file, video_ids in splits.items():
        split_path = project_root / f"data/raw/hbcu_engagement/splits/{split_file}"
        if not split_path.exists():
            with open(split_path, 'w') as f:
                f.write('\n'.join(video_ids))
            logger.info(f"✓ Created {split_file}")


def main():
    """Main setup function."""
    print("🎵 Concert Engagement with DWPose - Setup Script 🕺")
    print("=" * 60)
    
    # Check environment
    checker = EnvironmentChecker()
    env_ok = checker.run_all_checks()
    
    if not env_ok:
        print("\n❌ Environment check failed. Please fix errors before continuing.")
        return 1
    
    # Ask user what to do
    print("\n" + "=" * 60)
    print("SETUP OPTIONS")
    print("=" * 60)
    print("1. Install dependencies only")
    print("2. Set up directories only") 
    print("3. Create sample data only")
    print("4. Full setup (all of the above)")
    print("5. Skip setup")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1" or choice == "4":
            install_dependencies()
            ensure_face_id_weights()
        
        if choice == "2" or choice == "4":
            setup_data_directories()
        
        if choice == "3" or choice == "4":
            create_sample_data()
        
        if choice == "5":
            print("Setup skipped.")
        
        print("\n" + "=" * 60)
        print("🎉 Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Download the HBCU Multi-Modal Engagement dataset")
        print("2. Install DWPose model weights")
        print("3. Run the exploration notebook: notebooks/01_data_exploration.ipynb")
        print("4. Check the documentation: docs/dataset_onboarding.md")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        return 1


if __name__ == "__main__":
    sys.exit(main())