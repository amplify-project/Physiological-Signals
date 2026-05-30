"""
Data loading utilities for the HBCU Multi-Modal Engagement dataset.

This module provides classes and functions for loading video data, annotations,
and managing data splits for training engagement prediction models.
"""

import os
import json
import csv
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EngagementSample:
    """Data structure for a single engagement sample."""
    video_path: str
    frames: np.ndarray
    engagement_score: float
    metadata: Dict
    frame_indices: List[int]
    duration: float


class HBCUEngagementDataset(Dataset):
    """
    PyTorch Dataset for HBCU Multi-Modal Engagement data.
    
    Loads video clips and corresponding engagement annotations.
    Supports both frame-level and clip-level engagement labels.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        clip_duration: float = 5.0,
        frame_rate: int = 30,
        target_resolution: Tuple[int, int] = (256, 256),
        engagement_type: str = "continuous",  # "continuous" or "categorical"
        transform=None,
        load_frames: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_root: Path to the root directory containing the dataset
            split: Data split ("train", "val", "test")
            clip_duration: Duration of video clips in seconds
            frame_rate: Target frame rate for video processing
            target_resolution: Target resolution (width, height) for frames
            engagement_type: Type of engagement labels ("continuous" or "categorical")
            transform: Optional transform to apply to frames
            load_frames: Whether to load actual frame data (False for metadata only)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.clip_duration = clip_duration
        self.frame_rate = frame_rate
        self.target_resolution = target_resolution
        self.engagement_type = engagement_type
        self.transform = transform
        self.load_frames = load_frames
        
        # Load dataset components
        self.video_dir = self.data_root / "videos"
        self.annotations_dir = self.data_root / "annotations"
        self.splits_dir = self.data_root / "splits"
        
        # Load split file
        self._load_split()
        
        # Load annotations
        self._load_annotations()
        
        # Prepare samples
        self._prepare_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_split(self):
        """Load the data split file."""
        split_file = self.splits_dir / f"{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.video_ids = [line.strip() for line in f.readlines()]
    
    def _load_annotations(self):
        """Load engagement annotations."""
        # Load main engagement labels
        engagement_file = self.annotations_dir / "engagement_labels.json"
        if engagement_file.exists():
            with open(engagement_file, 'r') as f:
                self.engagement_labels = json.load(f)
        else:
            logger.warning(f"Engagement labels file not found: {engagement_file}")
            self.engagement_labels = {}
        
        # Load temporal annotations if available
        temporal_file = self.annotations_dir / "temporal_annotations.csv"
        if temporal_file.exists():
            self.temporal_annotations = pd.read_csv(temporal_file)
        else:
            logger.warning(f"Temporal annotations file not found: {temporal_file}")
            self.temporal_annotations = pd.DataFrame()
        
        # Load metadata
        metadata_file = self.annotations_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            logger.warning(f"Metadata file not found: {metadata_file}")
            self.metadata = {}
    
    def _prepare_samples(self):
        """Prepare individual samples from video files and annotations."""
        self.samples = []
        
        for video_id in self.video_ids:
            video_path = self.video_dir / f"{video_id}.mp4"
            if not video_path.exists():
                logger.warning(f"Video file not found: {video_path}")
                continue
            
            # Get video info
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / original_fps
            cap.release()
            
            # Get engagement score
            engagement_score = self._get_engagement_score(video_id)
            if engagement_score is None:
                logger.warning(f"No engagement score found for video: {video_id}")
                continue
            
            # Create clips from the video
            num_clips = max(1, int(duration / self.clip_duration))
            
            for clip_idx in range(num_clips):
                start_time = clip_idx * self.clip_duration
                end_time = min((clip_idx + 1) * self.clip_duration, duration)
                
                # Calculate frame indices
                start_frame = int(start_time * original_fps)
                end_frame = int(end_time * original_fps)
                frame_indices = list(range(start_frame, end_frame))
                
                # Get metadata for this video
                video_metadata = self.metadata.get(video_id, {})
                
                sample = EngagementSample(
                    video_path=str(video_path),
                    frames=None,  # Will be loaded in __getitem__ if needed
                    engagement_score=engagement_score,
                    metadata=video_metadata,
                    frame_indices=frame_indices,
                    duration=end_time - start_time
                )
                
                self.samples.append(sample)
    
    def _get_engagement_score(self, video_id: str) -> Optional[float]:
        """Get engagement score for a video."""
        if video_id in self.engagement_labels:
            score = self.engagement_labels[video_id]
            
            # Convert categorical to continuous if needed
            if self.engagement_type == "continuous":
                if isinstance(score, str):
                    # Convert categorical labels to continuous scores
                    score_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
                    return score_map.get(score.lower(), 0.5)
                return float(score)
            else:
                # Convert continuous to categorical if needed
                if isinstance(score, (int, float)):
                    if score < 0.33:
                        return 0  # low
                    elif score < 0.67:
                        return 1  # medium
                    else:
                        return 2  # high
                else:
                    # Map string labels to integers
                    label_map = {"low": 0, "medium": 1, "high": 2}
                    return label_map.get(score.lower(), 1)
        
        return None
    
    def _load_video_frames(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """Load specific frames from a video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame
                frame = cv2.resize(frame, self.target_resolution)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                logger.warning(f"Could not read frame {frame_idx} from {video_path}")
        
        cap.release()
        
        if frames:
            return np.stack(frames)
        else:
            # Return dummy frames if none could be loaded
            dummy_frame = np.zeros((*self.target_resolution[::-1], 3), dtype=np.uint8)
            return np.stack([dummy_frame] * len(frame_indices))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        sample = self.samples[idx]
        
        result = {
            "video_path": sample.video_path,
            "engagement_score": sample.engagement_score,
            "metadata": sample.metadata,
            "duration": sample.duration,
            "frame_indices": sample.frame_indices
        }
        
        # Load frames if requested
        if self.load_frames:
            frames = self._load_video_frames(sample.video_path, sample.frame_indices)
            
            if self.transform:
                frames = self.transform(frames)
            
            result["frames"] = frames
        
        return result


def create_data_loaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    clip_duration: float = 5.0,
    target_resolution: Tuple[int, int] = (256, 256),
    engagement_type: str = "continuous"
) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_root: Path to the dataset root directory
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        clip_duration: Duration of video clips in seconds
        target_resolution: Target resolution for frames
        engagement_type: Type of engagement labels
    
    Returns:
        Dictionary containing data loaders for each split
    """
    data_loaders = {}
    
    for split in ["train", "val", "test"]:
        try:
            dataset = HBCUEngagementDataset(
                data_root=data_root,
                split=split,
                clip_duration=clip_duration,
                target_resolution=target_resolution,
                engagement_type=engagement_type
            )
            
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            data_loaders[split] = data_loader
            
        except FileNotFoundError as e:
            logger.warning(f"Could not create data loader for {split}: {e}")
    
    return data_loaders


def validate_dataset(data_root: str) -> bool:
    """
    Validate the dataset structure and files.
    
    Args:
        data_root: Path to the dataset root directory
    
    Returns:
        True if dataset is valid, False otherwise
    """
    data_root = Path(data_root)
    
    # Check directory structure
    required_dirs = ["videos", "annotations", "splits"]
    for dir_name in required_dirs:
        dir_path = data_root / dir_name
        if not dir_path.exists():
            logger.error(f"Required directory not found: {dir_path}")
            return False
    
    # Check split files
    splits_dir = data_root / "splits"
    for split in ["train.txt", "val.txt", "test.txt"]:
        split_file = splits_dir / split
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
    
    # Check annotation files
    annotations_dir = data_root / "annotations"
    annotation_files = ["engagement_labels.json", "metadata.json"]
    for file_name in annotation_files:
        file_path = annotations_dir / file_name
        if not file_path.exists():
            logger.warning(f"Annotation file not found: {file_path}")
    
    logger.info("Dataset validation completed")
    return True


if __name__ == "__main__":
    # Example usage
    data_root = "data/raw/hbcu_engagement"
    
    if validate_dataset(data_root):
        # Create data loaders
        data_loaders = create_data_loaders(
            data_root=data_root,
            batch_size=4,
            clip_duration=3.0
        )
        
        # Test loading a batch
        if "train" in data_loaders:
            train_loader = data_loaders["train"]
            for batch in train_loader:
                print(f"Batch shape: {batch['frames'].shape}")
                print(f"Engagement scores: {batch['engagement_score']}")
                break