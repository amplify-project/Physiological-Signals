"""
Utility functions for the Concert Engagement with DWPose project.

This module provides helper functions for common tasks like file I/O,
visualization, data preprocessing, and experiment tracking.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to log to
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    logger.info(f"Logging configured with level {log_level}")


def create_experiment_id(experiment_name: str, config: Dict) -> str:
    """
    Create a unique experiment ID based on configuration.
    
    Args:
        experiment_name: Base name for the experiment
        config: Configuration dictionary
    
    Returns:
        Unique experiment ID
    """
    # Create hash from config
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_id = f"{experiment_name}_{timestamp}_{config_hash}"
    return experiment_id


def save_json(data: Dict, filepath: str, indent: int = 2):
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded JSON from {filepath}")
    return data


def save_pickle(data: Any, filepath: str):
    """
    Save data using pickle.
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Saved pickle to {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded pickle from {filepath}")
    return data


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'path': video_path,
        'filename': Path(video_path).name,
        'exists': Path(video_path).exists(),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': 0,
        'file_size_mb': 0
    }
    
    if info['exists']:
        info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
        info['file_size_mb'] = Path(video_path).stat().st_size / (1024 * 1024)
    
    cap.release()
    return info


def create_video_summary(video_dir: str) -> pd.DataFrame:
    """
    Create a summary of all videos in a directory.
    
    Args:
        video_dir: Directory containing videos
    
    Returns:
        DataFrame with video information
    """
    video_dir = Path(video_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    video_data = []
    
    for ext in video_extensions:
        for video_path in video_dir.glob(f'*{ext}'):
            info = get_video_info(str(video_path))
            video_data.append(info)
    
    df = pd.DataFrame(video_data)
    logger.info(f"Found {len(df)} videos in {video_dir}")
    
    return df


def visualize_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    connections: Optional[List[Tuple[int, int]]] = None,
    keypoint_threshold: float = 0.3,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize pose keypoints on an image.
    
    Args:
        image: Input image [H, W, 3]
        keypoints: Keypoints array [N, K, 3] (x, y, confidence)
        connections: List of keypoint connections to draw
        keypoint_threshold: Confidence threshold for drawing keypoints
        save_path: Optional path to save the visualization
    
    Returns:
        Image with keypoints drawn
    """
    vis_image = image.copy()
    
    # Define default connections for body keypoints (COCO-style)
    if connections is None:
        connections = [
            (0, 1), (1, 3), (0, 2), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
    
    # Colors for different people
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
    ]
    
    for person_idx, person_keypoints in enumerate(keypoints):
        color = colors[person_idx % len(colors)]
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            if (start_idx < len(person_keypoints) and 
                end_idx < len(person_keypoints)):
                
                start_point = person_keypoints[start_idx]
                end_point = person_keypoints[end_idx]
                
                if (start_point[2] > keypoint_threshold and 
                    end_point[2] > keypoint_threshold):
                    
                    cv2.line(
                        vis_image,
                        (int(start_point[0]), int(start_point[1])),
                        (int(end_point[0]), int(end_point[1])),
                        color,
                        2
                    )
        
        # Draw keypoints
        for keypoint in person_keypoints:
            if keypoint[2] > keypoint_threshold:
                cv2.circle(
                    vis_image,
                    (int(keypoint[0]), int(keypoint[1])),
                    3,
                    color,
                    -1
                )
    
    if save_path:
        cv2.imwrite(save_path, vis_image)
        logger.info(f"Saved keypoint visualization to {save_path}")
    
    return vis_image


def plot_engagement_distribution(
    engagement_scores: np.ndarray,
    title: str = "Engagement Score Distribution",
    save_path: Optional[str] = None
):
    """
    Plot the distribution of engagement scores.
    
    Args:
        engagement_scores: Array of engagement scores
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(engagement_scores, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Histogram')
    plt.xlabel('Engagement Score')
    plt.ylabel('Frequency')
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(engagement_scores)
    plt.title('Box Plot')
    plt.ylabel('Engagement Score')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None
):
    """
    Plot feature importance scores.
    
    Args:
        feature_names: Names of features
        importance_scores: Importance scores
        top_n: Number of top features to show
        title: Plot title
        save_path: Optional path to save the plot
    """
    # Sort by importance
    indices = np.argsort(importance_scores)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importance_scores[indices]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_features)), top_scores)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        plt.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        title: Plot title
        save_path: Optional path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def create_experiment_report(
    experiment_id: str,
    config: Dict,
    results: Dict,
    save_dir: str
):
    """
    Create a comprehensive experiment report.
    
    Args:
        experiment_id: Unique experiment identifier
        config: Experiment configuration
        results: Experiment results
        save_dir: Directory to save the report
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': results,
        'summary': {
            'model_type': config.get('model_type', 'unknown'),
            'dataset_size': results.get('dataset_size', 0),
            'best_score': results.get('best_score', 0),
            'training_time': results.get('training_time', 0)
        }
    }
    
    # Save full report as JSON
    report_path = save_dir / f"{experiment_id}_report.json"
    save_json(report, report_path)
    
    # Create markdown summary
    md_content = f"""# Experiment Report: {experiment_id}

## Summary
- **Experiment ID**: {experiment_id}
- **Timestamp**: {report['timestamp']}
- **Model Type**: {report['summary']['model_type']}
- **Dataset Size**: {report['summary']['dataset_size']}
- **Best Score**: {report['summary']['best_score']:.4f}
- **Training Time**: {report['summary']['training_time']:.2f}s

## Configuration
```json
{json.dumps(config, indent=2)}
```

## Results
```json
{json.dumps(results, indent=2)}
```
"""
    
    md_path = save_dir / f"{experiment_id}_summary.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    logger.info(f"Experiment report saved to {save_dir}")


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    current_file = Path(__file__).resolve()
    
    # Look for project indicators
    for parent in current_file.parents:
        if (parent / 'requirements.txt').exists() or (parent / 'setup.py').exists():
            return parent
    
    # Fallback to current directory
    return Path.cwd()


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


if __name__ == "__main__":
    # Example usage of utility functions
    
    # Setup logging
    setup_logging("INFO", "logs/utils_test.log")
    
    # Test video info
    print("Testing utility functions...")
    
    # Create sample data
    sample_engagement = np.random.beta(2, 5, 1000)  # Beta distribution for engagement
    sample_features = np.random.randn(50)
    feature_names = [f"feature_{i}" for i in range(50)]
    
    # Test plotting functions
    plot_engagement_distribution(sample_engagement, save_path="test_engagement_dist.png")
    plot_feature_importance(feature_names, np.abs(sample_features), save_path="test_feature_importance.png")
    
    print("Utility functions tested successfully!")