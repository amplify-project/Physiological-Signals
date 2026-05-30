"""
Feature engineering module for extracting engagement signals from pose dynamics.

This module transforms raw pose keypoints into meaningful features that can
be used to predict audience engagement levels.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import cv2

logger = logging.getLogger(__name__)


@dataclass
class EngagementFeatures:
    """Container for extracted engagement features."""
    temporal_features: np.ndarray
    spatial_features: np.ndarray
    group_features: np.ndarray
    statistical_features: np.ndarray
    feature_names: List[str]
    timestamp: float
    num_people: int


class PoseFeatureExtractor:
    """
    Extract engagement-related features from pose keypoints.
    
    This class converts raw DWPose keypoints into meaningful features
    that correlate with audience engagement levels.
    """
    
    def __init__(
        self,
        frame_rate: float = 30.0,
        window_size: int = 90,  # 3 seconds at 30fps
        overlap_ratio: float = 0.5
    ):
        """
        Initialize the feature extractor.
        
        Args:
            frame_rate: Video frame rate
            window_size: Number of frames for temporal feature extraction
            overlap_ratio: Overlap between sliding windows
        """
        self.frame_rate = frame_rate
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.step_size = int(window_size * (1 - overlap_ratio))
        
        # DWPose keypoint indices (133 total keypoints)
        self._define_keypoint_groups()
    
    def _define_keypoint_groups(self):
        """Define semantic groups of keypoints for feature extraction."""
        # Face keypoints (68 points)
        self.face_indices = list(range(0, 68))
        
        # Body keypoints (17 points)
        self.body_indices = list(range(68, 85))
        
        # Hand keypoints (21 points each)
        self.left_hand_indices = list(range(85, 106))
        self.right_hand_indices = list(range(106, 127))
        
        # Foot keypoints (6 points)
        self.foot_indices = list(range(127, 133))
        
        # Key body parts for engagement analysis
        self.head_indices = [68, 69, 70, 71, 72]  # Head keypoints
        self.torso_indices = [68, 74, 79, 84]     # Shoulders and hips
        self.arm_indices = [74, 75, 76, 77, 78, 79, 80, 81, 82, 83]  # Arms
    
    def extract_motion_features(self, keypoints_sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract motion-based features from pose sequences.
        
        Args:
            keypoints_sequence: [T, N, K, 3] where T=time, N=people, K=keypoints
        
        Returns:
            Dictionary of motion features
        """
        features = {}
        T, N, K, _ = keypoints_sequence.shape
        
        if T < 2:
            return self._empty_motion_features()
        
        # Calculate velocities and accelerations
        velocities = np.diff(keypoints_sequence[:, :, :, :2], axis=0)  # [T-1, N, K, 2]
        accelerations = np.diff(velocities, axis=0)  # [T-2, N, K, 2]
        
        # Motion magnitude
        velocity_mag = np.linalg.norm(velocities, axis=-1)  # [T-1, N, K]
        acceleration_mag = np.linalg.norm(accelerations, axis=-1)  # [T-2, N, K]
        
        # Average motion across people and time
        features['avg_velocity'] = np.mean(velocity_mag, axis=(0, 1))  # [K]
        features['avg_acceleration'] = np.mean(acceleration_mag, axis=(0, 1))  # [K]
        
        # Motion energy (sum of squared velocities)
        features['motion_energy'] = np.sum(velocity_mag**2, axis=(0, 1))  # [K]
        
        # Movement consistency (low variance indicates synchronized movement)
        features['velocity_std'] = np.std(velocity_mag, axis=0).mean(axis=0)  # [K]
        
        # Directional features
        features['horizontal_motion'] = np.mean(np.abs(velocities[:, :, :, 0]), axis=(0, 1))
        features['vertical_motion'] = np.mean(np.abs(velocities[:, :, :, 1]), axis=(0, 1))
        
        return features
    
    def extract_pose_features(self, keypoints: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract pose-based features from a single frame.
        
        Args:
            keypoints: [N, K, 3] where N=people, K=keypoints
        
        Returns:
            Dictionary of pose features
        """
        features = {}
        N, K, _ = keypoints.shape
        
        if N == 0:
            return self._empty_pose_features()
        
        # Extract coordinates and confidences
        coords = keypoints[:, :, :2]  # [N, K, 2]
        confidences = keypoints[:, :, 2]  # [N, K]
        
        # Average confidence (pose detection quality)
        features['avg_confidence'] = np.mean(confidences, axis=1)  # [N]
        
        # Pose spread/openness (distance between extremities)
        features['pose_spread'] = self._calculate_pose_spread(coords)  # [N]
        
        # Head orientation (approximate)
        features['head_orientation'] = self._calculate_head_orientation(coords)  # [N]
        
        # Arm positions (raised arms often indicate engagement)
        features['arm_elevation'] = self._calculate_arm_elevation(coords)  # [N]
        
        # Body symmetry
        features['body_symmetry'] = self._calculate_body_symmetry(coords)  # [N]
        
        # Pose variability across people
        if N > 1:
            features['pose_diversity'] = self._calculate_pose_diversity(coords)
        else:
            features['pose_diversity'] = np.array([0.0])
        
        return features
    
    def extract_group_features(self, keypoints: np.ndarray) -> Dict[str, float]:
        """
        Extract group-level engagement features.
        
        Args:
            keypoints: [N, K, 3] where N=people, K=keypoints
        
        Returns:
            Dictionary of group features
        """
        features = {}
        N, K, _ = keypoints.shape
        
        if N == 0:
            return self._empty_group_features()
        
        coords = keypoints[:, :, :2]
        
        # Crowd density and distribution
        features['crowd_density'] = self._calculate_crowd_density(coords)
        features['spatial_spread'] = self._calculate_spatial_spread(coords)
        
        # Group synchronization
        features['movement_sync'] = self._calculate_movement_synchronization(coords)
        
        # Attention focus (people looking in similar directions)
        features['attention_coherence'] = self._calculate_attention_coherence(coords)
        
        # Energy level (overall activity)
        features['group_energy'] = self._calculate_group_energy(coords)
        
        return features
    
    def extract_temporal_features(self, keypoints_sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract temporal features from pose sequences.
        
        Args:
            keypoints_sequence: [T, N, K, 3] where T=time, N=people, K=keypoints
        
        Returns:
            Dictionary of temporal features
        """
        features = {}
        T, N, K, _ = keypoints_sequence.shape
        
        if T < self.window_size:
            return self._empty_temporal_features()
        
        # Extract motion features over time
        motion_features = self.extract_motion_features(keypoints_sequence)
        
        # Rhythm and periodicity analysis
        features['movement_rhythm'] = self._analyze_movement_rhythm(keypoints_sequence)
        
        # Temporal consistency
        features['temporal_consistency'] = self._calculate_temporal_consistency(keypoints_sequence)
        
        # Engagement peaks (moments of high activity)
        features['engagement_peaks'] = self._detect_engagement_peaks(keypoints_sequence)
        
        # Trend analysis (increasing/decreasing engagement)
        features['engagement_trend'] = self._calculate_engagement_trend(keypoints_sequence)
        
        return features
    
    def _calculate_pose_spread(self, coords: np.ndarray) -> np.ndarray:
        """Calculate pose spread/openness for each person."""
        spreads = []
        for person_coords in coords:
            # Calculate bounding box dimensions
            valid_points = person_coords[~np.isnan(person_coords).any(axis=1)]
            if len(valid_points) > 0:
                x_range = np.max(valid_points[:, 0]) - np.min(valid_points[:, 0])
                y_range = np.max(valid_points[:, 1]) - np.min(valid_points[:, 1])
                spread = np.sqrt(x_range**2 + y_range**2)
            else:
                spread = 0.0
            spreads.append(spread)
        return np.array(spreads)
    
    def _calculate_head_orientation(self, coords: np.ndarray) -> np.ndarray:
        """Estimate head orientation for each person."""
        orientations = []
        for person_coords in coords:
            # Use face keypoints to estimate head orientation
            if len(self.face_indices) > 0:
                face_coords = person_coords[self.face_indices]
                valid_face = face_coords[~np.isnan(face_coords).any(axis=1)]
                if len(valid_face) > 10:  # Need enough face points
                    # Simple approximation: angle of face bounding box
                    x_range = np.max(valid_face[:, 0]) - np.min(valid_face[:, 0])
                    y_range = np.max(valid_face[:, 1]) - np.min(valid_face[:, 1])
                    orientation = np.arctan2(y_range, x_range)
                else:
                    orientation = 0.0
            else:
                orientation = 0.0
            orientations.append(orientation)
        return np.array(orientations)
    
    def _calculate_arm_elevation(self, coords: np.ndarray) -> np.ndarray:
        """Calculate arm elevation for each person."""
        elevations = []
        for person_coords in coords:
            # Compare shoulder and wrist heights
            if len(self.arm_indices) > 0:
                arm_coords = person_coords[self.arm_indices]
                valid_arms = arm_coords[~np.isnan(arm_coords).any(axis=1)]
                if len(valid_arms) > 4:  # Need shoulder and wrist points
                    # Simple approximation: average y-coordinate of arms
                    avg_arm_height = np.mean(valid_arms[:, 1])
                    # Normalize by body height
                    body_coords = person_coords[self.body_indices]
                    valid_body = body_coords[~np.isnan(body_coords).any(axis=1)]
                    if len(valid_body) > 0:
                        body_height = np.max(valid_body[:, 1]) - np.min(valid_body[:, 1])
                        elevation = avg_arm_height / (body_height + 1e-6)
                    else:
                        elevation = 0.0
                else:
                    elevation = 0.0
            else:
                elevation = 0.0
            elevations.append(elevation)
        return np.array(elevations)
    
    def _calculate_body_symmetry(self, coords: np.ndarray) -> np.ndarray:
        """Calculate body symmetry for each person."""
        symmetries = []
        for person_coords in coords:
            # Compare left and right body parts
            left_points = person_coords[self.left_hand_indices] if self.left_hand_indices else []
            right_points = person_coords[self.right_hand_indices] if self.right_hand_indices else []
            
            if len(left_points) > 0 and len(right_points) > 0:
                left_valid = left_points[~np.isnan(left_points).any(axis=1)]
                right_valid = right_points[~np.isnan(right_points).any(axis=1)]
                
                if len(left_valid) > 0 and len(right_valid) > 0:
                    # Calculate symmetry as similarity between left and right sides
                    left_center = np.mean(left_valid, axis=0)
                    right_center = np.mean(right_valid, axis=0)
                    symmetry = 1.0 / (1.0 + np.linalg.norm(left_center - right_center))
                else:
                    symmetry = 0.5
            else:
                symmetry = 0.5
            symmetries.append(symmetry)
        return np.array(symmetries)
    
    def _calculate_pose_diversity(self, coords: np.ndarray) -> np.ndarray:
        """Calculate diversity of poses in the group."""
        N, K, _ = coords.shape
        if N < 2:
            return np.array([0.0])
        
        # Flatten poses for each person
        poses_flat = coords.reshape(N, -1)
        
        # Calculate pairwise distances
        distances = pdist(poses_flat, metric='euclidean')
        diversity = np.mean(distances)
        
        return np.array([diversity])
    
    def _calculate_crowd_density(self, coords: np.ndarray) -> float:
        """Calculate crowd density."""
        N, K, _ = coords.shape
        if N == 0:
            return 0.0
        
        # Calculate average distance between people
        person_centers = np.nanmean(coords, axis=1)  # [N, 2]
        valid_centers = person_centers[~np.isnan(person_centers).any(axis=1)]
        
        if len(valid_centers) < 2:
            return 0.0
        
        distances = pdist(valid_centers)
        avg_distance = np.mean(distances)
        
        # Density is inverse of average distance
        density = 1.0 / (avg_distance + 1e-6)
        return density
    
    def _calculate_spatial_spread(self, coords: np.ndarray) -> float:
        """Calculate spatial spread of the crowd."""
        N, K, _ = coords.shape
        if N == 0:
            return 0.0
        
        person_centers = np.nanmean(coords, axis=1)
        valid_centers = person_centers[~np.isnan(person_centers).any(axis=1)]
        
        if len(valid_centers) == 0:
            return 0.0
        
        # Calculate bounding box of crowd
        x_range = np.max(valid_centers[:, 0]) - np.min(valid_centers[:, 0])
        y_range = np.max(valid_centers[:, 1]) - np.min(valid_centers[:, 1])
        spread = np.sqrt(x_range**2 + y_range**2)
        
        return spread
    
    def _calculate_movement_synchronization(self, coords: np.ndarray) -> float:
        """Calculate movement synchronization (placeholder)."""
        # This would require temporal information
        # For now, return a default value
        return 0.5
    
    def _calculate_attention_coherence(self, coords: np.ndarray) -> float:
        """Calculate attention coherence (placeholder)."""
        # This would require head orientation analysis
        return 0.5
    
    def _calculate_group_energy(self, coords: np.ndarray) -> float:
        """Calculate overall group energy level."""
        N, K, _ = coords.shape
        if N == 0:
            return 0.0
        
        # Use pose spread as a proxy for energy
        spreads = self._calculate_pose_spread(coords)
        energy = np.mean(spreads)
        
        return energy
    
    def _analyze_movement_rhythm(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        """Analyze rhythm in movement patterns."""
        T, N, K, _ = keypoints_sequence.shape
        
        # Calculate motion energy over time
        if T < 2:
            return np.array([0.0])
        
        velocities = np.diff(keypoints_sequence[:, :, :, :2], axis=0)
        motion_energy = np.sum(np.linalg.norm(velocities, axis=-1)**2, axis=(1, 2))
        
        # Analyze periodicity using FFT
        if len(motion_energy) > 10:
            fft = np.fft.fft(motion_energy)
            frequencies = np.fft.fftfreq(len(motion_energy), 1/self.frame_rate)
            
            # Find dominant frequency
            power_spectrum = np.abs(fft)
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_freq = frequencies[dominant_freq_idx]
            
            return np.array([dominant_freq])
        else:
            return np.array([0.0])
    
    def _calculate_temporal_consistency(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        """Calculate temporal consistency of poses."""
        T, N, K, _ = keypoints_sequence.shape
        
        if T < 2:
            return np.array([1.0])
        
        # Calculate frame-to-frame differences
        diffs = np.diff(keypoints_sequence, axis=0)
        diff_magnitudes = np.linalg.norm(diffs[:, :, :, :2], axis=-1)
        
        # Consistency is inverse of variation
        variation = np.std(diff_magnitudes, axis=0).mean()
        consistency = 1.0 / (1.0 + variation)
        
        return np.array([consistency])
    
    def _detect_engagement_peaks(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        """Detect peaks in engagement activity."""
        T, N, K, _ = keypoints_sequence.shape
        
        if T < 2:
            return np.array([0.0])
        
        # Calculate motion energy over time
        velocities = np.diff(keypoints_sequence[:, :, :, :2], axis=0)
        motion_energy = np.sum(np.linalg.norm(velocities, axis=-1)**2, axis=(1, 2))
        
        # Detect peaks
        if len(motion_energy) > 5:
            peaks, _ = signal.find_peaks(motion_energy, height=np.mean(motion_energy))
            peak_density = len(peaks) / len(motion_energy)
        else:
            peak_density = 0.0
        
        return np.array([peak_density])
    
    def _calculate_engagement_trend(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        """Calculate trend in engagement over time."""
        T, N, K, _ = keypoints_sequence.shape
        
        if T < 2:
            return np.array([0.0])
        
        # Calculate motion energy over time
        velocities = np.diff(keypoints_sequence[:, :, :, :2], axis=0)
        motion_energy = np.sum(np.linalg.norm(velocities, axis=-1)**2, axis=(1, 2))
        
        # Calculate linear trend
        if len(motion_energy) > 1:
            x = np.arange(len(motion_energy))
            trend = np.polyfit(x, motion_energy, 1)[0]  # Slope of linear fit
        else:
            trend = 0.0
        
        return np.array([trend])
    
    def _empty_motion_features(self) -> Dict[str, np.ndarray]:
        """Return empty motion features."""
        return {
            'avg_velocity': np.zeros(133),
            'avg_acceleration': np.zeros(133),
            'motion_energy': np.zeros(133),
            'velocity_std': np.zeros(133),
            'horizontal_motion': np.zeros(133),
            'vertical_motion': np.zeros(133)
        }
    
    def _empty_pose_features(self) -> Dict[str, np.ndarray]:
        """Return empty pose features."""
        return {
            'avg_confidence': np.array([0.0]),
            'pose_spread': np.array([0.0]),
            'head_orientation': np.array([0.0]),
            'arm_elevation': np.array([0.0]),
            'body_symmetry': np.array([0.5]),
            'pose_diversity': np.array([0.0])
        }
    
    def _empty_group_features(self) -> Dict[str, float]:
        """Return empty group features."""
        return {
            'crowd_density': 0.0,
            'spatial_spread': 0.0,
            'movement_sync': 0.5,
            'attention_coherence': 0.5,
            'group_energy': 0.0
        }
    
    def _empty_temporal_features(self) -> Dict[str, np.ndarray]:
        """Return empty temporal features."""
        return {
            'movement_rhythm': np.array([0.0]),
            'temporal_consistency': np.array([1.0]),
            'engagement_peaks': np.array([0.0]),
            'engagement_trend': np.array([0.0])
        }
    
    def extract_all_features(self, pose_sequence: List) -> List[EngagementFeatures]:
        """
        Extract all engagement features from a pose sequence.
        
        Args:
            pose_sequence: List of PoseKeypoints objects
        
        Returns:
            List of EngagementFeatures for each time window
        """
        if len(pose_sequence) < self.window_size:
            logger.warning(f"Sequence too short ({len(pose_sequence)} < {self.window_size})")
            return []
        
        features_list = []
        
        # Convert to numpy array
        T = len(pose_sequence)
        max_people = max(len(pose.keypoints) for pose in pose_sequence)
        K = 133  # DWPose keypoints
        
        keypoints_array = np.full((T, max_people, K, 3), np.nan)
        for t, pose in enumerate(pose_sequence):
            n_people = len(pose.keypoints)
            if n_people > 0:
                keypoints_array[t, :n_people] = pose.keypoints
        
        # Extract features using sliding windows
        for start_idx in range(0, T - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            window_data = keypoints_array[start_idx:end_idx]
            
            # Extract different types of features
            motion_features = self.extract_motion_features(window_data)
            temporal_features = self.extract_temporal_features(window_data)
            
            # Use middle frame for pose and group features
            mid_idx = start_idx + self.window_size // 2
            mid_frame = keypoints_array[mid_idx]
            
            pose_features = self.extract_pose_features(mid_frame)
            group_features = self.extract_group_features(mid_frame)
            
            # Combine all features
            all_features = {}
            all_features.update(motion_features)
            all_features.update(temporal_features)
            all_features.update(pose_features)
            all_features.update(group_features)
            
            # Flatten and concatenate features
            feature_vector = []
            feature_names = []
            
            for key, value in all_features.items():
                if isinstance(value, np.ndarray):
                    if value.ndim > 1:
                        value = value.flatten()
                    feature_vector.extend(value)
                    if len(value) > 1:
                        feature_names.extend([f"{key}_{i}" for i in range(len(value))])
                    else:
                        feature_names.append(key)
                else:
                    feature_vector.append(value)
                    feature_names.append(key)
            
            # Create EngagementFeatures object
            timestamp = pose_sequence[mid_idx].timestamp
            num_people = np.sum(~np.isnan(mid_frame[:, 0, 0]))
            
            features = EngagementFeatures(
                temporal_features=np.array(feature_vector[:50]),  # First 50 for temporal
                spatial_features=np.array(feature_vector[50:100]),  # Next 50 for spatial
                group_features=np.array(feature_vector[100:110]),   # Next 10 for group
                statistical_features=np.array(feature_vector[110:]), # Rest for statistical
                feature_names=feature_names,
                timestamp=timestamp,
                num_people=int(num_people)
            )
            
            features_list.append(features)
        
        return features_list


if __name__ == "__main__":
    # Example usage
    from .pose_estimation import DWPoseEstimator
    
    # Initialize feature extractor
    feature_extractor = PoseFeatureExtractor(
        frame_rate=30.0,
        window_size=90,  # 3 seconds
        overlap_ratio=0.5
    )
    
    # Load pose sequence (example)
    pose_estimator = DWPoseEstimator()
    pose_sequence = pose_estimator.load_pose_data("data/processed/poses/concert_001_poses.json")
    
    # Extract features
    features = feature_extractor.extract_all_features(pose_sequence)
    
    print(f"Extracted {len(features)} feature windows")
    if features:
        print(f"Feature vector size: {len(features[0].feature_names)}")
        print(f"Sample features: {features[0].feature_names[:10]}")