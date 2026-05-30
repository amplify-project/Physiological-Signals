"""
Multi-Person Engagement Estimation using YOLO + MediaPipe + Transformer
Publishes mean engagement score across all detected people to Redis

Usage:
    python scripts/multiperson_engagement.py --model models/engagement_mediapipe_local/fold_1_best_model.pth
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
import sys
import time
from pathlib import Path
from collections import deque
from ultralytics import YOLO
import mediapipe as mp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dwpose_engagement.features import PoseFeatureExtractor


# Temporal Transformer Model Definition (same as training)
class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformerModel(nn.Module):
    """Temporal Transformer for engagement prediction"""
    
    def __init__(self, input_size=1629, d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.3, max_frames=300):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_frames, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 4)  # 4 engagement dimensions
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global average pooling over time dimension
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Output layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Output in [0, 1] range
        
        return x

from src.dwpose_engagement.redis_publisher import EngagementPublisher


class MultiPersonEngagementEstimator:
    """Multi-person engagement estimation with YOLO + MediaPipe + Transformer."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize multi-person engagement estimator.
        
        Args:
            model_path: Path to trained transformer checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO26 for person detection (NMS-free, edge-optimized)
        print("Loading YOLO26 for person detection...")
        self.yolo = YOLO('yolo26n.pt')  # Nano model for speed
        
        # Initialize MediaPipe Holistic
        print("Loading MediaPipe Holistic...")
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load trained transformer model
        print(f"Loading trained model from {model_path}")
        self.model = TemporalTransformerModel(
            input_size=543 * 3,  # 543 keypoints * 3 coordinates
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.3,
            max_frames=300
        )
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)
        print(f"✓ Model loaded successfully")
        
        # Initialize Redis publisher
        self.publisher = EngagementPublisher(channel='amplify.engagement')
        
        # Frame buffer for temporal tracking
        self.max_frames = 300  # Keep 300 frames (~10 sec at 30fps)
        self.min_frames = 30   # Need 30 frames minimum for prediction
        self.pose_buffer = deque(maxlen=self.max_frames)  # Store poses from all people
        
        # Initialize rule-based feature extractor
        self.feature_extractor = PoseFeatureExtractor(
            frame_rate=30.0,
            window_size=30,  # 1 second window
            overlap_ratio=0.0
        )
        
        # Buffer for storing pose sequences for rule-based features
        self.pose_sequence_buffer = deque(maxlen=30)  # Last 1 second
        
        print("✓ Multi-person engagement estimator initialized")
        print("✓ Rule-based feature extractor initialized")
    
    def detect_people(self, frame):
        """
        Detect people in frame using YOLO.
        
        Args:
            frame: RGB image
            
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        results = self.yolo(frame, classes=[0], verbose=False)  # class 0 = person
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return []
        
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            if conf > 0.5:  # Confidence threshold
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        return boxes
    
    def extract_mediapipe_pose(self, frame, bbox=None):
        """
        Extract MediaPipe pose from frame or bounding box region.
        
        Args:
            frame: RGB image
            bbox: Optional (x1, y1, x2, y2) to crop person region
            
        Returns:
            Tuple of (keypoints array (543, 3), landmarks for visualization) or (None, None) if no person detected
        """
        # Crop to bounding box if provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            person_roi = frame[y1:y2, x1:x2]
        else:
            person_roi = frame
            x1, y1 = 0, 0
        
        # Convert to RGB for MediaPipe
        rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.holistic.process(rgb_roi)
        
        if not results.pose_landmarks:
            return None, None
        
        # Extract all keypoints (33 body + 468 face + 21 left hand + 21 right hand = 543)
        keypoints = []
        
        # Body landmarks (33)
        for lm in results.pose_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.visibility])
        
        # Face landmarks (468)
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.visibility])
        else:
            keypoints.extend([[0, 0, 0]] * 468)
        
        # Left hand landmarks (21)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.visibility])
        else:
            keypoints.extend([[0, 0, 0]] * 21)
        
        # Right hand landmarks (21)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.visibility])
        else:
            keypoints.extend([[0, 0, 0]] * 21)
        
        # Return keypoints and landmarks for drawing
        return np.array(keypoints, dtype=np.float32), (results, x1, y1, x2-x1, y2-y1)
    
    def calculate_rule_based_features(self):
        """
        Calculate rule-based features from pose sequence buffer.
        
        Returns:
            Dictionary with motion/pose features AND rule-based engagement estimates, or None if insufficient data
        """
        if len(self.pose_sequence_buffer) < 2:
            return None
        
        # Convert buffer to numpy array (T, N, K, 3) format expected by feature extractor
        # We have single person tracking for now, so N=1
        pose_array = np.array(list(self.pose_sequence_buffer))  # (T, 543, 3)
        pose_array = pose_array[:, np.newaxis, :, :]  # (T, 1, 543, 3)
        
        # Extract motion features
        motion_features = self.feature_extractor.extract_motion_features(pose_array)
        
        # Extract pose features from latest frame
        latest_pose = pose_array[-1]  # (1, 543, 3)
        pose_features = self.feature_extractor.extract_pose_features(latest_pose)
        
        # Calculate low-level metrics
        hand_velocity = float(np.mean([
            motion_features['avg_velocity'][85:106].mean(),  # Left hand
            motion_features['avg_velocity'][106:127].mean()  # Right hand
        ]))
        body_motion = float(motion_features['motion_energy'][:33].mean())  # Body keypoints
        head_motion = float(motion_features['avg_velocity'][:5].mean())  # Head area
        pose_openness = float(pose_features['pose_spread'][0]) if len(pose_features['pose_spread']) > 0 else 0.0
        
        # RULE-BASED ENGAGEMENT HEURISTICS (scaled 0-1)
        # These are simple rules, not ML-based
        
        # Engagement: High hand velocity + body motion = engaged
        # Normalize by typical values: velocity ~0.01-0.05, motion energy ~0.0001-0.001
        rule_engagement = min(1.0, (hand_velocity * 20 + body_motion * 1000) / 2)
        
        # Boredom: Low motion across all body parts = bored
        total_motion = hand_velocity + body_motion + head_motion
        rule_boredom = max(0.0, 1.0 - total_motion * 300)
        
        # Confusion: Erratic head motion = confused (high variance)
        rule_confusion = min(1.0, head_motion * 50)  # High head movement = looking around
        
        # Frustration: High body motion but low hand motion = restless/frustrated
        rule_frustration = max(0.0, min(1.0, (body_motion * 1000 - hand_velocity * 10)))
        
        features = {
            # Raw features
            'hand_velocity': hand_velocity,
            'body_motion': body_motion,
            'head_motion': head_motion,
            'pose_openness': pose_openness,
            
            # Rule-based estimates
            'rule_engagement': rule_engagement,
            'rule_boredom': rule_boredom,
            'rule_confusion': rule_confusion,
            'rule_frustration': rule_frustration,
        }
        
        return features
    
    def predict_engagement(self, pose_buffer):
        """
        Predict engagement from pose buffer.
        
        Args:
            pose_buffer: List or deque of pose keypoints over time, each (543*3,)
            
        Returns:
            Dictionary with engagement predictions scaled to [0, 1]
        """
        # Need at least min_frames for prediction
        if len(pose_buffer) < self.min_frames:
            return None
        
        # Convert buffer to numpy array
        # Take up to max_frames most recent poses
        recent_poses = list(pose_buffer)[-self.max_frames:]
        recent_poses = np.array(recent_poses)  # Shape: (num_frames, 543*3)
        
        # Pad if needed
        if len(recent_poses) < self.max_frames:
            padding = np.zeros((self.max_frames - len(recent_poses), 543 * 3))
            recent_poses = np.vstack([padding, recent_poses])  # Shape: (300, 543*3)
        
        # Run inference
        with torch.no_grad():
            features = torch.from_numpy(recent_poses).unsqueeze(0).float().to(self.device)
            predictions = self.model(features)  # Output already in [0, 1] from sigmoid
            predictions = predictions.cpu().numpy()[0]
        
        # Debug: Print input statistics and predictions
        if len(pose_buffer) % 60 == 0:  # Every 60 frames (~2 seconds)
            print(f"\n[DEBUG] Input stats - Mean: {recent_poses.mean():.4f}, Std: {recent_poses.std():.4f}, Min: {recent_poses.min():.4f}, Max: {recent_poses.max():.4f}")
            print(f"[DEBUG] Predictions - Boredom: {predictions[0]:.3f}, Engagement: {predictions[1]:.3f}, Confusion: {predictions[2]:.3f}, Frustration: {predictions[3]:.3f}")
        
        # Model outputs: [boredom, engagement, confusion, frustration]
        return {
            'boredom': float(predictions[0]),
            'engagement': float(predictions[1]),
            'confusion': float(predictions[2]),
            'frustration': float(predictions[3]),
            'confidence': 1.0  # Model confidence is implicit in the sigmoid output
        }
    
    def run(self):
        """Run multi-person engagement estimation from webcam."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\n" + "="*70)
        print("Multi-Person Engagement Estimation")
        print("="*70)
        print("Press 'q' to quit")
        print("="*70 + "\n")
        
        frame_count = 0
        last_prediction_time = time.time()
        prediction_interval = 1.0  # Predict every 1 second
        
        # Initialize mean values
        mean_engagement = 0.0
        mean_boredom = 0.0
        mean_confusion = 0.0
        mean_frustration = 0.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Detect people
                bboxes = self.detect_people(frame)
                
                # Process each detected person
                person_predictions = []
                from mediapipe.python.solutions import drawing_utils as mp_drawing
                from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
                
                for idx, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox
                    
                    # Extract pose
                    pose, landmarks_data = self.extract_mediapipe_pose(frame, bbox)
                    
                    if pose is not None and landmarks_data is not None:
                        # Add pose to buffer (flatten to 1D)
                        self.pose_buffer.append(pose.flatten())
                        
                        # Add pose to sequence buffer for rule-based features (keep 3D shape)
                        self.pose_sequence_buffer.append(pose)
                        
                        # Draw skeleton overlay
                        results, bbox_x, bbox_y, bbox_w, bbox_h = landmarks_data
                        
                        # Create a copy of the person region
                        person_roi = frame[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w].copy()
                        
                        # Draw pose landmarks on the person region
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                person_roi,
                                results.pose_landmarks,
                                self.mp_holistic.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                            )
                        
                        # Draw hand landmarks
                        if results.left_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                person_roi,
                                results.left_hand_landmarks,
                                self.mp_holistic.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                            )
                        
                        if results.right_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                person_roi,
                                results.right_hand_landmarks,
                                self.mp_holistic.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                            )
                        
                        # Put the annotated region back
                        frame[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w] = person_roi
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {idx+1}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Run prediction every second using accumulated buffer
                current_time = time.time()
                if (current_time - last_prediction_time) >= prediction_interval:
                    # Only predict if people were detected in this frame
                    if len(bboxes) > 0:
                        pred = self.predict_engagement(self.pose_buffer)
                        if pred is not None:
                            person_predictions = [pred]  # Single prediction from buffer
                    else:
                        # No people detected - reset predictions to zero
                        person_predictions = [{
                            'engagement': 0.0,
                            'boredom': 0.0,
                            'confusion': 0.0,
                            'frustration': 0.0,
                            'confidence': 0.0
                        }]
                    last_prediction_time = current_time
                
                # Update mean scores when predictions are available
                if person_predictions:
                    mean_engagement = person_predictions[0]['engagement']
                    mean_boredom = person_predictions[0]['boredom']
                    mean_confusion = person_predictions[0]['confusion']
                    mean_frustration = person_predictions[0]['frustration']
                    mean_confidence = person_predictions[0]['confidence']
                    
                    # Publish to Redis
                    self.publisher.publish_engagement(
                        engagement=float(mean_engagement),
                        boredom=float(mean_boredom),
                        confusion=float(mean_confusion),
                        frustration=float(mean_frustration),
                        score=float(mean_confidence),
                        video_id=f"webcam_live_buffer_{len(self.pose_buffer)}_frames"
                    )
                
                # Get frame dimensions for positioning
                frame_height, frame_width = frame.shape[:2]
                
                # MAIN PANEL: AI Predictions Only
                overlay = frame.copy()
                cv2.rectangle(overlay, (5, 5), (180, 110), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.rectangle(frame, (5, 5), (180, 110), (0, 255, 0), 2)
                
                # Display AI metrics (font size: 0.35)
                cv2.putText(frame, f"People: {len(bboxes)}", (10, 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                cv2.putText(frame, f"Engagement: {mean_engagement:.3f}", (10, 42),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                cv2.putText(frame, f"Boredom: {mean_boredom:.3f}", (10, 62),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)
                cv2.putText(frame, f"Confusion: {mean_confusion:.3f}", (10, 82),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 150, 0), 1)
                cv2.putText(frame, f"Frustration: {mean_frustration:.3f}", (10, 102),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 150, 255), 1)
                
                # # COMMENTED OUT: Composite Metrics
                # # Calculate composite metrics
                # flow_state = mean_engagement * (1 - mean_confusion) * (1 - mean_frustration)
                # attention_quality = mean_engagement * (1 - mean_confusion)
                # energy_level = (mean_engagement + mean_frustration - mean_boredom) / 2.0
                # energy_level = max(0.0, min(1.0, energy_level))  # Clamp to [0, 1]
                # 
                # overlay = frame.copy()
                # panel_x = frame_width - 185
                # cv2.rectangle(overlay, (panel_x, 5), (frame_width - 5, 90), (0, 0, 0), -1)
                # cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                # cv2.rectangle(frame, (panel_x, 5), (frame_width - 5, 90), (255, 0, 255), 2)
                # 
                # # Display composite metrics (font size: 0.35)
                # cv2.putText(frame, "COMPOSITE", (panel_x + 5, 22),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
                # cv2.putText(frame, f"Flow: {flow_state:.3f}", (panel_x + 5, 46),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 255), 1)
                # cv2.putText(frame, f"Attention: {attention_quality:.3f}", (panel_x + 5, 66),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 100, 255), 1)
                # cv2.putText(frame, f"Energy: {energy_level:.3f}", (panel_x + 5, 86),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 50, 255), 1)
                
                # # COMMENTED OUT: Rule-Based Motion Features
                # rule_features = self.calculate_rule_based_features()
                # if rule_features:
                #     overlay = frame.copy()
                #     cv2.rectangle(overlay, (5, frame_height - 95), (200, frame_height - 5), (0, 0, 0), -1)
                #     cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                #     cv2.rectangle(frame, (5, frame_height - 95), (200, frame_height - 5), (0, 255, 255), 2)
                #     
                #     # Display raw motion features
                #     cv2.putText(frame, "MOTION", (10, frame_height - 78),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                #     cv2.putText(frame, f"Hand Vel: {rule_features['hand_velocity']:.4f}", (10, frame_height - 58),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 255), 1)
                #     cv2.putText(frame, f"Body: {rule_features['body_motion']:.4f}", (10, frame_height - 38),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 200), 1)
                #     cv2.putText(frame, f"Head: {rule_features['head_motion']:.4f}", (10, frame_height - 18),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 150), 1)
                #     
                #     # BOTTOM-RIGHT PANEL: Rule-Based Engagement Estimates
                #     overlay = frame.copy()
                #     panel_x = frame_width - 185
                #     cv2.rectangle(overlay, (panel_x, frame_height - 115), (frame_width - 5, frame_height - 5), (0, 0, 0), -1)
                #     cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                #     cv2.rectangle(frame, (panel_x, frame_height - 115), (frame_width - 5, frame_height - 5), (255, 165, 0), 2)
                #     
                #     # Display rule-based estimates
                #     cv2.putText(frame, "RULE-BASED", (panel_x + 5, frame_height - 98),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 165, 0), 1)
                #     cv2.putText(frame, f"Engage: {rule_features['rule_engagement']:.3f}", (panel_x + 5, frame_height - 74),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                #     cv2.putText(frame, f"Boredom: {rule_features['rule_boredom']:.3f}", (panel_x + 5, frame_height - 50),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)
                #     cv2.putText(frame, f"Confuse: {rule_features['rule_confusion']:.3f}", (panel_x + 5, frame_height - 26),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 150, 0), 1)
                
                # Display frame
                cv2.imshow('Multi-Person Engagement', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.publisher.close()
            print("\n✓ Stopped multi-person engagement estimation")


def main():
    parser = argparse.ArgumentParser(description='Multi-person engagement estimation')
    parser.add_argument('--model', type=str, 
                       default='models/engagement_mediapipe_local/fold_1_best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Warning: Model not found at {args.model}")
        print("Using placeholder predictions for testing")
    
    # Create and run estimator
    estimator = MultiPersonEngagementEstimator(args.model, args.device)
    estimator.run()


if __name__ == '__main__':
    main()
