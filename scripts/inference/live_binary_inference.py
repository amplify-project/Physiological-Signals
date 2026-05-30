import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
import time
import sys

# =============================================================================
# MODEL DEFINITION (Copied from train_action_transformer_ddp_v2.py)
# =============================================================================

class TemporalTransformer(nn.Module):
    """Temporal Transformer for action recognition."""
    
    def __init__(self, input_dim=543*3, num_classes=2, d_model=256, nhead=8, 
                 num_layers=4, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

# =============================================================================
# INFERENCE CLASS
# =============================================================================

class BinaryEngagementEstimator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 1. Load YOLO26 for person detection (NMS-free, edge-optimized)
        print("Loading YOLO26...")
        self.yolo = YOLO('yolo26n.pt')
        
        # 2. Initialize MediaPipe Holistic
        print("Initializing MediaPipe...")
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 3. Load Engagement Model
        print(f"Loading Engagement Model from {model_path}...")
        self.model = TemporalTransformer(num_classes=2).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle DDP state dict (remove 'module.' prefix)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
        # Buffer for temporal sequence
        self.sequence_length = 64
        self.feature_buffer = deque(maxlen=self.sequence_length)
        
        # Classes
        self.classes = ['Disengaged', 'Engaged']
        
    def extract_features(self, frame_rgb, bbox):
        """Extract 543x3 keypoints from person crop."""
        x1, y1, x2, y2 = bbox
        
        # Ensure crop is within bounds
        h, w, _ = frame_rgb.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros((543, 3))
            
        person_crop = frame_rgb[y1:y2, x1:x2]
        
        # MediaPipe inference
        results = self.holistic.process(person_crop)
        
        # Extract keypoints
        keypoints = np.zeros((543, 3))
        
        # Helper to extract landmarks
        def get_landmarks(landmarks, offset):
            if landmarks:
                for i, lm in enumerate(landmarks.landmark):
                    keypoints[offset + i] = [lm.x, lm.y, lm.visibility if hasattr(lm, 'visibility') else 1.0]
        
        # 1. Pose (33)
        get_landmarks(results.pose_landmarks, 0)
        # 2. Face (468)
        get_landmarks(results.face_landmarks, 33)
        # 3. Left Hand (21)
        get_landmarks(results.left_hand_landmarks, 33 + 468)
        # 4. Right Hand (21)
        get_landmarks(results.right_hand_landmarks, 33 + 468 + 21)
        
        return keypoints

    def process_frame(self, frame):
        """Process a single frame and return engagement score."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Detect Person (YOLO)
        results = self.yolo(frame_rgb, verbose=False, classes=[0]) # class 0 is person
        
        largest_person = None
        max_area = 0
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_person = (x1, y1, x2, y2)
        
        engagement_score = 0.0
        is_engaged = False
        
        if largest_person:
            # 2. Extract Features
            features = self.extract_features(frame_rgb, largest_person)
            
            # Flatten features (543, 3) -> (1629,)
            features_flat = features.flatten()
            self.feature_buffer.append(features_flat)
            
            # 3. Inference (if buffer full)
            if len(self.feature_buffer) == self.sequence_length:
                # Prepare tensor (1, 64, 1629)
                input_tensor = torch.tensor(np.array(self.feature_buffer), dtype=torch.float32)
                input_tensor = input_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits = self.model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    
                    # Assuming class 1 is 'Engaged' and class 0 is 'Disengaged'
                    # Check idx_to_action in checkpoint if unsure, but usually alphabetical or 0/1
                    # For binary, usually 0=Negative, 1=Positive
                    engagement_score = probs[0][1].item()
                    is_engaged = engagement_score > 0.5
            
            return largest_person, engagement_score, is_engaged
            
        else:
            # No person detected, clear buffer to avoid mixing people
            if len(self.feature_buffer) > 0:
                self.feature_buffer.clear()
            return None, 0.0, False

# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    model_path = 'models/action_transformer_12gpus_binary_v2_cleaned/best_model.pth'
    
    try:
        estimator = BinaryEngagementEstimator(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        return
    except Exception as e:
        print(f"Error initializing estimator: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting inference... Press 'q' to quit.")
    
    fps_time = time.time()
    frame_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process
        bbox, score, is_engaged = estimator.process_frame(frame)
        
        # Visualization
        if bbox:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if is_engaged else (0, 0, 255)
            
            # Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Progress Bar
            bar_x = x1
            bar_y = y1 - 45  # Moved up
            bar_w = x2 - x1
            bar_h = 15       # Slightly thinner
            
            # Ensure bar doesn't go off-screen
            if bar_y < 0:
                bar_y = y2 + 10  # Move to bottom if too close to top
            
            # Background for Bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            # Fill Bar
            fill_w = int(bar_w * score)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
            
            # Label with background
            label = f"Engaged: {score:.1%}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            text_x = x1
            text_y = bar_y - 10 # Above the bar
            
            # Ensure text doesn't go off-screen
            if text_y - text_h < 0:
                 text_y = bar_y + bar_h + 25 # Below the bar if bar is at top
            
            # Text Background
            cv2.rectangle(frame, (text_x, text_y - text_h - 5), (text_x + text_w + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # FPS
        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count / (time.time() - fps_time)
            frame_count = 0
            fps_time = time.time()
            
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Buffer Status
        buffer_len = len(estimator.feature_buffer)
        cv2.putText(frame, f"Buffer: {buffer_len}/{estimator.sequence_length}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow('Live Binary Engagement', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
