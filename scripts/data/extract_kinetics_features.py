#!/usr/bin/env python3
"""
Extract MediaPipe Holistic features from Kinetics-700-2020 videos.
Uses YOLO for person detection (GPU) and MediaPipe for pose extraction (CPU).
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import time
import json

# MediaPipe imports
import mediapipe as mp_module
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# YOLO imports
from ultralytics import YOLO

# Paths
DATA_ROOT = Path.home() / "concert_engagement" / "data" / "raw" / "Kinetics700"
OUTPUT_ROOT = Path.home() / "concert_engagement" / "data" / "processed" / "features_kinetics700"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

# Create output directory
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def get_video_files():
    """Get all video files from train and val directories."""
    video_files = []
    
    # Train videos
    if TRAIN_DIR.exists():
        for action_dir in sorted(TRAIN_DIR.iterdir()):
            if action_dir.is_dir():
                for video_file in action_dir.glob("*.mp4"):
                    video_files.append(("train", action_dir.name, video_file))
    
    # Validation videos
    if VAL_DIR.exists():
        for action_dir in sorted(VAL_DIR.iterdir()):
            if action_dir.is_dir():
                for video_file in action_dir.glob("*.mp4"):
                    video_files.append(("val", action_dir.name, video_file))
    
    return video_files

def extract_video_features(video_path, gpu_id):
    """
    Extract features from a single video using YOLO (GPU) + MediaPipe (CPU).
    
    Returns:
        np.ndarray: Shape (num_frames, 543, 3) with keypoints
        None: If extraction failed
    """
    try:
        # CRITICAL: Set CUDA device for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Set GPU for YOLO - use device 0 since we limited visibility above
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Initialize YOLO26 for person detection (NMS-free, edge-optimized)
        yolo = YOLO('yolo26n.pt')
        yolo.to(device)
        
        # Initialize MediaPipe Holistic (runs on CPU)
        mp_holistic = mp_module.solutions.holistic
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        frame_features = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect people with YOLO
            results = yolo(frame_rgb, verbose=False, device=device)
            
            # Find largest person (by bbox area)
            largest_person = None
            max_area = 0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls[0]) == 0:  # Person class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            largest_person = (x1, y1, x2, y2)
            
            # Extract MediaPipe features for largest person
            if largest_person is not None:
                x1, y1, x2, y2 = largest_person
                person_crop = frame_rgb[y1:y2, x1:x2]
                
                # Process with MediaPipe
                results_mp = holistic.process(person_crop)
                
                # Extract keypoints (543 total)
                keypoints = np.zeros((543, 3))  # x, y, confidence
                
                # Body pose (33 landmarks)
                if results_mp.pose_landmarks:
                    for i, lm in enumerate(results_mp.pose_landmarks.landmark):
                        keypoints[i] = [lm.x, lm.y, lm.visibility]
                
                # Face (468 landmarks)
                if results_mp.face_landmarks:
                    for i, lm in enumerate(results_mp.face_landmarks.landmark):
                        keypoints[33 + i] = [lm.x, lm.y, lm.visibility if hasattr(lm, 'visibility') else 1.0]
                
                # Left hand (21 landmarks)
                if results_mp.left_hand_landmarks:
                    for i, lm in enumerate(results_mp.left_hand_landmarks.landmark):
                        keypoints[501 + i] = [lm.x, lm.y, 1.0]
                
                # Right hand (21 landmarks)
                if results_mp.right_hand_landmarks:
                    for i, lm in enumerate(results_mp.right_hand_landmarks.landmark):
                        keypoints[522 + i] = [lm.x, lm.y, 1.0]
                
                frame_features.append(keypoints)
            else:
                # No person detected - append zeros
                frame_features.append(np.zeros((543, 3)))
        
        cap.release()
        holistic.close()
        
        if len(frame_features) == 0:
            return None
        
        return np.array(frame_features)
    
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def process_worker(gpu_id, video_queue, results_queue, failed_list):
    """Worker process for extracting features."""
    # CRITICAL: Set GPU visibility at process start, before any CUDA initialization
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Initialize YOLO26 in this process after setting CUDA_VISIBLE_DEVICES
    yolo = YOLO('yolo26n.pt')
    yolo.to('cuda:0')  # Use cuda:0 since we limited visibility to one GPU
    
    # Initialize MediaPipe Holistic
    mp_holistic = mp_module.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    while True:
        try:
            item = video_queue.get()
            if item is None:  # Poison pill
                break
            
            split, action, video_path = item
            
            # Create output path
            output_dir = OUTPUT_ROOT / split / action
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{video_path.stem}.npz"
            
            # Skip if already processed
            if output_file.exists():
                results_queue.put(('cached', split, action))
                continue
            
            # Extract features using initialized models
            features = extract_video_features_worker(video_path, yolo, holistic)
            
            if features is not None:
                # Save features
                np.savez_compressed(output_file, features=features)
                results_queue.put(('success', split, action))
            else:
                failed_list.append(str(video_path))
                results_queue.put(('failed', split, action))
        
        except Exception as e:
            print(f"Worker {gpu_id} error: {e}")
            failed_list.append(str(video_path) if 'video_path' in locals() else 'unknown')
            results_queue.put(('failed', split if 'split' in locals() else '', action if 'action' in locals() else ''))
    
    holistic.close()

def extract_video_features_worker(video_path, yolo, holistic):
    """
    Extract features using pre-initialized models.
    This avoids reinitializing models for every video.
    """
    try:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        frame_features = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect people with YOLO
            results = yolo(frame_rgb, verbose=False)
            
            # Find largest person (by bbox area)
            largest_person = None
            max_area = 0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls[0]) == 0:  # Person class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            largest_person = (x1, y1, x2, y2)
            
            # Extract MediaPipe features for largest person
            if largest_person is not None:
                x1, y1, x2, y2 = largest_person
                person_crop = frame_rgb[y1:y2, x1:x2]
                
                # Process with MediaPipe
                results_mp = holistic.process(person_crop)
                
                # Extract keypoints (543 total)
                keypoints = np.zeros((543, 3))  # x, y, confidence
                
                # Body pose (33 landmarks)
                if results_mp.pose_landmarks:
                    for i, lm in enumerate(results_mp.pose_landmarks.landmark):
                        keypoints[i] = [lm.x, lm.y, lm.visibility]
                
                # Face (468 landmarks)
                if results_mp.face_landmarks:
                    for i, lm in enumerate(results_mp.face_landmarks.landmark):
                        keypoints[33 + i] = [lm.x, lm.y, lm.visibility if hasattr(lm, 'visibility') else 1.0]
                
                # Left hand (21 landmarks)
                if results_mp.left_hand_landmarks:
                    for i, lm in enumerate(results_mp.left_hand_landmarks.landmark):
                        keypoints[501 + i] = [lm.x, lm.y, 1.0]
                
                # Right hand (21 landmarks)
                if results_mp.right_hand_landmarks:
                    for i, lm in enumerate(results_mp.right_hand_landmarks.landmark):
                        keypoints[522 + i] = [lm.x, lm.y, 1.0]
                
                frame_features.append(keypoints)
            else:
                # No person detected - append zeros
                frame_features.append(np.zeros((543, 3)))
        
        cap.release()
        
        if len(frame_features) == 0:
            return None
        
        return np.array(frame_features)
    
    except Exception as e:
        return None

def main():
    """Main extraction pipeline."""
    print("=" * 80)
    print("🎸 KINETICS-700-2020 FEATURE EXTRACTION")
    print("=" * 80)
    print()
    
    # Get all video files
    print("📂 Scanning video files...")
    video_files = get_video_files()
    total_videos = len(video_files)
    print(f"✅ Found {total_videos:,} videos")
    print()
    
    # Count by split and action
    train_count = sum(1 for v in video_files if v[0] == 'train')
    val_count = sum(1 for v in video_files if v[0] == 'val')
    print(f"   Train: {train_count:,} videos")
    print(f"   Val: {val_count:,} videos")
    print()
    
    # GPU setup
    num_gpus = torch.cuda.device_count()
    num_workers = num_gpus if num_gpus > 0 else 1
    print(f"🚀 Using {num_workers} workers ({num_gpus} GPUs)")
    print()
    
    # Create queues
    manager = mp.Manager()
    video_queue = manager.Queue()
    results_queue = manager.Queue()
    failed_list = manager.list()
    
    # Fill video queue
    for video_item in video_files:
        video_queue.put(video_item)
    
    # Add poison pills
    for _ in range(num_workers):
        video_queue.put(None)
    
    # Start workers
    workers = []
    for gpu_id in range(num_workers):
        worker = mp.Process(
            target=process_worker,
            args=(gpu_id, video_queue, results_queue, failed_list)
        )
        worker.start()
        workers.append(worker)
    
    # Monitor progress
    start_time = time.time()
    success_count = 0
    cached_count = 0
    failed_count = 0
    
    with tqdm(total=total_videos, desc="Processing", unit="video") as pbar:
        while success_count + cached_count + failed_count < total_videos:
            try:
                status, split, action = results_queue.get(timeout=1)
                
                if status == 'success':
                    success_count += 1
                elif status == 'cached':
                    cached_count += 1
                elif status == 'failed':
                    failed_count += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'new': success_count,
                    'cached': cached_count,
                    'failed': failed_count
                })
            except:
                continue
    
    # Wait for workers
    for worker in workers:
        worker.join()
    
    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 80)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Total videos: {total_videos:,}")
    print(f"Successful: {success_count:,} ({success_count/total_videos*100:.1f}%)")
    print(f"  - New: {success_count:,}")
    print(f"  - Cached: {cached_count:,}")
    print(f"Failed: {failed_count:,} ({failed_count/total_videos*100:.1f}%)")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Speed: {total_videos/elapsed*60:.1f} videos/minute")
    print("=" * 80)
    print()
    
    # Save failed videos
    if len(failed_list) > 0:
        failed_file = OUTPUT_ROOT / "failed_extractions.txt"
        with open(failed_file, 'w') as f:
            for video_path in failed_list:
                f.write(f"{video_path}\n")
        print(f"⚠️  Failed videos saved to: {failed_file}")

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
