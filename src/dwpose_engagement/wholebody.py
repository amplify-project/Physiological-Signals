import cv2
import numpy as np

import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose

class Wholebody:
    def __init__(self, det_model_path='models/dwpose/yolox_l.onnx', 
                 pose_model_path='models/dwpose/dw-ll_ucoco_384.onnx',
                 device='cuda',
                 device_id=0):
        """
        Initialize DWPose with ONNX runtime.
        
        Args:
            det_model_path: Path to detection model
            pose_model_path: Path to pose model
            device: 'cuda' or 'cpu'
            device_id: GPU device ID (only used when device='cuda')
        """
        import os
        # Convert to absolute paths if needed
        if not os.path.isabs(det_model_path):
            det_model_path = os.path.join(os.getcwd(), det_model_path)
        if not os.path.isabs(pose_model_path):
            pose_model_path = os.path.join(os.getcwd(), pose_model_path)
            
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            # Configure CUDA provider with specific device
            cuda_provider_options = {
                'device_id': device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
            providers = [
                ('CUDAExecutionProvider', cuda_provider_options),
                'CPUExecutionProvider'
            ]
        
        print(f"🔥 Loading DWPose models on {device} (device_id={device_id})...")
        print(f"   Detection model: {det_model_path}")
        print(f"   Pose model: {pose_model_path}")
        
        self.session_det = ort.InferenceSession(path_or_bytes=det_model_path, providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=pose_model_path, providers=providers)
        print("✅ DWPose models loaded successfully!")
    
    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]
        
        return keypoints, scores


