import os
import sys
import math

# =============================================================================
# VENV GUARD — must run before any third-party imports
# =============================================================================
# sys.prefix != sys.base_prefix is the canonical cross-platform venv check.
# VIRTUAL_ENV env var is set by activate scripts (bash/zsh/fish/PowerShell).
# Either condition is sufficient; we require at least one.
_in_venv = (sys.prefix != sys.base_prefix) or bool(os.environ.get('VIRTUAL_ENV'))
if not _in_venv:
    print("=" * 68)
    print("  ERROR: Virtual environment not active.")
    print()
    print("  Dependencies (torch, mediapipe, ultralytics, etc.) are installed")
    print("  inside .venv and will not be found without activating it first.")
    print()
    print("  Activate with:")
    print("    Windows:      .venv\\Scripts\\Activate.ps1")
    print("    macOS/Linux:  source .venv/bin/activate")
    print()
    print("  Then re-run this script.")
    print("=" * 68)
    sys.exit(1)

# =============================================================================
# MS STORE PYTHON WARNING — its sandbox blocks camera / hardware access
# =============================================================================
if sys.platform == 'win32' and 'WindowsApps' in sys.executable:
    print("=" * 68)
    print("  WARNING: You are running the Microsoft Store version of Python.")
    print("  Its sandbox restricts camera and hardware access.")
    print("  Install Python from https://www.python.org/downloads/ instead.")
    print("=" * 68)
    sys.exit(1)

# Suppress MediaPipe/TFLite C++ warnings (feedback manager, landmark projection)
# These are harmless internal logging from MediaPipe's C++ layer that fire from
# background threads directly to OS stderr fd — Python sys.stderr wrappers can't catch them.
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

# Redirect OS-level stderr (fd 2) to devnull to suppress C++ warnings,
# then restore it after MediaPipe model initialization.
# IMPORTANT: we set up the fds here but do NOT redirect yet — that happens
# lazily inside _silence_native_stderr(), called just before MediaPipe init.
# Doing it at module top hides genuine startup errors (YOLO/checkpoint loads)
# from the console and makes failures look like silent exits.
_stderr_fd = os.dup(2)
_devnull = os.open(os.devnull, os.O_WRONLY)

def _silence_native_stderr():
    """Redirect OS fd 2 to devnull (call right before MediaPipe init)."""
    os.dup2(_devnull, 2)

import cv2
if not hasattr(cv2, 'VideoCapture'):
    print("ERROR: cv2 imported but is non-functional (opencv-python may be broken).")
    print("       Reinstall with:  pip install --force-reinstall opencv-contrib-python")
    sys.exit(1)
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from ultralytics import YOLO

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import redis
import json
import argparse
import platform
import uuid
import signal
from datetime import datetime, timezone
from pathlib import Path
import shutil
from face_identifier import FaceIdentifier, IDENTIFIED_COLOR

# Shared async logger (writes to data/logs/engagement/<ts>.log).
# In the handover folder applog.py sits beside this script.
import sys as _sys
_here = Path(__file__).resolve().parent
if str(_here) not in _sys.path:
    _sys.path.insert(0, str(_here))
from applog import setup_logging, install_excepthook  # noqa: E402
import logging as _logging
log = _logging.getLogger('engagement')

# Optional: 360° support
try:
    import py360convert
    HAS_360_SUPPORT = True
except ImportError:
    HAS_360_SUPPORT = False
    print("ℹ️  py360convert not installed. 360° video support disabled.")
    print("   Install with: pip install py360convert")

# =============================================================================
# CROSS-PLATFORM DEVICE DETECTION
# =============================================================================

def get_platform_info():
    """Detect the current operating system.
    
    Returns:
        dict: Platform information including OS name and architecture
    """
    os_name = platform.system()  # 'Windows', 'Linux', 'Darwin' (macOS)
    machine = platform.machine()  # 'x86_64', 'arm64', 'AMD64', etc.
    
    is_apple_silicon = os_name == 'Darwin' and machine == 'arm64'
    
    return {
        'os': os_name,
        'machine': machine,
        'is_macos': os_name == 'Darwin',
        'is_windows': os_name == 'Windows',
        'is_linux': os_name == 'Linux',
        'is_apple_silicon': is_apple_silicon
    }

def get_best_device():
    """Detect the best available device for inference based on OS.
    
    Logic:
    - macOS: Check for Apple Silicon (MPS) → fallback to CPU
    - Windows/Linux: Check for CUDA (NVIDIA GPU) → fallback to CPU
    
    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    platform_info = get_platform_info()
    
    if platform_info['is_macos']:
        # macOS: Check for Apple Silicon MPS support
        if platform_info['is_apple_silicon']:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                print("⚠️  Apple Silicon detected but MPS not available (check PyTorch version)")
        else:
            print("ℹ️  Intel Mac detected - using CPU (no GPU acceleration)")
        return 'cpu'
    else:
        # Windows or Linux: Check for NVIDIA CUDA
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

def get_device(requested_device='auto'):
    """Get the device to use, validating the request against OS.
    
    Args:
        requested_device: 'auto', 'cuda', 'mps', or 'cpu'
        
    Returns:
        tuple: (torch.device, bool) - device and whether MPS is being used
    """
    platform_info = get_platform_info()
    use_mps = False
    
    if requested_device == 'auto':
        device_str = get_best_device()
        use_mps = device_str == 'mps'
    elif requested_device == 'cuda':
        if platform_info['is_macos']:
            print("⚠️  CUDA not available on macOS, using best available")
            device_str = get_best_device()
            use_mps = device_str == 'mps'
        elif torch.cuda.is_available():
            device_str = 'cuda'
        else:
            print("⚠️  CUDA not available, falling back to CPU")
            device_str = 'cpu'
    elif requested_device == 'mps':
        if not platform_info['is_macos']:
            print("⚠️  MPS only available on macOS, falling back to CUDA/CPU")
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_str = 'mps'
            use_mps = True
        else:
            print("⚠️  MPS not available, falling back to CPU")
            device_str = 'cpu'
    else:
        device_str = 'cpu'
    
    return torch.device(device_str), use_mps

# =============================================================================
# VIRTUAL ENVIRONMENT CHECK
# =============================================================================
def check_venv():
    """Warn if not running in a virtual environment."""
    import sys
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    if not in_venv:
        print("⚠️" + "="*60)
        print("⚠️  WARNING: Not running in virtual environment!")
        print("⚠️  This may cause missing dependencies or wrong PyTorch version.")
        print("⚠️")
        print("⚠️  Activate the venv first:")
        print("⚠️    Windows:    .\\.venv\\Scripts\\Activate.ps1")
        print("⚠️    macOS/Linux: source .venv/bin/activate")
        print("⚠️" + "="*60)
        print()
        
        # Give user a chance to see the warning
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Exiting. Please activate the venv and try again.")
            sys.exit(1)

# Run check on import
check_venv()

# =============================================================================
# CONFIGURATION (Can be overridden via command line arguments)
# =============================================================================
DEFAULT_REDIS_HOST = 'localhost'
DEFAULT_REDIS_PORT = 6379
REDIS_CHANNEL = 'engagement_score'  # Channel to publish to

# Cross-platform paths using pathlib.
# Two supported layouts:
#   1. Self-contained handover bundle (end-user, no git):
#        <bundle>/live_multiperson_binary_v2.py
#        <bundle>/model/best_model.pth
#        <bundle>/yolo11n.pt
#        <bundle>/yolo26n.pt
#   2. Project tree (developer, scripts/inference/):
#        <repo>/scripts/inference/live_multiperson_binary_v2.py
#        <repo>/models/action_transformer_12gpus_binary_v2_cleaned/best_model.pth
#        <repo>/yolo11n.pt
SCRIPT_DIR = Path(__file__).parent.resolve()
_LOCAL_MODEL = SCRIPT_DIR / 'model' / 'best_model.pth'
if _LOCAL_MODEL.exists():
    # Self-contained bundle
    MODEL_PATH = _LOCAL_MODEL
    YOLO_MODEL_PATH = SCRIPT_DIR / 'yolo11n.pt'
    YOLO_FALLBACK_PATH = SCRIPT_DIR / 'yolo26n.pt'
else:
    # Project tree
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    MODEL_PATH = PROJECT_ROOT / 'models' / 'action_transformer_12gpus_binary_v2_cleaned' / 'best_model.pth'
    YOLO_MODEL_PATH = PROJECT_ROOT / 'yolo11n.pt'
    YOLO_FALLBACK_PATH = PROJECT_ROOT / 'yolo26n.pt'

# Default session data directory: relative to CWD (not __file__)
# This ensures compatibility when packaged as an executable (PyInstaller etc.)
# where __file__ resolves inside the frozen bundle.
# Users can override with --save-dir.
SESSIONS_DIR = Path.cwd() / 'data' / 'sessions'

CONFIDENCE_THRESHOLD = 0.5          # Engagement threshold for binary decision

# FPS-Adaptive Sequence Length Configuration
# Training used 300 frames @ 30 FPS = 10 seconds of action context
# We dynamically calculate sequence length to always capture ~10 seconds
TARGET_DURATION_SECONDS = 10.0      # Target temporal context (matches training)
FPS_WARMUP_FRAMES = 60              # Frames to measure FPS during warmup
MIN_SEQUENCE_LENGTH = 60            # Minimum frames (2 sec @ 30fps fallback)
MAX_SEQUENCE_LENGTH = 300           # Maximum frames (model positional encoding limit)
DEFAULT_SEQUENCE_LENGTH = 64        # Fallback if FPS measurement fails
MIN_INFERENCE_FRAMES = 30           # Minimum frames before first estimate (~1s @ 30fps)

# 360° Video Configuration
NUM_360_VIEWS = 4                   # Number of perspective views to extract from 360° video
VIEW_FOV = (90, 90)                 # Field of view (horizontal, vertical) in degrees
VIEW_OUTPUT_SIZE = (480, 640)       # Output resolution (height, width) per view
EQUIRECT_MAX_WIDTH = 2880           # Downscale equirect source before e2p (saves ~4x CPU)

# =============================================================================
# 360° VIDEO DETECTION AND PROCESSING
# =============================================================================

def detect_video_format(frame):
    """Detect if frame is equirectangular (360°) or standard perspective.
    
    Equirectangular frames have a 2:1 aspect ratio (width = 2 * height).
    Common resolutions: 3840x1920, 5760x2880, 7680x3840
    
    Args:
        frame: Input frame (numpy array)
        
    Returns:
        str: 'equirectangular' or 'perspective'
    """
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    
    # Equirectangular frames are typically 2:1 aspect ratio (allow 5% tolerance)
    if 1.9 <= aspect_ratio <= 2.1:
        return 'equirectangular'
    else:
        return 'perspective'

def extract_perspective_views(equirect_frame, num_views=NUM_360_VIEWS, fov=VIEW_FOV, out_hw=VIEW_OUTPUT_SIZE):
    """Extract perspective views from 360° equirectangular frame.
    
    Args:
        equirect_frame: Input equirectangular frame (numpy array, RGB)
        num_views: Number of views to extract (default 4: front, right, back, left)
        fov: Field of view (h_fov, v_fov) in degrees
        out_hw: Output size (height, width)
        
    Returns:
        list: List of perspective view frames
        list: List of yaw angles for each view
    """
    if not HAS_360_SUPPORT:
        raise RuntimeError("py360convert not installed. Run: pip install py360convert")
    
    # Downscale equirectangular source to reduce e2p CPU cost
    h_src, w_src = equirect_frame.shape[:2]
    if w_src > EQUIRECT_MAX_WIDTH:
        scale = EQUIRECT_MAX_WIDTH / w_src
        new_w = EQUIRECT_MAX_WIDTH
        new_h = int(h_src * scale)
        # Ensure 2:1 ratio is maintained
        new_h = new_w // 2
        equirect_frame = cv2.resize(equirect_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    views = []
    yaw_angles = []
    
    def _extract_single(yaw):
        return py360convert.e2p(
            equirect_frame,
            fov_deg=fov,
            u_deg=yaw,
            v_deg=0,
            out_hw=out_hw,
            mode='bilinear'
        )
    
    for i in range(num_views):
        yaw_angles.append(i * (360 / num_views))
    
    # Parallel e2p — numpy releases the GIL so threads get true parallelism
    with ThreadPoolExecutor(max_workers=num_views) as pool:
        views = list(pool.map(_extract_single, yaw_angles))
    
    return views, yaw_angles

# =============================================================================
# DISK SPACE CHECK
# =============================================================================

# Frame prefetch — reads & decodes the next video frame while the main
# thread processes the current one.  Saves ~15-30 ms per iteration on
# high-res equirectangular files.
import queue as _queue

class FramePrefetcher:
    """Reads frames from a cv2.VideoCapture in a background thread."""
    
    def __init__(self, cap):
        self.cap = cap
        self._q = _queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
    
    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            self._q.put((ret, frame))
            if not ret:
                break
    
    def read(self):
        return self._q.get()
    
    def stop(self):
        self._stop.set()
        # Drain queue so reader thread can finish if blocked on put
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except _queue.Empty:
            pass
    
    def restart(self):
        """Restart after a video loop / seek."""
        self.stop()
        self._thread.join(timeout=2)
        self._stop.clear()
        # Drain any stale frames
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except _queue.Empty:
                break
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()


ESTIMATED_SESSION_GB_LOW  = 5.0   # lower bound estimate (GB per 30-min, 50 people)
ESTIMATED_SESSION_GB_HIGH = 7.0   # upper bound estimate
DISK_WARN_FREE_GB         = ESTIMATED_SESSION_GB_HIGH

def check_disk_space_for_save(output_dir):
    """Check available disk space before starting a logging session.

    Prints a summary of free space vs estimated session usage.
    If free space is below DISK_WARN_FREE_GB, warns the user and prompts
    to confirm before proceeding (or Ctrl+C to cancel).

    Returns (ok, free_gb): ok=False means space may be insufficient.
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    free_gb  = usage.free  / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)

    print(f"\n\U0001f4be Disk Space Check (--save is active):")
    print(f"   Available : {free_gb:.1f} GB  (of {total_gb:.1f} GB total)")
    print(f"   Estimated : {ESTIMATED_SESSION_GB_LOW:.0f}\u2013{ESTIMATED_SESSION_GB_HIGH:.0f} GB per 30-min session (~50 people)")

    if free_gb < DISK_WARN_FREE_GB:
        print(f"\n   \u26a0\ufe0f  WARNING: Low disk space!")
        print(f"   Estimated session usage ({ESTIMATED_SESSION_GB_LOW:.0f}\u2013{ESTIMATED_SESSION_GB_HIGH:.0f} GB) "
              f"may exceed available space ({free_gb:.1f} GB).")
        print(f"   Free up disk space before continuing, or use --save-engagement only (~370 MB / 30 min).")
        print(f"   Press ENTER to continue anyway, or Ctrl+C to cancel.")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print("Cancelled.")
            sys.exit(0)
    else:
        print(f"   \u2705  Sufficient space available.")

    return free_gb >= DISK_WARN_FREE_GB, free_gb

# =============================================================================
# ENGAGEMENT DATA LOGGER
# =============================================================================

class EngagementLogger:
    """Logs per-frame engagement data to JSONL for post-experience analysis.
    
    Writes two files per session:
    - {session_id}.jsonl: Per-frame granular data (append-only, crash-safe)
    - {session_id}_summary.json: Session summary (written on exit)
    
    JSONL format chosen for:
    - Append-only writes (no data loss on crash)
    - Trivial pandas loading: pd.read_json('file.jsonl', lines=True)
    - Clean handling of nested fields (bbox, view info)
    """
    
    def __init__(self, output_dir=None, video_source='', video_format='perspective',
                 device='cpu', model_path='', sequence_length=0,
                 save_engagement=True, save_keypoints=True):
        self.session_id = uuid.uuid4().hex[:12]
        self.start_time = datetime.now(timezone.utc)
        self.frame_number = 0
        self.save_engagement = save_engagement
        self.save_keypoints = save_keypoints
        
        # Session metadata
        self.video_source = str(video_source)
        self.video_format = video_format
        self.device = device
        self.model_path = str(model_path)
        self.sequence_length = sequence_length
        
        # Raw keypoint buffer for NPZ export
        # Stores (frame, track_id, keypoints_543x3, bbox) per person per frame
        self._kp_frames = []       # int
        self._kp_track_ids = []    # int
        self._kp_data = []         # np.ndarray (543, 3) each
        self._kp_bboxes = []       # (x1, y1, x2, y2) padded crop coords
        self._kp_frame_size = None # (width, height) of source frame

        # Disk safety state
        self.disk_full = False        # set True on first OSError — halts all further writes
        self._kp_chunk_files = []     # paths of periodic chunk NPZ files flushed to disk
        self._kp_chunk_interval = 900 # flush keypoint buffer every ~30s at 30 fps
        
        # Setup output directory and files
        # Each session gets its own subfolder: data/sessions/2026-02-10_15-30-00_abc123/
        if output_dir is None:
            output_dir = SESSIONS_DIR
        
        timestamp_str = self.start_time.strftime('%Y-%m-%d_%H-%M-%S')
        session_folder_name = f"{timestamp_str}_{self.session_id}"
        self.output_dir = Path(output_dir) / session_folder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.jsonl_path = self.output_dir / f"engagement_data.jsonl"
        # Keypoints are stored as numbered chunk files: keypoints_001.npz, _002.npz ...
        # No single merged file is written — use scripts/data/merge_keypoint_chunks.py for post-analysis
        
        # Open JSONL file for append-only writing
        self.jsonl_file = open(self.jsonl_path, 'a', encoding='utf-8')
        self._flush_counter = 0
        self._flush_interval = 30  # Flush to disk every 30 frames
        
        print(f"📊 Data Logging: {self.jsonl_path}")

    def _halt_logging(self, reason):
        """Disable all future file writes and clear buffered keypoints."""
        if self.disk_full:
            return
        self.disk_full = True
        self._kp_frames.clear()
        self._kp_track_ids.clear()
        self._kp_data.clear()
        self._kp_bboxes.clear()
        print(f"\n⚠️  DATA LOGGING STOPPED: {reason}")
    
    def log_frame(self, people_data, crowd_average, fps, is_360=False, frame_size=None):
        """Log one frame of engagement data.
        
        Args:
            people_data: List of dicts with keys: bbox, padded_bbox, id, score, keypoints, etc.
            crowd_average: Overall crowd engagement score (0.0-1.0)
            fps: Current measured FPS
            is_360: Whether this is a 360° frame
            frame_size: (width, height) of the source frame in pixels
        """
        now = time.time()
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')
        self.frame_number += 1
        
        # Capture frame dimensions once for NPZ export
        if frame_size is not None and self._kp_frame_size is None:
            self._kp_frame_size = frame_size
        
        # Log each person in this frame
        for person in people_data:
            track_id = int(person['id'])
            score = float(person['score'])
            bbox = [int(c) for c in person['bbox']]
            buffer_fill = person.get('buffer_fill', 0.0)
            
            record = {
                'timestamp': timestamp,
                'frame': self.frame_number,
                'track_id': track_id,
                'engagement_score': round(score, 4),
                'bbox': bbox,
                'buffer_fill': round(buffer_fill, 3),
                'crowd_average': round(crowd_average, 4),
                'people_count': len(people_data),
                'fps': round(fps, 1),
            }
            
            # Add 360° view info if present
            if is_360:
                record['view'] = person.get('view')
                record['yaw'] = person.get('yaw')
            
            # Add face identification if matched
            if person.get('identified_as'):
                record['identified_as'] = person['identified_as']
                record['face_similarity'] = round(person.get('face_similarity', 0.0), 3)
            
            if self.save_engagement and not self.disk_full:
                try:
                    self.jsonl_file.write(json.dumps(record) + '\n')
                except (OSError, MemoryError) as e:
                    self._halt_logging(f"JSONL write failed ({e})")
            
            # Buffer raw keypoints for NPZ export
            if self.save_keypoints and not self.disk_full:
                kp = person.get('keypoints')
                if kp is not None:
                    try:
                        self._kp_frames.append(self.frame_number)
                        self._kp_track_ids.append(track_id)
                        self._kp_data.append(kp)
                        self._kp_bboxes.append(person.get('padded_bbox', person['bbox']))
                    except MemoryError as e:
                        self._halt_logging(f"keypoint buffer allocation failed ({e})")
            
        # Periodic JSONL flush
        self._flush_counter += 1
        if self._flush_counter >= self._flush_interval:
            if self.jsonl_file and not self.disk_full:
                try:
                    self.jsonl_file.flush()
                except (OSError, MemoryError) as e:
                    self._halt_logging(f"JSONL flush failed ({e})")
            self._flush_counter = 0

        # Periodic NPZ chunk flush — keeps RAM usage bounded (~900 frames ≈ 30s at 30fps)
        if self.save_keypoints and not self.disk_full and len(self._kp_data) >= self._kp_chunk_interval:
            self._flush_keypoints_chunk()
    
    def _flush_keypoints_chunk(self):
        """Write the current in-memory keypoint buffer to a numbered chunk NPZ and clear it.

        Called periodically during the session (~every 30s at 30fps) to keep
        RAM usage bounded. Each chunk is a self-contained NPZ with the same
        array schema. Chunks are numbered from 001. No merging is done at
        shutdown — use scripts/data/merge_keypoint_chunks.py for post-analysis.
        """
        if not self._kp_data:
            return
        chunk_idx  = len(self._kp_chunk_files) + 1   # 1-based numbering
        chunk_path = self.output_dir / f"keypoints_{chunk_idx:03d}.npz"
        try:
            chunk_dict = dict(
                frames    = np.array(self._kp_frames,    dtype=np.int32),
                track_ids = np.array(self._kp_track_ids, dtype=np.int32),
                keypoints = np.stack(self._kp_data).astype(np.float16),
                bboxes    = np.array(self._kp_bboxes,    dtype=np.int32),
            )
            if self._kp_frame_size is not None:
                chunk_dict['frame_size'] = np.array(self._kp_frame_size, dtype=np.int32)
            np.savez_compressed(chunk_path, **chunk_dict)
            self._kp_chunk_files.append(chunk_path)
            # Clear in-memory buffer
            self._kp_frames.clear()
            self._kp_track_ids.clear()
            self._kp_data.clear()
            self._kp_bboxes.clear()
            print(f"   \U0001f4be Keypoint chunk {chunk_idx:03d} saved ({chunk_path.name})")
        except (OSError, MemoryError) as e:
            self._halt_logging(f"keypoint chunk write failed ({e})")

    def close(self):
        """Flush remaining keypoint buffer as a final chunk, then close JSONL."""
        if self.save_keypoints and not self.disk_full:
            try:
                self._flush_keypoints_chunk()  # flush any remaining in-memory data
            except Exception as e:
                print(f"\u26a0\ufe0f  Error saving final keypoint chunk: {e}")
        if self.jsonl_file:
            try:
                self.jsonl_file.flush()
                self.jsonl_file.close()
            except (OSError, MemoryError) as e:
                print(f"\u26a0\ufe0f  Error closing JSONL: {e}")

        duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        n_chunks  = len(self._kp_chunk_files)
        print(f"\n\U0001f4ca Session ended: {duration:.1f}s, {self.frame_number} frames")
        if self.save_engagement:
            print(f"   JSONL:      {self.jsonl_path}")
        if self.save_keypoints:
            print(f"   Keypoints:  {self.output_dir} ({n_chunks} chunk(s))")
            if n_chunks > 0:
                print(f"   To merge:   python scripts/data/merge_keypoint_chunks.py {self.output_dir}")


# =============================================================================
# MODEL DEFINITION
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
# INFERENCE SYSTEM
# =============================================================================

class MultiPersonEngagementSystem:
    def __init__(self, model_path, device='auto', redis_host='localhost', redis_port=6379,
                 sequence_length=None):
        self.device, self.use_mps = get_device(device)
        print(f"Using device: {self.device}")
        
        # MPS-specific settings
        if self.use_mps:
            print("ℹ️  MPS mode: Using float32 tensors for Apple Silicon compatibility")
            # MPS works best with float32
            self.tensor_dtype = torch.float32
        else:
            self.tensor_dtype = torch.float32  # Keep consistent across platforms
        
        # Sequence length (will be set dynamically if None)
        self.sequence_length = sequence_length if sequence_length else DEFAULT_SEQUENCE_LENGTH
        self.sequence_length_locked = sequence_length is not None  # If user specified, don't override
        
        # 1. Redis Connection
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
            self.redis_client.ping()
            print(f"✅ Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError:
            print(f"⚠️  WARNING: Could not connect to Redis at {redis_host}:{redis_port}. Publishing will be disabled.")
            self.redis_client = None
        
        # Publish throttle (1 Hz)
        self.last_publish_time = 0
        self.publish_interval = 1.0  # seconds

        # 2. Load YOLO (person detection)
        def _check_lfs(p):
            """Detect unmaterialised Git LFS pointer files (<1 KB and starts with 'version')."""
            try:
                if p.stat().st_size < 1024:
                    with open(p, 'rb') as fh:
                        head = fh.read(64)
                    if head.startswith(b'version https://git-lfs'):
                        print(f"❌ {p.name} is an unfetched Git LFS pointer ({p.stat().st_size} bytes).")
                        print(f"   Run 'git lfs install' once, then 'git lfs pull' in the repo root.")
                        sys.exit(1)
            except FileNotFoundError:
                pass
        _check_lfs(YOLO_MODEL_PATH)
        _check_lfs(YOLO_FALLBACK_PATH)
        _check_lfs(MODEL_PATH)
        if YOLO_MODEL_PATH.exists():
            print(f"Loading {YOLO_MODEL_PATH.name}...")
            try:
                self.yolo = YOLO(str(YOLO_MODEL_PATH))
            except Exception as e:
                print(f"❌ YOLO failed to load {YOLO_MODEL_PATH}: {e}")
                sys.exit(1)
        elif YOLO_FALLBACK_PATH.exists():
            print(f"⚠️  {YOLO_MODEL_PATH.name} not found, using {YOLO_FALLBACK_PATH.name}")
            self.yolo = YOLO(str(YOLO_FALLBACK_PATH))
        else:
            print(f"Downloading yolo11n.pt...")
            self.yolo = YOLO('yolo11n.pt')
        
        # 3. Initialize MediaPipe Holistic
        print("Initializing MediaPipe...")
        _silence_native_stderr()  # MediaPipe C++ init spams fd 2; restored after first frame
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 4. Load Engagement Model
        print(f"Loading Engagement Model from {model_path}...")
        self.model = TemporalTransformer(num_classes=2).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle DDP state dict (remove 'module.' prefix if present)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)
            self.model.eval()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
        
        # State Management
        # Dictionary to store feature buffers for each track_id
        # { track_id: deque(maxlen=self.sequence_length) }
        self.person_buffers = {}
        
        # Store latest scores for visualization and averaging
        # { track_id: score }
        self.person_scores = {}
        
        # Per-view YOLO trackers for 360° mode (avoids cross-view ID confusion)
        self._view_yolos = {}
        self._view_holistcs = {}
        
        # Persistent thread pool for parallel MediaPipe across views
        self._view_pool = ThreadPoolExecutor(max_workers=4)
    
    def _get_view_yolo(self, view_idx):
        """Get or create a YOLO tracker dedicated to a specific 360° view."""
        if view_idx not in self._view_yolos:
            if YOLO_MODEL_PATH.exists():
                self._view_yolos[view_idx] = YOLO(str(YOLO_MODEL_PATH))
            elif YOLO_FALLBACK_PATH.exists():
                self._view_yolos[view_idx] = YOLO(str(YOLO_FALLBACK_PATH))
            else:
                self._view_yolos[view_idx] = YOLO('yolo11n.pt')
        return self._view_yolos[view_idx]
    
    def _get_view_holistic(self, view_idx):
        """Get or create a MediaPipe Holistic instance for a specific view."""
        if view_idx not in self._view_holistcs:
            self._view_holistcs[view_idx] = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self._view_holistcs[view_idx]
    
    def detect_view(self, frame_bgr, view_idx):
        """Phase 1: Run YOLO detection on a single view (GPU, sequential).
        
        Returns list of (track_id, bbox) detections for this view.
        """
        yolo = self._get_view_yolo(view_idx)
        results = yolo.track(frame_bgr, persist=True, verbose=False, classes=[0], conf=0.15)
        
        detections = []
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            for box, raw_id in zip(boxes, track_ids):
                track_id = f"v{view_idx}_{raw_id}"
                x1, y1, x2, y2 = map(int, box)
                if track_id not in self.person_buffers:
                    self.person_buffers[track_id] = deque(maxlen=self.sequence_length)
                    self.person_scores[track_id] = 0.0
                detections.append((track_id, (x1, y1, x2, y2)))
        return detections
    
    def extract_and_infer_person(self, frame_rgb, track_id, bbox, view_idx):
        """Phase 2: MediaPipe features + engagement inference for one person (CPU-heavy).
        
        Safe to call from multiple threads — each view has its own holistic instance.
        """
        holistic = self._get_view_holistic(view_idx)
        features, padded_bbox = self._extract_features_with_holistic(frame_rgb, bbox, holistic)
        features_flat = features.flatten()
        self.person_buffers[track_id].append(features_flat)
        
        buf_len = len(self.person_buffers[track_id])
        if buf_len >= MIN_INFERENCE_FRAMES:
            buf_array = np.array(self.person_buffers[track_id])
            if buf_len < self.sequence_length:
                pad_rows = self.sequence_length - buf_len
                padding = np.zeros((pad_rows, buf_array.shape[1]), dtype=buf_array.dtype)
                buf_array = np.concatenate([padding, buf_array], axis=0)
            
            input_tensor = torch.tensor(
                buf_array, dtype=self.tensor_dtype, device=self.device
            ).unsqueeze(0)
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                score = probs[0][1].item()
                self.person_scores[track_id] = score
        
        return {
            'bbox': bbox,
            'padded_bbox': padded_bbox,
            'id': track_id,
            'score': self.person_scores[track_id],
            'buffer_fill': len(self.person_buffers[track_id]) / self.sequence_length,
            'keypoints': features,
        }
    
    def process_view(self, frame_bgr, view_idx, frame_rgb=None):
        """Process a single 360° view (sequential fallback)."""
        if frame_rgb is None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        detections = self.detect_view(frame_bgr, view_idx)
        current_frame_data = []
        for track_id, bbox in detections:
            person = self.extract_and_infer_person(frame_rgb, track_id, bbox, view_idx)
            current_frame_data.append(person)
        
        if current_frame_data:
            weighted_sum = sum(d['score'] * d['buffer_fill'] for d in current_frame_data
                              if len(self.person_buffers[d['id']]) >= MIN_INFERENCE_FRAMES)
            weight_total = sum(d['buffer_fill'] for d in current_frame_data
                              if len(self.person_buffers[d['id']]) >= MIN_INFERENCE_FRAMES)
            crowd_average = weighted_sum / weight_total if weight_total > 0 else 0.0
        else:
            crowd_average = 0.0
        
        return current_frame_data, crowd_average
    
    def _extract_features_with_holistic(self, frame_rgb, bbox, holistic):
        """Extract MediaPipe features using a specific holistic instance."""
        x1, y1, x2, y2 = bbox
        h, w = frame_rgb.shape[:2]
        pad_h = int((y2 - y1) * 0.2)
        pad_w = int((x2 - x1) * 0.2)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        padded_bbox = (x1, y1, x2, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros((543, 3)), padded_bbox
        
        person_crop = frame_rgb[y1:y2, x1:x2]
        results = holistic.process(person_crop)
        
        keypoints = np.zeros((543, 3))
        
        def get_landmarks(landmarks, offset):
            if landmarks:
                for i, lm in enumerate(landmarks.landmark):
                    keypoints[offset + i] = [lm.x, lm.y, lm.visibility if hasattr(lm, 'visibility') else 1.0]
        
        get_landmarks(results.pose_landmarks, 0)
        get_landmarks(results.face_landmarks, 33)
        get_landmarks(results.left_hand_landmarks, 33 + 468)
        get_landmarks(results.right_hand_landmarks, 33 + 468 + 21)
        
        return keypoints, padded_bbox

    
    def update_sequence_length(self, new_length):
        """Update sequence length and resize all existing buffers."""
        new_length = max(MIN_SEQUENCE_LENGTH, min(MAX_SEQUENCE_LENGTH, new_length))
        
        if new_length == self.sequence_length:
            return False  # No change needed
        
        old_length = self.sequence_length
        self.sequence_length = new_length
        
        # Resize all existing buffers
        for track_id in list(self.person_buffers.keys()):
            old_buffer = self.person_buffers[track_id]
            new_buffer = deque(maxlen=new_length)
            
            # Copy data from old buffer (most recent frames)
            # If shrinking, we keep the most recent frames
            # If expanding, we keep all existing frames
            data = list(old_buffer)
            if len(data) > new_length:
                data = data[-new_length:]  # Keep most recent
            new_buffer.extend(data)
            self.person_buffers[track_id] = new_buffer
        
        return True  # Buffer size changed

    def extract_features(self, frame_rgb, bbox):
        """Extract 543x3 keypoints from person crop.
        
        Returns:
            keypoints: (543, 3) array — landmarks in crop-relative [0,1] space
            padded_bbox: (x1, y1, x2, y2) — padded crop box in pixel coords
        """
        x1, y1, x2, y2 = bbox
        h, w, _ = frame_rgb.shape
        
        # Padding to ensure we get the whole person
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        padded_bbox = (x1, y1, x2, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros((543, 3)), padded_bbox
            
        person_crop = frame_rgb[y1:y2, x1:x2]
        
        # MediaPipe inference
        results = self.holistic.process(person_crop)
        
        # Extract keypoints
        keypoints = np.zeros((543, 3))
        
        def get_landmarks(landmarks, offset):
            if landmarks:
                for i, lm in enumerate(landmarks.landmark):
                    keypoints[offset + i] = [lm.x, lm.y, lm.visibility if hasattr(lm, 'visibility') else 1.0]
        
        get_landmarks(results.pose_landmarks, 0)
        get_landmarks(results.face_landmarks, 33)
        get_landmarks(results.left_hand_landmarks, 33 + 468)
        get_landmarks(results.right_hand_landmarks, 33 + 468 + 21)
        
        return keypoints, padded_bbox

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. YOLO Tracking — pass BGR frame (YOLO expects BGR, converts internally)
        # persist=True is crucial for ID tracking across frames
        results = self.yolo.track(frame, persist=True, verbose=False, classes=[0], conf=0.15)
        
        current_frame_data = [] # List of (bbox, track_id, score)
        
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            # Clean up old buffers for IDs that are no longer tracked
            active_ids = set(track_ids)
            # Optional: Implement a timeout instead of immediate removal to handle occlusion
            # For now, we'll keep it simple: if YOLO loses track, we reset.
            # self.person_buffers = {k: v for k, v in self.person_buffers.items() if k in active_ids}
            # self.person_scores = {k: v for k, v in self.person_scores.items() if k in active_ids}

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Initialize buffer if new person
                if track_id not in self.person_buffers:
                    self.person_buffers[track_id] = deque(maxlen=self.sequence_length)
                    self.person_scores[track_id] = 0.0 # Default start
                
                # 2. Extract Features
                features, padded_bbox = self.extract_features(frame_rgb, (x1, y1, x2, y2))
                features_flat = features.flatten()
                self.person_buffers[track_id].append(features_flat)
                
                # 3. Inference (progressive: run once we have MIN_INFERENCE_FRAMES)
                buf_len = len(self.person_buffers[track_id])
                if buf_len >= MIN_INFERENCE_FRAMES:
                    buf_array = np.array(self.person_buffers[track_id])  # (buf_len, 1629)
                    
                    # Pad to full sequence_length if buffer is not yet full
                    if buf_len < self.sequence_length:
                        pad_rows = self.sequence_length - buf_len
                        padding = np.zeros((pad_rows, buf_array.shape[1]), dtype=buf_array.dtype)
                        buf_array = np.concatenate([padding, buf_array], axis=0)
                    
                    input_tensor = torch.tensor(
                        buf_array, 
                        dtype=self.tensor_dtype,
                        device=self.device
                    ).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits = self.model(input_tensor)
                        probs = torch.softmax(logits, dim=1)
                        # Class 1 is 'Engaged'
                        score = probs[0][1].item()
                        self.person_scores[track_id] = score
                
                current_frame_data.append({
                    'bbox': (x1, y1, x2, y2),
                    'padded_bbox': padded_bbox,
                    'id': track_id,
                    'score': self.person_scores[track_id],
                    'buffer_fill': len(self.person_buffers[track_id]) / self.sequence_length,
                    'keypoints': features,
                })
        
        # 4. Calculate Crowd Average (confidence-weighted)
        if self.person_scores:
            # Include all people with at least MIN_INFERENCE_FRAMES, weighted by confidence
            weighted_sum = 0.0
            weight_total = 0.0
            for d in current_frame_data:
                bf = d.get('buffer_fill', 0.0)
                if len(self.person_buffers[d['id']]) >= MIN_INFERENCE_FRAMES:
                    weighted_sum += d['score'] * bf
                    weight_total += bf
            
            if weight_total > 0:
                crowd_average = weighted_sum / weight_total
            else:
                crowd_average = 0.0
        else:
            crowd_average = 0.0
            
        # 5. Publish to Redis (throttled to 1 Hz)
        current_time = time.time()
        if self.redis_client and (current_time - self.last_publish_time) >= self.publish_interval:
            try:
                payload = f"{crowd_average:.4f}"
                self.redis_client.publish(REDIS_CHANNEL, payload)
                self.last_publish_time = current_time
                print(f"📡 Redis pub → {REDIS_CHANNEL}: {payload}")
            except Exception as e:
                print(f"Redis Error: {e}")

        return current_frame_data, crowd_average

SIDEBAR_W = 280  # pixel width of the EmotiBit physio sidebar panel

# =============================================================================
# EMOTIBIT SIDEBAR
# =============================================================================

def draw_emotibit_sidebar(h, enrolled_names, emotibit_data, emotibit_lock,
                          focus_target, sidebar_radio_rects, sidebar_w):
    """Return a (h, sidebar_w, 3) uint8 image for the physio sidebar.
    Mutates sidebar_radio_rects in place with (y_top, y_bot, serial) tuples."""
    sidebar = np.full((h, sidebar_w, 3), 28, dtype=np.uint8)
    cv2.putText(sidebar, "EmotiBit", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1, cv2.LINE_AA)
    sidebar_radio_rects.clear()
    if not enrolled_names:
        cv2.putText(sidebar, "No enrolled", (8, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(sidebar, "participants", (8, 73),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(sidebar, "[R]=register serial", (8, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70, 70, 70), 1, cv2.LINE_AA)
        return sidebar
    n = len(enrolled_names)
    # Scale row height to fit up to 6 devices: generous when few, compact when many
    available_h = h - 28
    row_h = max(60, min(140, available_h // max(n, 1)))
    for i, serial in enumerate(enrolled_names):
        y0 = 28 + i * row_h
        y1 = min(y0 + row_h - 2, h - 2)
        sidebar_radio_rects.append((y0, y1, serial))
        selected = (focus_target == serial)
        if selected:
            cv2.rectangle(sidebar, (2, y0), (sidebar_w - 2, y1), (45, 28, 45), -1)
        # Radio button
        rb_cx, rb_cy = 11, y0 + 12
        if selected:
            cv2.circle(sidebar, (rb_cx, rb_cy), 6, (255, 0, 255), -1)
        else:
            cv2.circle(sidebar, (rb_cx, rb_cy), 6, (140, 140, 140), 1)
        # Serial label — truncate long IDs to fit
        label = serial if len(serial) <= 14 else serial[-14:]
        cv2.putText(sidebar, f"#{label}", (22, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
        # HR — same line as label, right-aligned area
        with emotibit_lock:
            d = emotibit_data.get(serial, {})
            snap_eda    = list(d.get('eda', []))
            snap_hr     = list(d.get('hr',  []))
            snap_eda_sd = list(d.get('eda_sd', []))
            snap_hr_sd  = list(d.get('hr_sd',  []))
            snap_metrics = dict(d.get('metrics', {}))
        if snap_hr:
            hr_text = f"{snap_hr[-1]:.0f}bpm"
            hr_col  = (100, 210, 100)
        elif snap_hr_sd:
            hr_text = f"HR SD {snap_hr_sd[-1]:.2f}"
            hr_col  = (100, 210, 100)
        else:
            hr_text = "--"
            hr_col  = (80, 80, 80)
        cv2.putText(sidebar, hr_text, (sidebar_w - 86, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, hr_col, 1, cv2.LINE_AA)
        # Temp ROC SD — small secondary readout right under HR
        temp_roc = snap_metrics.get('temperature_roc_sd')
        if temp_roc is not None:
            cv2.putText(sidebar, f"Ṫ SD {temp_roc:.3f}",
                        (sidebar_w - 86, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (170, 170, 220), 1, cv2.LINE_AA)
        # EDA spline plot — fills remaining vertical space.
        # Prefer raw EDA stream when available; otherwise plot the EDA SD stream
        # (which is what the SD-pipeline publishes since May 28 2026).
        if snap_eda:
            plot_arr   = snap_eda
            plot_label = "EDA"
            plot_unit  = "µS"
            line_col   = (80, 200, 255)
        elif snap_eda_sd:
            plot_arr   = snap_eda_sd
            plot_label = "EDA SD"
            plot_unit  = ""
            line_col   = (80, 200, 255)
        else:
            plot_arr = []
            plot_label = "EDA"
            plot_unit = ""
            line_col = (80, 200, 255)
        plot_x  = 4
        plot_y  = y0 + 22
        plot_pw = sidebar_w - 8
        plot_ph = y1 - plot_y - 3
        if plot_ph > 12:
            cv2.rectangle(sidebar, (plot_x, plot_y),
                          (plot_x + plot_pw, plot_y + plot_ph), (42, 42, 42), -1)
            if len(plot_arr) > 2:
                arr = np.array(plot_arr, dtype=np.float32)
                mn, mx = float(arr.min()), float(arr.max())
                rng = (mx - mn) if mx != mn else 1.0
                xs = np.linspace(plot_x + 1, plot_x + plot_pw - 2,
                                 len(arr)).round().astype(np.int32)
                ys = (plot_y + plot_ph - 2
                      - ((arr - mn) / rng * (plot_ph - 4))).round().astype(np.int32)
                ys = np.clip(ys, plot_y, plot_y + plot_ph - 2)
                pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
                cv2.polylines(sidebar, [pts], False, line_col, 1, cv2.LINE_AA)
                lbl_col = (75, 120, 150)
                cv2.putText(sidebar, f"{plot_label} {mx:.2f}{plot_unit}",
                            (plot_x + 2, plot_y + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.22, lbl_col, 1, cv2.LINE_AA)
                cv2.putText(sidebar, f"{mn:.2f}",
                            (plot_x + 2, plot_y + plot_ph - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.22, lbl_col, 1, cv2.LINE_AA)
                cv2.putText(sidebar, f"{arr[-1]:.3f}",
                            (sidebar_w - 46, plot_y + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.22, line_col, 1, cv2.LINE_AA)
            else:
                cv2.putText(sidebar, "Physio: no signal",
                            (plot_x + 4, plot_y + plot_ph // 2 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (65, 65, 65), 1, cv2.LINE_AA)
        if i < n - 1:
            cv2.line(sidebar, (4, y1), (sidebar_w - 4, y1), (50, 50, 50), 1)
    return sidebar


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Person Engagement Detection with Redis Streaming')
    parser.add_argument('--camera', type=int, default=None,
                        help='Force specific camera index (skips auto-detection)')
    parser.add_argument('--select-camera', '-s', action='store_true', default=False,
                        help='Always show the camera-selection preview (ignores last-used cache). '
                             'Useful when an external webcam is plugged in and the user wants '
                             'to pick it instead of the built-in laptop camera.')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file (overrides --camera)')
    parser.add_argument('--redis-host', type=str, default=DEFAULT_REDIS_HOST,
                        help=f'Redis server host/IP address (default: {DEFAULT_REDIS_HOST})')
    parser.add_argument('--redis-port', type=int, default=DEFAULT_REDIS_PORT,
                        help=f'Redis server port (default: {DEFAULT_REDIS_PORT})')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device for inference: auto (best available), cuda (NVIDIA), mps (Apple Silicon), cpu')
    parser.add_argument('--format', type=str, default='auto',
                        choices=['auto', 'perspective', '360'],
                        help='Video format: auto (detect from aspect ratio), perspective (standard 2D), 360 (equirectangular)')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Enable all data logging (engagement JSONL + keypoints NPZ)')
    parser.add_argument('--save-engagement', action='store_true', default=False,
                        help='Save per-frame engagement data only (JSONL)')
    parser.add_argument('--save-keypoints', action='store_true', default=False,
                        help='Save raw MediaPipe keypoints only (NPZ, float16)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Custom directory for saved session data (default: data/sessions/)')
    parser.add_argument('--no-face-id', action='store_true', default=False,
                        help='Disable facial identification (skip FaceIdentifier loading)')
    args = parser.parse_args()
    
    # Detect platform and device
    platform_info = get_platform_info()
    actual_device = get_best_device() if args.device == 'auto' else args.device

    # Wire up the shared async logger first so everything below is captured.
    setup_logging('engagement', extra_context={
        'camera': str(args.camera) if args.camera is not None else (
            'video:' + args.video if args.video else 'auto'),
        'redis': f"{args.redis_host}:{args.redis_port}",
        'device': args.device,
        'save': str(args.save or args.save_engagement or args.save_keypoints),
    })
    install_excepthook(log)
    log.info('Engagement inference starting (args=%s)', vars(args))

    print(f"🖥️  Platform: {platform_info['os']} ({platform_info['machine']})")
    if args.video:
        print(f"🎥 Video File: {args.video}")
    elif args.camera is not None:
        print(f"🎥 Camera: index {args.camera} (manual)")
    elif args.select_camera:
        print(f"🎥 Camera: interactive selection (--select-camera)")
    else:
        print(f"🎥 Camera: auto-detect  (tip: re-run with --select-camera to pick a different one)")
    print(f"📡 Redis Server: {args.redis_host}:{args.redis_port}")
    print(f"💻 Device: {actual_device} {'(auto-detected)' if args.device == 'auto' else ''}")
    
    # Initialize system with Redis configuration
    system = MultiPersonEngagementSystem(
        MODEL_PATH, 
        device=args.device,
        redis_host=args.redis_host,
        redis_port=args.redis_port
    )
    
    # Initialize Face Identifier (unless --no-face-id)
    face_id = None
    if not args.no_face_id:
        try:
            face_id = FaceIdentifier(device=actual_device)
        except Exception as e:
            print(f"⚠️  FaceIdentifier init failed: {e}  — running without face ID")
            face_id = None
    
    # Registration mode state
    registration_mode = False
    registration_flash_until = 0  # timestamp for on-screen flash message
    registration_flash_msg = ''
    
    # Face ID throttle: run MTCNN matching every N frames, carry forward results
    FACE_ID_INTERVAL = 10  # frames between face matching runs
    face_id_counter = 0
    face_id_cache = {}  # {track_id: {'name': str, 'similarity': float}}
    
    # Focus mode: which identified person(s) get magenta highlight
    # Values: 'all' | 'none' | '<emotibit serial>'
    # Cycle with F key: all → none → serial1 → serial2 → ... → all
    focus_target = 'all'

    # EmotiBit physio data — populated by subscriber thread
    emotibit_data = {}        # {serial: {'eda': deque, 'hr': deque, 'metrics': dict}}
    emotibit_lock = threading.Lock()
    sidebar_radio_rects = []  # [(y_top, y_bot, serial), ...] updated each frame
    video_w_box    = [0]      # video pixel width, set on first frame
    _mouse_cb_set  = [False]  # set mouse callback once after first imshow

    def on_mouse(event, x, y, flags, param):
        nonlocal focus_target
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        vw = video_w_box[0]
        if vw == 0 or x < vw:   # click is inside the video, not the sidebar
            return
        for (y0, y1, serial) in sidebar_radio_rects:
            if y0 <= y <= y1:
                focus_target = serial if focus_target != serial else 'all'
                return

    # EmotiBit Redis subscriber — runs in background, fills emotibit_data
    def _emotibit_subscriber():
        try:
            r_sub = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
            ps = r_sub.pubsub()
            ps.psubscribe('device:*:eda_filtered', 'device:*:hr_filtered', 'device:*:physio_metrics')
            for msg in ps.listen():
                if msg['type'] != 'pmessage':
                    continue
                try:
                    data = json.loads(msg['data'])
                    ch = msg['channel']
                    if isinstance(ch, bytes):
                        ch = ch.decode()
                    serial = ch.split(':')[1]
                    with emotibit_lock:
                        if serial not in emotibit_data:
                            emotibit_data[serial] = {
                                'eda': deque(maxlen=125),
                                'hr':  deque(maxlen=125),
                                'eda_sd': deque(maxlen=180),
                                'hr_sd':  deque(maxlen=180),
                                'metrics': {},
                            }
                        if 'EDA_filtered' in data:
                            emotibit_data[serial]['eda'].append(float(data['EDA_filtered']))
                        if 'HR_filtered' in data:
                            emotibit_data[serial]['hr'].append(float(data['HR_filtered']))
                        if ch.endswith(':physio_metrics'):
                            metric_aliases = {
                                'edl_sd': 'eda_sd',
                                'eda_sd': 'eda_sd',
                                'EDA_sd': 'eda_sd',
                                'hr_sd': 'hr_sd',
                                'HR_sd': 'hr_sd',
                                'temperature_roc_sd': 'temperature_roc_sd',
                                'scr_frequency_sd': 'scr_frequency_sd',
                                'ibi_sd': 'ibi_sd',
                            }
                            for source_key, target_key in metric_aliases.items():
                                if source_key in data:
                                    v = float(data[source_key])
                                    emotibit_data[serial]['metrics'][target_key] = v
                                    if target_key == 'eda_sd':
                                        emotibit_data[serial]['eda_sd'].append(v)
                                    elif target_key == 'hr_sd':
                                        emotibit_data[serial]['hr_sd'].append(v)
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️  EmotiBit subscriber error: {e}")

    threading.Thread(target=_emotibit_subscriber, daemon=True).start()

    # =========================================================================
    # SMART CAMERA DETECTION
    # Prioritize 360° cameras, then fall back to standard 2D cameras
    # Filters out virtual cameras (OBS, NDI) that don't produce real content
    # Caches last-used camera index for instant startup on repeat runs
    # =========================================================================
    CAMERA_CACHE_FILE = Path(__file__).parent / '.last_camera'
    cap = None
    video_source_name = ""
    
    def is_real_camera(cap, num_test_frames=3):
        """
        Check if a camera produces real, changing video content.
        Virtual cameras (OBS, NDI) often return blank or static frames.
        """
        frames = []
        for _ in range(num_test_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                return False, None
            frames.append(frame)
        
        last_frame = frames[-1]
        
        # Check 1: Is the frame mostly blank/black?
        mean_brightness = np.mean(last_frame)
        if mean_brightness < 5:
            return False, None
        
        # Check 2: Is the frame mostly uniform (single color)?
        std_dev = np.std(last_frame)
        if std_dev < 10:
            return False, None
        
        # Check 3: Are frames changing over time?
        if len(frames) >= 2:
            frame_diffs = []
            for i in range(1, len(frames)):
                diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float)).mean()
                frame_diffs.append(diff)
            if np.mean(frame_diffs) < 0.5:
                return False, None
        
        return True, last_frame

    def _open_camera(index, try_msmf_fallback=True):
        """Open camera trying DSHOW first (faster on most Windows), fall back to default (MSMF).
        Verifies the backend actually produces non-black frames before committing.
        Set try_msmf_fallback=False for scanning (avoids slow MSMF timeouts on non-existent cameras)."""
        if platform.system() == 'Windows':
            c = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if c.isOpened():
                ret, frame = c.read()
                if ret and frame is not None and np.mean(frame) > 5:
                    # DSHOW works and produces real frames — keep it
                    return c
                # DSHOW opened but returns black frames
                c.release()
                if try_msmf_fallback:
                    c = cv2.VideoCapture(index)
                    return c
                # No fallback requested — return a closed capture
                return cv2.VideoCapture()
            else:
                c.release()
                return cv2.VideoCapture()  # closed cap — no camera at this index
        # Non-Windows: default backend
        return cv2.VideoCapture(index)

    def select_camera_interactively(detected_cameras, default_camera, force=False):
        """
        Show a live tiled preview of all detected cameras and let the user
        choose one via a keypress.

        Key bindings:
            1 / 2 / 3 ...  — select camera N (1-based)
            Enter / Space  — accept highlighted default (360° preferred)
            Q / Escape     — quit application

        Returns the chosen camera dict from detected_cameras.
        When force=False (default) the selector is skipped for a single camera;
        set force=True (via --select-camera) to always show it.
        """
        if len(detected_cameras) == 1 and not force:
            return detected_cameras[0]

        TILE_W, TILE_H = 320, 240
        COLS = min(len(detected_cameras), 3)
        ROWS = math.ceil(len(detected_cameras) / COLS)
        BORDER = 8
        LABEL_H = 28
        tile_inner_w = TILE_W - 2 * BORDER
        tile_inner_h = TILE_H - 2 * BORDER - LABEL_H

        # Open a preview capture for every detected camera
        preview_caps = []
        for cam in detected_cameras:
            pc = _open_camera(cam['index'])
            preview_caps.append(pc)

        default_idx_in_list = detected_cameras.index(default_camera)

        win_name = "Camera Selection — press 1/2/3... or Enter for default, Q to quit"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, COLS * TILE_W, ROWS * TILE_H)

        print(f"\n\U0001f3a5  Multiple cameras detected — select one:")
        for i, cam in enumerate(detected_cameras):
            tags = []
            if cam.get('last_used'):
                tags.append("last used")
            if cam == default_camera:
                tags.append("default")
            tag_str = f"  \u2190 {', '.join(tags)}" if tags else ""
            print(f"   [{i+1}] Camera {cam['index']}: {cam['resolution']} ({cam['format']}){tag_str}")
        print(f"   Enter / Space \u2014 accept default  |  Q \u2014 quit\n")

        selected_number = None
        while selected_number is None:
            canvas = np.zeros((ROWS * TILE_H, COLS * TILE_W, 3), dtype=np.uint8)

            for i, (cam, pc) in enumerate(zip(detected_cameras, preview_caps)):
                row_i = i // COLS
                col_i = i % COLS
                x0 = col_i * TILE_W + BORDER
                y0 = row_i * TILE_H + LABEL_H + BORDER

                ret, frame = pc.read()
                if ret and frame is not None:
                    thumb = cv2.resize(frame, (tile_inner_w, tile_inner_h))
                else:
                    thumb = np.zeros((tile_inner_h, tile_inner_w, 3), dtype=np.uint8)
                    cv2.putText(thumb, "No feed", (10, tile_inner_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

                canvas[y0:y0 + tile_inner_h, x0:x0 + tile_inner_w] = thumb

                # Label bar at top of each tile
                lx0 = col_i * TILE_W
                ly0 = row_i * TILE_H
                is_default = (cam == default_camera)
                label_bg = (0, 130, 0) if is_default else (50, 50, 50)
                cv2.rectangle(canvas,
                              (lx0, ly0), (lx0 + TILE_W, ly0 + LABEL_H),
                              label_bg, -1)
                tag_360     = "  360\u00b0" if cam['is_360'] else ""
                tag_last    = "  last used" if cam.get('last_used') else ""
                default_tag = "  [default]" if is_default else ""
                label = f"[{i+1}] cam{cam['index']}  {cam['resolution']}{tag_360}{tag_last}{default_tag}"
                cv2.putText(canvas, label, (lx0 + 6, ly0 + 19),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1,
                            cv2.LINE_AA)

            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(30) & 0xFF

            # Number keys 1–9
            if ord('1') <= key <= ord('9'):
                choice = key - ord('0')  # 1-based
                if 1 <= choice <= len(detected_cameras):
                    selected_number = choice
            # Enter or Space → accept default
            elif key in (13, 32):
                selected_number = default_idx_in_list + 1
            # Q or Escape → quit
            elif key in (ord('q'), ord('Q'), 27):
                cv2.destroyAllWindows()
                for _ in range(30): cv2.waitKey(1)
                for pc in preview_caps:
                    pc.release()
                print("Quit from camera selection.")
                raise SystemExit(0)

        cv2.destroyAllWindows()
        for _ in range(30): cv2.waitKey(1)
        for pc in preview_caps:
            pc.release()

        chosen = detected_cameras[selected_number - 1]
        fmt_tag = "360\u00b0" if chosen['is_360'] else "2D"
        print(f"\u2705 Selected camera {chosen['index']} ({chosen['resolution']}, {fmt_tag})")
        return chosen

    if args.video:
        # Video file specified
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file {args.video}")
            return
        video_source_name = args.video
        print(f"📁 Opened video file: {args.video}")
    elif args.camera is not None:
        # Manual camera index specified - skip auto-detection
        cap = _open_camera(args.camera)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {args.camera}")
            return
        print(f"📷 Using camera index {args.camera} (manual)")
    else:
        # Auto-detect cameras (cross-platform compatible)
        # Try cached camera first, fall back to full sequential scan
        MAX_CAMERA_INDEX = 10
        detected_cameras = []
        skipped_virtual = 0

        # --- Phase 1: Try cached camera (fast path, 3 test frames) ---
        # When --select-camera is set, skip the cache entirely so the user
        # always gets a full scan + selector (e.g. they just plugged in a
        # USB webcam and want to pick it).
        cached_idx = None
        if not args.select_camera and CAMERA_CACHE_FILE.exists():
            try:
                cached_idx = int(CAMERA_CACHE_FILE.read_text().strip())
            except (ValueError, OSError):
                cached_idx = None

        if cached_idx is not None:
            print(f"⚡ Checking cached camera {cached_idx}...")
            test_cap = _open_camera(cached_idx, try_msmf_fallback=False)

            if test_cap.isOpened():
                is_real, test_frame = is_real_camera(test_cap)
                test_cap.release()
                if is_real and test_frame is not None:
                    fmt = detect_video_format(test_frame)
                    h, w = test_frame.shape[:2]
                    detected_cameras.append({
                        'index': cached_idx,
                        'format': fmt,
                        'resolution': f"{w}x{h}",
                        'is_360': fmt == 'equirectangular',
                        'last_used': True,
                    })
                    print(f"   ✓ Camera {cached_idx}: {w}x{h} ({fmt}) - LIVE (cached)")
                else:
                    print(f"   ✗ Cached camera {cached_idx} no longer valid, scanning all...")
            else:
                print(f"   ✗ Cached camera {cached_idx} not found, scanning all...")

        # --- Phase 2: Scan remaining indices for additional cameras ---
        # If cached camera was found, use a quick 1-frame check per index for speed;
        # brightness + uniformity checks still apply, frame-diff is skipped.
        # If cache miss, do a full 3-frame scan of all indices.
        remaining = [i for i in range(MAX_CAMERA_INDEX) if i != cached_idx]
        quick_mode = bool(detected_cameras)  # True = cached found, just looking for extras

        if quick_mode:
            print(f"🔍 Quick-scanning for additional cameras...")
        else:
            print(f"🔍 Scanning for cameras...")

        for idx in remaining:
            test_cap = _open_camera(idx, try_msmf_fallback=False)

            if test_cap.isOpened():
                # Always use 3 frames so frame-diff check runs — catches static
                # virtual cameras (e.g. SMPTE colour bars) that pass brightness/uniformity
                is_real, test_frame = is_real_camera(test_cap, num_test_frames=3)

                if is_real and test_frame is not None:
                    fmt = detect_video_format(test_frame)
                    h, w = test_frame.shape[:2]
                    detected_cameras.append({
                        'index': idx,
                        'format': fmt,
                        'resolution': f"{w}x{h}",
                        'is_360': fmt == 'equirectangular',
                        'last_used': False,
                    })
                    print(f"   ✓ Camera {idx}: {w}x{h} ({fmt}) - LIVE")
                else:
                    skipped_virtual += 1
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        print(f"   ✗ Camera {idx}: {w}x{h} - virtual/inactive")
                    else:
                        print(f"   ✗ Camera {idx}: no frames")

                test_cap.release()

        if not detected_cameras:
            # DSHOW scan found nothing — try MSMF backend on a few common indices
            # (MSMF is slower but works on some cameras where DSHOW fails)
            print(f"   DSHOW found no real cameras, trying alternative backend...")
            for idx in range(min(4, MAX_CAMERA_INDEX)):
                test_cap = cv2.VideoCapture(idx)
                if test_cap.isOpened():
                    is_real, test_frame = is_real_camera(test_cap, num_test_frames=3)
                    if is_real and test_frame is not None:
                        fmt = detect_video_format(test_frame)
                        h, w = test_frame.shape[:2]
                        detected_cameras.append({
                            'index': idx,
                            'format': fmt,
                            'resolution': f"{w}x{h}",
                            'is_360': fmt == 'equirectangular',
                            'last_used': False,
                            'backend': 'msmf',
                        })
                        print(f"   ✓ Camera {idx}: {w}x{h} ({fmt}) - LIVE (MSMF)")
                    test_cap.release()

        if not detected_cameras:
            print(f"\n❌ Error: No real cameras detected ({skipped_virtual} virtual cameras skipped)")
            print("   Virtual cameras (OBS, NDI) were filtered out.")
            print("   Try specifying a camera index with --camera <index>")
            print("   Or provide a video file with --video <path>")
            return
        
        # Determine default: last-used first, then 360°, then first 2D
        cameras_last = [c for c in detected_cameras if c.get('last_used')]
        cameras_360  = [c for c in detected_cameras if c['is_360']]
        if cameras_last:
            default_camera = cameras_last[0]
        elif cameras_360:
            default_camera = cameras_360[0]
        else:
            default_camera = detected_cameras[0]

        if skipped_virtual > 0:
            print(f"   ({skipped_virtual} virtual cameras filtered out)")

        if len(detected_cameras) > 1 or args.select_camera:
            selected = select_camera_interactively(
                detected_cameras, default_camera, force=args.select_camera,
            )
        else:
            selected = default_camera
            icon = "\U0001f310" if selected['is_360'] else "\U0001f4f7"
            kind = "360\u00b0" if selected['is_360'] else "2D"
            print(f"\n{icon} {kind} camera auto-selected: index {selected['index']} ({selected['resolution']})")
        
        # Cache selected camera for next startup
        try:
            CAMERA_CACHE_FILE.write_text(str(selected['index']))
        except OSError:
            pass
        
        # Open the selected camera (use MSMF if that's what worked during detection)
        if selected.get('backend') == 'msmf':
            cap = cv2.VideoCapture(selected['index'])
        else:
            cap = _open_camera(selected['index'])
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {selected['index']}")
            return
        video_source_name = f"Camera {selected['index']}"

    # =========================================================================
    # VIDEO FORMAT DETECTION (2D vs 360°)
    # =========================================================================
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read first frame")
        return
    
    # Determine video format
    if args.format == 'auto':
        video_format = detect_video_format(first_frame)
    elif args.format == '360':
        video_format = 'equirectangular'
    else:
        video_format = 'perspective'
    
    is_360 = video_format == 'equirectangular'
    
    if is_360:
        if not HAS_360_SUPPORT:
            print("❌ Error: 360° video detected but py360convert not installed")
            print("   Install with: pip install py360convert")
            return
        print(f"🌐 Video Format: 360° Equirectangular ({first_frame.shape[1]}x{first_frame.shape[0]})")
        print(f"   Extracting {NUM_360_VIEWS} perspective views @ {VIEW_FOV[0]}° FOV each")
    else:
        print(f"📷 Video Format: Standard Perspective ({first_frame.shape[1]}x{first_frame.shape[0]})")
    
    # Reset video to beginning (for video files)
    if args.video:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # =========================================================================
    # FPS CALIBRATION PHASE
    # Measure actual device FPS to calculate appropriate sequence length
    # Training used 300 frames @ 30 FPS = 10 seconds of context
    # Confidence will ramp from 0% → 100% as per-person buffers fill
    # =========================================================================
    print(f"⏱️  Calibrating FPS ({FPS_WARMUP_FRAMES} frames)...")
    fps_times = []
    warmup_start = time.time()
    
    for i in range(FPS_WARMUP_FRAMES):
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frames during calibration")
            return
        # Do minimal processing to simulate real load
        _ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fps_times.append(time.time() - frame_start)
        
        # Show calibration progress
        if (i + 1) % 20 == 0:
            current_fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
            print(f"   Calibrating: {i+1}/{FPS_WARMUP_FRAMES} frames, current estimate: {current_fps:.1f} FPS")
    
    warmup_duration = time.time() - warmup_start
    measured_fps = FPS_WARMUP_FRAMES / warmup_duration if warmup_duration > 0 else 30.0
    
    # Calculate dynamic sequence length
    calculated_seq_length = int(measured_fps * TARGET_DURATION_SECONDS)
    calculated_seq_length = max(MIN_SEQUENCE_LENGTH, min(MAX_SEQUENCE_LENGTH, calculated_seq_length))
    
    # Update system's sequence length
    system.sequence_length = calculated_seq_length
    
    print(f"\n📊 Calibration Results:")
    print(f"   Measured FPS: {measured_fps:.1f}")
    print(f"   Target Duration: {TARGET_DURATION_SECONDS} seconds")
    print(f"   Sequence Length: {calculated_seq_length} frames")
    print(f"   Actual Duration: {calculated_seq_length / measured_fps:.1f} seconds")
    print(f"   First estimate after: ~{MIN_INFERENCE_FRAMES} frames ({MIN_INFERENCE_FRAMES / measured_fps:.1f}s)")
    print(f"   Full confidence after: ~{calculated_seq_length} frames ({calculated_seq_length / measured_fps:.1f}s)")
    print()

    # =========================================================================
    # DATA LOGGING SETUP (if --save flag is set)
    # =========================================================================
    logger = None
    save_engagement = args.save or args.save_engagement
    save_keypoints = args.save or args.save_keypoints
    if save_engagement or save_keypoints:
        check_disk_space_for_save(args.save_dir if args.save_dir else SESSIONS_DIR)
        logger = EngagementLogger(
            output_dir=args.save_dir,
            video_source=video_source_name or 'camera',
            video_format=video_format,
            device=actual_device,
            model_path=str(MODEL_PATH),
            sequence_length=calculated_seq_length,
            save_engagement=save_engagement,
            save_keypoints=save_keypoints,
        )
        
        # Register graceful shutdown to ensure summary is written
        def graceful_shutdown(signum=None, frame=None):
            if logger:
                logger.close()
            sys.exit(0)
        signal.signal(signal.SIGINT, graceful_shutdown)
        signal.signal(signal.SIGTERM, graceful_shutdown)
    
    print(f"🚀 System Running!")
    print(f"📡 Publishing to Redis Channel: '{REDIS_CHANNEL}'")
    if is_360:
        print(f"🌐 360° Mode: Processing {NUM_360_VIEWS} views per frame")
    if logger:
        print(f"📊 Data Logging: ENABLED")
    print("Press 'q' to quit.")
    if face_id:
        print("Press 'r' to register, 'f' to cycle focus, 'c' to clear.")
    
    # Restore OS stderr after first MediaPipe inference triggers C++ init warnings
    _stderr_restored = False
    
    # Live FPS tracking
    fps_frame_times = deque(maxlen=30)  # Rolling window for FPS calculation
    last_frame_time = time.time()
    live_fps = measured_fps  # Start with measured value
    last_adaptation_time = time.time()
    adaptation_interval = 2.0  # Adapt every 2 seconds
    
    # Use prefetcher for 360° mode (large frame decode benefits from overlap)
    prefetcher = FramePrefetcher(cap) if is_360 else None

    # Name of the main display window. The window itself is created lazily on
    # the first imshow below (with WINDOW_NORMAL) so that any earlier OpenCV
    # windows (e.g. the --select-camera preview) finish cleanly first and we
    # don't leave a grey placeholder window on screen.
    _WIN_NAME = 'Concert Engagement System'

    while True:
        frame_start_time = time.time()
        
        if prefetcher:
            ret, frame = prefetcher.read()
        else:
            ret, frame = cap.read()
        if not ret:
            if args.video:
                # Loop video file — reset capture and tracker state
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if prefetcher:
                    prefetcher.restart()
                system.person_buffers.clear()
                system.person_scores.clear()
                system.yolo.predictor = None  # Reset main YOLO tracker
                # Reset per-view YOLO trackers (360° mode)
                for vy in system._view_yolos.values():
                    vy.predictor = None
                print("\n\U0001f501 Video looped — tracker reset")
                continue
            break
        
        # =====================================================================
        # PROCESS FRAME (2D vs 360°)
        # =====================================================================
        if is_360:
            # Convert to RGB for py360convert
            t0 = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract perspective views
            views, yaw_angles = extract_perspective_views(frame_rgb)
            t_e2p = time.time()
            
            # === PHASE 1: YOLO detection — sequential on GPU ===
            view_detections = []   # list of (view_idx, yaw, view_rgb, detections)
            for view_idx, (view_rgb, yaw) in enumerate(zip(views, yaw_angles)):
                view_bgr = cv2.cvtColor(view_rgb, cv2.COLOR_RGB2BGR)
                dets = system.detect_view(view_bgr, view_idx)
                view_detections.append((view_idx, yaw, view_rgb, dets))
            t_yolo = time.time()
            
            # === PHASE 2: MediaPipe + engagement — parallel per-view on CPU ===
            # Each view's holistic instance is used by exactly one thread.
            def _process_view_detections(view_idx, view_rgb, dets):
                results = []
                for track_id, bbox in dets:
                    results.append(
                        system.extract_and_infer_person(view_rgb, track_id, bbox, view_idx)
                    )
                return results
            
            all_people_data = []
            all_scores = []
            
            n_people = sum(len(d[3]) for d in view_detections)
            futures = []
            for view_idx, yaw, view_rgb, dets in view_detections:
                if dets:
                    fut = system._view_pool.submit(_process_view_detections, view_idx, view_rgb, dets)
                    futures.append((fut, view_idx, yaw))
            
            for fut, view_idx, yaw in futures:
                for person in fut.result():
                    person['view'] = view_idx
                    person['yaw'] = yaw
                    all_people_data.append(person)
                    if person['score'] > 0:
                        all_scores.append(person['score'])
            t_mp = time.time()
            
            # Print timing every 30 frames
            if hasattr(system, '_profile_counter'):
                system._profile_counter += 1
            else:
                system._profile_counter = 0
            if system._profile_counter % 30 == 0:
                print(f"  [PROFILE] e2p={(t_e2p-t0)*1000:.0f}ms  yolo={(t_yolo-t_e2p)*1000:.0f}ms  mp+eng={(t_mp-t_yolo)*1000:.0f}ms  people={n_people}  total={(t_mp-t0)*1000:.0f}ms")
            
            
            # Aggregate crowd average across all views
            if all_scores:
                crowd_avg = sum(all_scores) / len(all_scores)
            else:
                crowd_avg = 0.0
            
            people_data = all_people_data
            
            # Create visualization mosaic (2x2 grid of views with border)
            h, w = views[0].shape[:2]
            border = 2  # pixels between views
            mosaic = np.zeros((h * 2 + border, w * 2 + border, 3), dtype=np.uint8)
            mosaic[h:h + border, :] = (128, 128, 128)
            mosaic[:, w:w + border] = (128, 128, 128)
            
            view_labels = ['Front (0 deg)', 'Left (90 deg)', 'Right (180 deg)', 'Back (270 deg)']
            
            for view_idx, (view, label) in enumerate(zip(views, view_labels)):
                view_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                
                # Draw people boxes for this view
                for person in people_data:
                    if person.get('view') == view_idx:
                        x1, y1, x2, y2 = person['bbox']
                        score = person['score']
                        pid = person['id']
                        bf = person.get('buffer_fill', 1.0)
                        color = (0, int(255 * score), int(255 * (1-score)))
                        cv2.rectangle(view_bgr, (x1, y1), (x2, y2), color, 2)
                        if bf < 0.9:
                            # Draw confidence fill bar under bbox
                            bar_y = y2 + 2
                            bar_h = 4
                            bar_w = int((x2 - x1) * bf)
                            cv2.rectangle(view_bgr, (x1, bar_y), (x1 + bar_w, bar_y + bar_h), (255, 255, 0), -1)
                            plabel = f"ID:{pid} {score:.0%} conf:{bf:.0%}"
                        else:
                            plabel = f"ID:{pid} {score:.0%}"
                        cv2.putText(view_bgr, plabel, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Add view label
                cv2.putText(view_bgr, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Place in mosaic
                row, col = view_idx // 2, view_idx % 2
                y_off = row * (h + border)
                x_off = col * (w + border)
                mosaic[y_off:y_off + h, x_off:x_off + w] = view_bgr
            
            # Use mosaic as display frame
            display_frame = mosaic
        else:
            # Standard 2D processing
            people_data, crowd_avg = system.process_frame(frame)
            display_frame = frame.copy()
        
        # Restore OS stderr after first frame (MediaPipe C++ init warnings now past)
        if not _stderr_restored:
            import time as _t; _t.sleep(0.1)  # Let background threads finish writing
            os.dup2(_stderr_fd, 2)
            os.close(_devnull)
            os.close(_stderr_fd)
            _stderr_restored = True
        
        # =================================================================
        # FACE IDENTIFICATION — match tracked people against enrolled faces
        # Throttled: run MTCNN every FACE_ID_INTERVAL frames, cache results
        # =================================================================
        if face_id and face_id.has_enrollments and not is_360:
            face_id_counter += 1
            run_face_id = (face_id_counter % FACE_ID_INTERVAL == 0)
            
            for person in people_data:
                tid = person['id']
                if run_face_id:
                    x1, y1, x2, y2 = person['bbox']
                    fh, fw = frame.shape[:2]
                    cx1, cy1 = max(0, x1), max(0, y1)
                    cx2, cy2 = min(fw, x2), min(fh, y2)
                    if cx2 > cx1 and cy2 > cy1:
                        person_crop = frame[cy1:cy2, cx1:cx2]
                        match = face_id.identify_crop(person_crop)
                        if match:
                            face_id_cache[tid] = match
                        else:
                            face_id_cache.pop(tid, None)
                
                # Apply cached result
                cached = face_id_cache.get(tid)
                if cached:
                    person['identified_as'] = cached['name']
                    person['face_similarity'] = cached['similarity']
        
        # Draw Individual Boxes (2D mode only — 360° draws in mosaic above)
        if not is_360:
            for person in people_data:
                x1, y1, x2, y2 = person['bbox']
                score = person['score']
                pid = person['id']
                bf = person.get('buffer_fill', 1.0)
                identified = person.get('identified_as')
                
                # Determine if this person should be highlighted
                # Magenta box only when a specific EmotiBit is selected for this person
                show_highlight = (
                    identified and
                    focus_target == identified
                )
                
                if show_highlight:
                    # Magenta box + name for focused participant
                    color = IDENTIFIED_COLOR
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                    name_label = f"{identified} {score:.0%}"
                    cv2.putText(display_frame, name_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # Standard engagement gradient box
                    color = (0, int(255 * score), int(255 * (1-score)))
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    # Always show EmotiBit serial if enrolled; fall back to auto ID
                    id_str = identified if identified else f"ID:{pid}"
                    if bf < 0.9:
                        label = f"{id_str} {score:.0%} conf:{bf:.0%}"
                        cv2.putText(display_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        # Confidence fill bar under bounding box
                        bar_w_px = x2 - x1
                        bar_y_top = y2 + 2
                        bar_y_bot = y2 + 6
                        cv2.rectangle(display_frame, (x1, bar_y_top), (x2, bar_y_bot), (50, 50, 50), -1)
                        cv2.rectangle(display_frame, (x1, bar_y_top), (x1 + int(bar_w_px * bf), bar_y_bot), (255, 200, 0), -1)
                    else:
                        cv2.putText(display_frame, f"{id_str} {score:.0%}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate live FPS
        frame_duration = time.time() - frame_start_time
        fps_frame_times.append(frame_duration)
        if len(fps_frame_times) > 0:
            avg_frame_time = sum(fps_frame_times) / len(fps_frame_times)
            live_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Continuously adapt sequence length based on live FPS
        current_time = time.time()
        if current_time - last_adaptation_time >= adaptation_interval and len(fps_frame_times) >= 15:
            optimal_seq_length = int(live_fps * TARGET_DURATION_SECONDS)
            optimal_seq_length = max(MIN_SEQUENCE_LENGTH, min(MAX_SEQUENCE_LENGTH, optimal_seq_length))
            
            # Only adapt if drift is significant (more than 10% off target)
            current_duration = system.sequence_length / live_fps if live_fps > 0 else TARGET_DURATION_SECONDS
            drift = abs(current_duration - TARGET_DURATION_SECONDS) / TARGET_DURATION_SECONDS
            
            if drift > 0.10:  # More than 10% off target duration
                if system.update_sequence_length(optimal_seq_length):
                    new_duration = system.sequence_length / live_fps if live_fps > 0 else 0
                    print(f"\U0001f504 Adapted buffer: {system.sequence_length} frames @ {live_fps:.1f} FPS = {new_duration:.1f}s")
            
            last_adaptation_time = current_time
        
        # Calculate equivalent inference duration
        inference_duration = system.sequence_length / live_fps if live_fps > 0 else 0
        
        # --- VISUALIZATION (common for both modes) ---
        h, w = display_frame.shape[:2]
        
        # Draw Crowd Average Bar (Top Center)
        bar_w = min(400, w - 40)
        bar_h = 30
        bar_x = (w - bar_w) // 2
        bar_y = 20
        
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        # Fill Bar
        fill_w = int(bar_w * crowd_avg)
        avg_color = (0, int(255 * crowd_avg), int(255 * (1-crowd_avg)))
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), avg_color, -1)
        
        # Engagement Text
        mode_str = "360" if is_360 else "2D"
        text = f"CROWD ENGAGEMENT ({mode_str}): {crowd_avg:.1%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx = (w - tw) // 2
        ty = bar_y + bar_h + 25
        
        cv2.putText(display_frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)
        cv2.putText(display_frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FPS and Info (Bottom Left)
        platform_str = f"{platform_info['os']} | {actual_device.upper()}"
        people_count = len(people_data)
        context_seconds = system.sequence_length / live_fps if live_fps > 0 else 0
        fps_text = f"{platform_str} | FPS: {live_fps:.1f} | People: {people_count} | Context: {context_seconds:.1f}s"
        cv2.putText(display_frame, fps_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        cv2.putText(display_frame, fps_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Face ID status overlay (top-right corner, minimal)
        if face_id and not is_360:
            enrolled_names = face_id.enrolled_names
            if enrolled_names:
                if focus_target == 'all':
                    focus_label = 'all'
                elif focus_target == 'none':
                    focus_label = 'off'
                else:
                    focus_label = focus_target
                id_text = f"ID({len(enrolled_names)}) [{focus_label}] [F/R/C]"
            else:
                id_text = "[R]=register"
            (itw, ith), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            ix = w - itw - 8
            iy = 18
            cv2.putText(display_frame, id_text, (ix, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(display_frame, id_text, (ix, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, IDENTIFIED_COLOR, 1)
        
        # Registration flash message
        if registration_flash_until > time.time():
            flash_text = registration_flash_msg
            (ftw, fth), _ = cv2.getTextSize(flash_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            fx = (w - ftw) // 2
            fy = h // 2
            cv2.putText(display_frame, flash_text, (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(display_frame, flash_text, (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, IDENTIFIED_COLOR, 2)

        # --- DATA LOGGING ---
        if logger:
            logger.log_frame(people_data, crowd_avg, live_fps, is_360=is_360, frame_size=(w, h))

        # Disk-full warning banner (bottom of frame, red background)
        if logger and logger.disk_full:
            warn_text = "!! DATA LOGGING STOPPED - LIVE MODE CONTINUES !!"
            (wt_w, wt_h), _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            wt_x = (w - wt_w) // 2
            wt_y = h - 15
            cv2.rectangle(display_frame, (0, h - wt_h - 20), (w, h), (0, 0, 180), -1)
            cv2.putText(display_frame, warn_text, (wt_x, wt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        _sidebar = draw_emotibit_sidebar(
            h,
            face_id.enrolled_names if face_id and not is_360 else [],
            emotibit_data, emotibit_lock, focus_target, sidebar_radio_rects, SIDEBAR_W,
        )
        # Lazily create the main window as resizable on the very first frame.
        # Doing this here (after any --select-camera preview has been torn
        # down) avoids a stray empty grey window on screen during startup.
        # Without WINDOW_NORMAL OpenCV defaults to WINDOW_AUTOSIZE, which on
        # smaller laptop screens pushes the title bar, physio sidebar (RHS)
        # and FPS overlay (bottom) off-screen and breaks the 'q' shortcut.
        if not _mouse_cb_set[0]:
            cv2.namedWindow(_WIN_NAME, cv2.WINDOW_NORMAL)

        cv2.imshow(_WIN_NAME, np.hstack([display_frame, _sidebar]))
        if not _mouse_cb_set[0]:
            cv2.setMouseCallback(_WIN_NAME, on_mouse)
            video_w_box[0] = w
            _mouse_cb_set[0] = True
            # Fit the window to the composited frame width on first frame,
            # but cap at 1600px so it never opens larger than typical laptop screens.
            _full_w = w + SIDEBAR_W
            _init_w = min(_full_w, 1600)
            _init_h = int(h * (_init_w / _full_w))
            cv2.resizeWindow(_WIN_NAME, _init_w, _init_h)

        key = cv2.waitKey(1) & 0xFF
        
        # --- KEY HANDLERS ---
        if key == ord('q'):
            break
        
        # R = Register face — pick from detected EmotiBit serials
        elif key == ord('r') and face_id and not is_360:
            with emotibit_lock:
                detected_serials = sorted(emotibit_data.keys())
            already_enrolled = set(face_id.enrolled_names)
            available = [s for s in detected_serials if s not in already_enrolled]

            chosen_serial = None
            if not available:
                # No EmotiBits detected yet — fall back to a no-op flash
                if detected_serials:
                    registration_flash_msg = "All detected EmotiBits already assigned"
                else:
                    registration_flash_msg = "No EmotiBit detected — start physio publisher"
                registration_flash_until = time.time() + 2.5
            else:
                sel_idx = 0
                input_cancelled = False
                while True:
                    overlay = display_frame.copy()
                    row_h  = 34
                    bw     = 420
                    bh_box = 52 + len(available) * row_h
                    bx1 = (w - bw) // 2
                    by1 = max(20, (h - bh_box) // 2)
                    bx2, by2 = bx1 + bw, by1 + bh_box
                    cv2.rectangle(overlay, (bx1 - 2, by1 - 2), (bx2 + 2, by2 + 2), (0, 0, 0), -1)
                    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), IDENTIFIED_COLOR, 2)
                    cv2.putText(overlay, "Select EmotiBit to assign to this face:",
                                (bx1 + 12, by1 + 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, (200, 200, 200), 1, cv2.LINE_AA)
                    for i, serial in enumerate(available):
                        ry = by1 + 38 + i * row_h
                        if i == sel_idx:
                            cv2.rectangle(overlay, (bx1 + 6, ry - 2),
                                          (bx2 - 6, ry + row_h - 6), (60, 20, 60), -1)
                            cv2.rectangle(overlay, (bx1 + 6, ry - 2),
                                          (bx2 - 6, ry + row_h - 6), IDENTIFIED_COLOR, 1)
                            txt_col = IDENTIFIED_COLOR
                        else:
                            txt_col = (170, 170, 170)
                        arrow = "> " if i == sel_idx else "  "
                        cv2.putText(overlay, f"{arrow}{serial}",
                                    (bx1 + 16, ry + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.58, txt_col, 1, cv2.LINE_AA)
                    cv2.putText(overlay, "Up/Down or 1-9 to select   Enter=confirm   Esc=cancel",
                                (bx1 + 12, by2 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.30, (100, 100, 100), 1, cv2.LINE_AA)
                    _sb_reg = draw_emotibit_sidebar(
                        h,
                        face_id.enrolled_names if face_id and not is_360 else [],
                        emotibit_data, emotibit_lock, focus_target,
                        sidebar_radio_rects, SIDEBAR_W,
                    )
                    cv2.imshow(_WIN_NAME, np.hstack([overlay, _sb_reg]))
                    k = cv2.waitKey(50) & 0xFF
                    if k == 27:                        # Esc — cancel
                        input_cancelled = True
                        break
                    elif k == 13:                      # Enter — confirm
                        chosen_serial = available[sel_idx]
                        break
                    elif k == 82 or k == 0:            # Up arrow (82 on Windows)
                        sel_idx = (sel_idx - 1) % len(available)
                    elif k == 84 or k == 1:            # Down arrow (84 on Windows)
                        sel_idx = (sel_idx + 1) % len(available)
                    elif ord('1') <= k <= ord('9'):    # Number shortcut
                        ni = k - ord('1')
                        if ni < len(available):
                            sel_idx = ni

            if chosen_serial:
                if people_data:
                    unidentified = [p for p in people_data if not p.get('identified_as')]
                    candidates = unidentified if unidentified else people_data
                    best = max(candidates,
                               key=lambda p: (p['bbox'][2]-p['bbox'][0]) * (p['bbox'][3]-p['bbox'][1]))
                    x1, y1, x2, y2 = best['bbox']
                    fh, fw = frame.shape[:2]
                    cx1, cy1 = max(0, x1), max(0, y1)
                    cx2, cy2 = min(fw, x2), min(fh, y2)
                    if cx2 > cx1 and cy2 > cy1:
                        crop = frame[cy1:cy2, cx1:cx2]
                        ok = face_id.register_from_crop(crop, name=chosen_serial)
                        if ok:
                            face_id_cache[best['id']] = {'name': chosen_serial, 'similarity': 1.0}
                            registration_flash_msg = f"Enrolled: {chosen_serial}  (ID:{best['id']})"
                        else:
                            registration_flash_msg = "Registration FAILED — no face detected"
                        registration_flash_until = time.time() + 2.0
                else:
                    registration_flash_msg = "No person tracked — step into frame"
                    registration_flash_until = time.time() + 2.0
        
        # F = Cycle focus target: all → none → Participant 1 → ... → all
        elif key == ord('f') and face_id and not is_360:
            names = face_id.enrolled_names
            if names:
                cycle = ['all', 'none'] + names
                try:
                    idx = cycle.index(focus_target)
                except ValueError:
                    idx = -1  # focus_target was deleted, reset
                focus_target = cycle[(idx + 1) % len(cycle)]
                if focus_target == 'all':
                    registration_flash_msg = "Focus: ALL identified"
                elif focus_target == 'none':
                    registration_flash_msg = "Focus: OFF (no highlights)"
                else:
                    registration_flash_msg = f"Focus: {focus_target}"
                registration_flash_until = time.time() + 1.5
        
        # C = Clear all enrollments
        elif key == ord('c') and face_id:
            face_id.clear_enrollments()
            focus_target = 'all'
            registration_flash_msg = "All enrollments cleared"
            registration_flash_until = time.time() + 2.0
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    if prefetcher:
        prefetcher.stop()
    cap.release()
    cv2.destroyAllWindows()
    
    if logger:
        logger.close()

if __name__ == "__main__":
    main()