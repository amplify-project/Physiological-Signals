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

def _restore_native_stderr():
    """Point OS fd 2 back at the real stderr. Safe to call repeatedly and even
    after the saved fd is closed, so the crash handler can un-hide a traceback
    when the app dies while stderr is still silenced."""
    try:
        os.dup2(_stderr_fd, 2)
    except OSError:
        pass

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
from face_identifier import create_face_identifier, head_crop, IDENTIFIED_COLOR

# Shared async logger (writes to data/logs/engagement/<ts>.log).
# In the handover folder applog.py sits beside this script.
import sys as _sys
_here = Path(__file__).resolve().parent
if str(_here) not in _sys.path:
    _sys.path.insert(0, str(_here))
from applog import setup_logging, install_excepthook, mirror_console_to_log  # noqa: E402
import logging as _logging
log = _logging.getLogger('engagement')


def _short_identity(name):
    """On-overlay identity: an EmotiBit serial (…0000334) collapses to its
    last 4 digits (0334); other labels are shown unchanged."""
    if name and name[-4:].isdigit():
        return name[-4:]
    return name

# Gaze-first rules layer (shared module, sits beside this script; same file
# as scripts/inference/gaze_rules.py). Provides the per-person gaze-ray ->
# crowd-focal-point engagement score and sticky performer detection that the
# action-transformer output is FUSED with (gaze = base, actions = rescue /
# override). See gaze_rules.py for the rule set and tunable thresholds.
from gaze_rules import (GazeRulesEngine, person_geometry, null_geometry,  # noqa: E402
                        appearance_sig, draw_gaze_rays, COL_PERFORMER)

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
# Per-participant engagement, one channel per EmotiBit. Serial matches Sowmya's
# physio source_id (the DI device id, e.g. MD-V5-0000334), so the AR client
# pattern-subscribes device:*:engagement alongside device:*:physio_metrics.
INDIVIDUAL_CHANNEL_FMT = 'device:{serial}:engagement'

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
#
# Engagement action model selection:
#   The family-concert model (attention-first disengagement detector) supersedes
#   the original 86-class binary model. CRITICAL: the two models use OPPOSITE
#   class orders, so ENGAGEMENT_IDX is bound to whichever file is resolved:
#     family_concert_v1 -> engaged = class 0  (disengaged = class 1)
#     original binary_v2 -> engaged = class 1  (disengaged = class 0)
#   Preference order per layout: family_concert first, original as fallback.
SCRIPT_DIR = Path(__file__).parent.resolve()
_BUNDLE_FC = SCRIPT_DIR / 'model' / 'best_model_family_concert.pth'
_BUNDLE_OLD = SCRIPT_DIR / 'model' / 'best_model.pth'
if _BUNDLE_FC.exists() or _BUNDLE_OLD.exists():
    # Self-contained bundle
    if _BUNDLE_FC.exists():
        MODEL_PATH = _BUNDLE_FC
        ENGAGEMENT_IDX = 0
    else:
        MODEL_PATH = _BUNDLE_OLD
        ENGAGEMENT_IDX = 1
    YOLO_MODEL_PATH = SCRIPT_DIR / 'yolo11n.pt'
    YOLO_FALLBACK_PATH = SCRIPT_DIR / 'yolo26n.pt'
else:
    # Project tree
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    _REPO_FC = PROJECT_ROOT / 'models' / 'action_transformer_family_concert_v1' / 'best_model.pth'
    if _REPO_FC.exists():
        MODEL_PATH = _REPO_FC
        ENGAGEMENT_IDX = 0
    else:
        MODEL_PATH = PROJECT_ROOT / 'models' / 'action_transformer_12gpus_binary_v2_cleaned' / 'best_model.pth'
        ENGAGEMENT_IDX = 1
    YOLO_MODEL_PATH = PROJECT_ROOT / 'yolo11n.pt'
    YOLO_FALLBACK_PATH = PROJECT_ROOT / 'yolo26n.pt'

# Default session data directory: relative to CWD (not __file__)
# This ensures compatibility when packaged as an executable (PyInstaller etc.)
# where __file__ resolves inside the frozen bundle.
# Users can override with --save-dir.
SESSIONS_DIR = Path.cwd() / 'data' / 'sessions'

CONFIDENCE_THRESHOLD = 0.5          # Engagement threshold for binary decision

# --- Gaze + actions late fusion (Gaze_Rules integration) ---
# Design (validated offline on the 4th-lab concert, family_concert_v1):
#   The gaze rules produce the BASE attention score and already presume
#   engaged (SCORE_SEED=1.0), decaying on off-focal / shifting gaze. The
#   action transformer is a DISENGAGEMENT detector: we only let it SUBTRACT
#   from the gaze base when it is confident the person is disengaged. There is
#   no additive "rescue" — a calm attentive audience must not be dragged down
#   by the model's ~50/50 idle output.
#     p_dis   = P(disengaged) from the action head
#     penalty = clamp((p_dis - ACTION_DIS_TAU)/(1-ACTION_DIS_TAU), 0, 1)
#               * ACTION_PENALTY_MAX
#     fused   = clamp(gaze_score - penalty, 0, 1)
# Before the show starts (no common focal point) the pipeline falls back to the
# pure action-model engaged score.
ACTION_DIS_TAU = 0.60               # only penalise above this disengagement prob
ACTION_PENALTY_MAX = 0.7            # max the action head can subtract from gaze
# Gaze rule thresholds are tuned at ~3 processed fps (offline stride-10
# validation); the live loop targets TARGET_FPS_FLOOR=12, so frame-count
# windows inside the engine are scaled by 4 to keep wall-clock behaviour.
GAZE_RATE_SCALE = 4.0

# FPS-Adaptive Sequence Length Configuration
# Training used 300 frames @ 30 FPS = 10 seconds of action context
# We dynamically calculate sequence length to always capture ~10 seconds
TARGET_DURATION_SECONDS = 10.0      # Target temporal context (matches training)
FPS_WARMUP_FRAMES = 60              # Frames to measure FPS during warmup
MIN_SEQUENCE_LENGTH = 60            # Minimum frames (2 sec @ 30fps fallback)
MAX_SEQUENCE_LENGTH = 300           # Maximum frames (model positional encoding limit)
DEFAULT_SEQUENCE_LENGTH = 64        # Fallback if FPS measurement fails
MIN_INFERENCE_FRAMES = 30           # Minimum frames before first estimate (~1s @ 30fps)

# --- Time-windowed buffer (Post-Bremen fix for context drift, roadmap 2.c) ---
# Training used 300-frame clips spanning 10 s at 30 fps. At inference our FPS
# varies with crowd size (3-30 fps), so a fixed-FRAME deque silently warps the
# real-time window the model sees (e.g. 300 frames @ 3 fps = 100 s of context,
# wildly off-distribution). Fix: store (timestamp, features) pairs covering the
# last TARGET_DURATION_SECONDS of wall-clock time, then RESAMPLE to exactly
# MODEL_INPUT_FRAMES before each model call. The model always sees its trained
# input shape over its trained time window, regardless of live FPS.
MODEL_INPUT_FRAMES = 300            # Fixed model input length (matches training)
MIN_INFERENCE_SECONDS = 1.0         # Wait until buffer spans >=1s before first inference
BUFFER_HARD_CAP_FRAMES = 600        # Defensive cap (10s @ 60fps); time-prune is primary

# --- Crowd-load FPS floor (Post-Bremen roadmap 2.d) ---
# MediaPipe holistic runs once per detected person per frame at ~30-50 ms each
# on CPU, so per-frame cost scales linearly with crowd size and FPS collapses
# in dense audiences (30 people * 40 ms = 1.2 s/frame = 0.8 FPS). Fix: cap
# MediaPipe calls per frame to a budget computed from a target FPS floor and a
# running EWMA of per-person extraction cost. Persons not picked this frame
# keep their cached score (still drawn, still aggregated) and are refreshed on
# a later frame via round-robin. The time-windowed buffer (roadmap 2.c) makes
# this safe: sparse-in-time features get resampled to MODEL_INPUT_FRAMES
# evenly, so the model is unaware of the throttle.
TARGET_FPS_FLOOR = 12.0             # Minimum FPS we try to hold regardless of crowd size
MP_BUDGET_FRACTION = 0.70           # Fraction of frame budget reserved for MediaPipe
MEDIAPIPE_BUDGET_MS = (1000.0 / TARGET_FPS_FLOOR) * MP_BUDGET_FRACTION
MP_TIME_EWMA_ALPHA = 0.1            # Per-frame smoothing for measured extraction cost
MP_TIME_DEFAULT_MS = 40.0           # Initial estimate before any measurement
MIN_PERSONS_PER_FRAME = 1           # Always extract at least one person per frame

# --- Stale-track eviction (Post-Bremen fix for ID inflation / RAM growth) ---
# YOLO botsort/bytetrack assigns a fresh monotonically-increasing id every time
# a person is re-detected after being lost. In dense crowds the live id set
# climbs into the 1000s while only ~20 people are present, leaking ~400 KB per
# stale id (each 64-frame x 1629-float feature buffer).
#
# STALE_TRACK_TIMEOUT_FRAMES: how many frames a track may be absent before its
#   buffer is dropped. Set comfortably above typical short occlusions so brief
#   YOLO drops still recover the same id, but short enough to bound RAM.
# MAX_TRACKED_IDS: hard cap; when exceeded, evict the oldest-seen ids until
#   under the cap. Protects against pathological ID inflation in crowds.
STALE_TRACK_TIMEOUT_FRAMES = 60      # ~2 s @ 30 fps
MAX_TRACKED_IDS = 64                  # generous upper bound for live audiences

# --- Long-range MediaPipe (Audience_Distance) ---
# Holistic's person/face detectors fail on small distant crops (all-zero
# keypoints -> engagement collapses to 0 beyond ~2-3m). Upscaling the crop
# before MediaPipe restores detection; landmarks are crop-relative so
# downstream geometry is unaffected.
MP_MIN_CROP_H = 384                  # upscale person crops shorter than this
MP_UPSCALE_MAX = 4.0                 # cap the blow-up of very distant crops

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
        
        # None crowd_average (nobody scorable) is logged as JSON null, not 0.0
        crowd_val = round(crowd_average, 4) if crowd_average is not None else None

        # Log each person in this frame
        for person in people_data:
            track_id = int(person['id'])
            scored = bool(person.get('scored'))
            score = float(person['score'])
            bbox = [int(c) for c in person['bbox']]
            buffer_fill = person.get('buffer_fill', 0.0)
            
            record = {
                'timestamp': timestamp,
                'frame': self.frame_number,
                'track_id': track_id,
                # EmotiBit serial = the persistent participant identity; always
                # present (null when unmatched) so every engagement row can be
                # grouped per-participant for post-concert analysis.
                'emotibit_id': person.get('identified_as'),
                # How that id was resolved: 'face' (live match this frame),
                # 'coast' (same track, face briefly unseen), 'inferred'
                # (re-bound by appearance after track churn) or null. Paired
                # with id_confidence so inferred guesses stay filterable.
                'id_source': person.get('id_source'),
                'id_confidence': round(person['id_confidence'], 3) if person.get('id_confidence') is not None else None,
                # null (not 0.0) for people we couldn't score, so unknowns don't
                # pollute downstream stats; 'scored' flags the real values.
                'engagement_score': round(score, 4) if scored else None,
                'scored': scored,
                'bbox': bbox,
                'buffer_fill': round(buffer_fill, 3),
                'crowd_average': crowd_val,
                'people_count': len(people_data),
                'fps': round(fps, 1),
            }
            if person.get('performer'):
                record['performer'] = True
            
            # Add 360° view info if present
            if is_360:
                record['view'] = person.get('view')
                record['yaw'] = person.get('yaw')
            
            # Face-match confidence for the EmotiBit binding (only when matched)
            if person.get('identified_as'):
                record['face_similarity'] = round(person.get('face_similarity', 0.0), 3)
            
            # Native face width in px — the distance-test ground truth
            if person.get('face_px') is not None:
                record['face_px'] = round(person['face_px'], 1)
            
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
        # Dual-camera mode disables this per-System crowd publish so the main
        # loop can publish ONE merged crowd average across both feeds instead
        # of each camera racing to overwrite engagement_score.
        self.publish_crowd = True
        # Console-print throttle for the per-publish line: print every second
        # for the first 3 publishes (so the user can confirm the pipeline is
        # alive), then once per 60 s. Mirrors multiemotibit_UDP_SD_RFv2's
        # throttling pattern to keep terminal I/O off the hot path.
        self._console_pub_count = 0
        self._console_pub_last = 0.0
        self._console_pub_interval = 60.0
        self._console_pub_warmup = 3
        # Per-participant publish throttle (independent of the crowd channel)
        self.last_individual_publish_time = 0.0
        self.individual_publish_interval = 1.0
        self._indiv_pub_count = 0
        self._indiv_pub_last = 0.0

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
        _silence_native_stderr()  # MediaPipe C++ constructor spams fd 2
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            # static_image_mode=True: process() runs on DIFFERENT people's
            # crops back-to-back; tracking mode carries landmark state from
            # the previous call (a DIFFERENT person), producing leaked
            # landmarks and stray gaze rays. No measurable speed cost.
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5
        )
        # Restore stderr straight after the noisy constructor so later startup
        # errors (camera probing, checkpoint load, first inference) are NOT
        # swallowed and mistaken for a silent exit.
        _restore_native_stderr()
        
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
        # Per-track feature buffer: deque of (monotonic_timestamp, features_flat)
        # tuples covering ~TARGET_DURATION_SECONDS of wall-clock time. Pruned
        # by timestamp on every append; resampled to MODEL_INPUT_FRAMES for
        # model calls. See _append_features() / _resample_buffer().
        self.person_buffers = {}
        
        # Store latest scores for visualization and averaging
        # { track_id: score }
        self.person_scores = {}

        # Registered-only engagement: when True, the crowd score aggregates ONLY
        # track_ids whose face matched an enrolled (registered) person — the
        # EmotiBit-wearing parents — not the whole crowd. registered_ids is
        # refreshed each frame by the main loop from face_id_cache. When face ID
        # is disabled / no enrollments / 360° mode, the main loop flips
        # registered_only off so aggregation falls back to all tracked people.
        self.registered_only = True
        self.registered_ids = set()
        # Real face-ID registrations are exempt from performer promotion (an
        # enrolled parent walking must not be mistaken for a performer). The
        # --max-people sensor sim turns this OFF so a mis-registered performer
        # can still promote out of the registered set.
        self.exempt_registered = True

        # Frame index of last sighting per track_id (for stale-track eviction).
        # Updated each time a track is seen; consulted by _evict_stale_tracks().
        self.person_last_seen = {}
        self._frame_counter = 0

        # Display-label remap (issue 5): YOLO's bytetrack inflates the id space
        # in crowded scenes (2k+ ids over a 47 min run). The raw ids are still
        # logged to JSONL/NPZ for analysis, but the on-screen overlay uses a
        # compact recyclable pool of P1/P2/... labels so the operator sees a
        # stable, audience-sized set of names. Slots are returned to the pool
        # when their underlying track is evicted by _evict_stale_tracks().
        self._display_id_map = {}    # track_id -> "P12"
        self._display_id_pool = []   # free integer slots, smallest-first
        self._display_id_next = 1    # next never-used integer

        # Longest wall-clock buffer span across active tracks, refreshed at the
        # end of each process_frame / process_view. Read by the overlay so the
        # "Context: X.Xs" readout reflects real time, not frame-count / FPS.
        self.current_context_seconds = 0.0

        # Crowd-throttle state (roadmap 2.d): EWMA of per-person MediaPipe cost
        # and a round-robin cursor so every person eventually gets refreshed.
        self._mp_avg_ms = MP_TIME_DEFAULT_MS
        self._rr_offset = 0
        self.last_selected_count = 0
        self.last_total_count = 0

        # Gaze rules engine (Gaze_Rules integration): crowd focal point,
        # per-person gaze scores, sticky performer registry. last_gaze_result
        # is read by the main loop for the overlay (rays, performer boxes,
        # PRE-SHOW / LIVE HUD state).
        self.gaze_engine = GazeRulesEngine(rate_scale=GAZE_RATE_SCALE)
        self.last_gaze_result = None
        # Tuned BoT-SORT config (long lost-track buffer, curbs id inflation)
        # when bundled beside the script; ultralytics default otherwise.
        _tracker_cfg = SCRIPT_DIR / 'botsort_gaze.yaml'
        self.tracker_cfg = str(_tracker_cfg) if _tracker_cfg.exists() else 'botsort.yaml'
        
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
                self._ensure_buffer(track_id)
                detections.append((track_id, (x1, y1, x2, y2)))
        return detections
    
    def extract_and_infer_person(self, frame_rgb, track_id, bbox, view_idx, selected=True):
        """Phase 2: MediaPipe features + engagement inference for one person (CPU-heavy).
        
        Safe to call from multiple threads — each view has its own holistic instance.
        When selected=False, skips MediaPipe + model and returns cached score with a
        fresh bbox; this is the throttled-frame path (roadmap 2.d).
        """
        if not selected:
            self._ensure_buffer(track_id)
            return {
                'bbox': bbox,
                'padded_bbox': bbox,
                'id': track_id,
                'score': self.person_scores[track_id],
                'buffer_fill': min(self._buffer_time_span(track_id) / TARGET_DURATION_SECONDS, 1.0),
                'keypoints': None,
            }

        holistic = self._get_view_holistic(view_idx)
        mp_t0 = time.perf_counter()
        features, padded_bbox = self._extract_features_with_holistic(frame_rgb, bbox, holistic)
        self._record_mp_time((time.perf_counter() - mp_t0) * 1000.0)
        features_flat = features.flatten()
        self._append_features(track_id, features_flat)

        if self._buffer_ready(track_id):
            buf_array = self._resample_buffer(track_id)
            if buf_array is not None:
                input_tensor = torch.tensor(
                    buf_array, dtype=self.tensor_dtype, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    logits = self.model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    score = probs[0][ENGAGEMENT_IDX].item()
                    self.person_scores[track_id] = score

        return {
            'bbox': bbox,
            'padded_bbox': padded_bbox,
            'id': track_id,
            'score': self.person_scores[track_id],
            'buffer_fill': min(self._buffer_time_span(track_id) / TARGET_DURATION_SECONDS, 1.0),
            'keypoints': features,
        }
    
    def process_view(self, frame_bgr, view_idx, frame_rgb=None):
        """Process a single 360° view (sequential fallback)."""
        if frame_rgb is None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._frame_counter += 1

        detections = self.detect_view(frame_bgr, view_idx)
        active_ids = set(tid for tid, _ in detections)
        self._evict_stale_tracks(active_ids)

        # Per-view person budget (roadmap 2.d). Each view runs sequentially in
        # its own thread so a per-view budget keeps per-view frame cost bounded.
        ordered_ids = [tid for tid, _ in detections]
        selected_ids = self._select_persons_for_extraction(ordered_ids)

        current_frame_data = []
        for track_id, bbox in detections:
            person = self.extract_and_infer_person(
                frame_rgb, track_id, bbox, view_idx,
                selected=(track_id in selected_ids),
            )
            current_frame_data.append(person)
        
        crowd_average = None
        if current_frame_data:
            scorable = [d for d in current_frame_data
                        if self._buffer_ready(d['id'])
                        and (not self.registered_only or int(d['id']) in self.registered_ids)]
            for d in scorable:
                d['scored'] = True
            weight_total = sum(d['buffer_fill'] for d in scorable)
            if weight_total > 0:
                weighted_sum = sum(d['score'] * d['buffer_fill'] for d in scorable)
                crowd_average = weighted_sum / weight_total

        self._update_context_seconds()
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
        ch, cw = person_crop.shape[:2]
        if ch < MP_MIN_CROP_H:
            s = min(MP_UPSCALE_MAX, MP_MIN_CROP_H / ch)
            person_crop = cv2.resize(person_crop, (max(1, int(cw * s)), int(ch * s)),
                                     interpolation=cv2.INTER_CUBIC)
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
        ch, cw = person_crop.shape[:2]
        if ch < MP_MIN_CROP_H:
            s = min(MP_UPSCALE_MAX, MP_MIN_CROP_H / ch)
            person_crop = cv2.resize(person_crop, (max(1, int(cw * s)), int(ch * s)),
                                     interpolation=cv2.INTER_CUBIC)
        
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

    def _evict_stale_tracks(self, active_ids):
        """Drop buffers for ids absent longer than STALE_TRACK_TIMEOUT_FRAMES,\n        and enforce MAX_TRACKED_IDS by evicting the longest-unseen first.\n        Bounds RAM and per-frame cost in crowded scenes where YOLO's\n        re-detection inflates the id space (see Post-Bremen roadmap 2.b)."""
        for tid in active_ids:
            self.person_last_seen[tid] = self._frame_counter
        cutoff = self._frame_counter - STALE_TRACK_TIMEOUT_FRAMES
        stale = [tid for tid, last in self.person_last_seen.items()
                 if last < cutoff and tid not in active_ids]
        for tid in stale:
            self.person_buffers.pop(tid, None)
            self.person_scores.pop(tid, None)
            self.person_last_seen.pop(tid, None)
            self._release_display_id(tid)
            self.gaze_engine.evict(int(tid))
        if len(self.person_buffers) > MAX_TRACKED_IDS:
            ordered = sorted(self.person_last_seen.items(), key=lambda kv: kv[1])
            n_drop = len(self.person_buffers) - MAX_TRACKED_IDS
            for tid, _ in ordered[:n_drop]:
                if tid in active_ids:
                    continue
                self.person_buffers.pop(tid, None)
                self.person_scores.pop(tid, None)
                self.person_last_seen.pop(tid, None)
                self._release_display_id(tid)
                self.gaze_engine.evict(int(tid))

    # --- Display-label remap -------------------------------------------------
    def display_label(self, track_id):
        """Return a short, recyclable on-screen label (e.g. 'P3') for a YOLO
        track id. Allocates the smallest free slot, or a fresh one if the
        pool is empty. Slots are returned to the pool when the track is
        evicted by _evict_stale_tracks(). Raw track_id is still kept in the
        JSONL/NPZ logs for offline analysis."""
        tid = int(track_id)
        lbl = self._display_id_map.get(tid)
        if lbl is not None:
            return lbl
        if self._display_id_pool:
            slot = self._display_id_pool.pop(0)
        else:
            slot = self._display_id_next
            self._display_id_next += 1
        lbl = f"P{slot}"
        self._display_id_map[tid] = lbl
        return lbl

    def _release_display_id(self, track_id):
        tid = int(track_id)
        lbl = self._display_id_map.pop(tid, None)
        if lbl is None:
            return
        try:
            slot = int(lbl[1:])
        except ValueError:
            return
        # Insert in sorted order so display_label() always picks the smallest
        # free slot first — keeps the visible numbers small and stable.
        idx = 0
        while idx < len(self._display_id_pool) and self._display_id_pool[idx] < slot:
            idx += 1
        self._display_id_pool.insert(idx, slot)

    # --- Time-windowed feature buffer (roadmap 2.c) -------------------------
    # Buffer entries are (monotonic_timestamp, features_flat) tuples. Stored in
    # a deque so old entries pop in O(1) from the left during time-pruning; the
    # deque maxlen is a defensive cap only (the time-window prune is primary).

    def _ensure_buffer(self, track_id):
        if track_id not in self.person_buffers:
            self.person_buffers[track_id] = deque(maxlen=BUFFER_HARD_CAP_FRAMES)
            self.person_scores[track_id] = 0.0

    def _append_features(self, track_id, features_flat):
        """Append a feature vector with the current monotonic timestamp,
        then prune entries older than TARGET_DURATION_SECONDS."""
        self._ensure_buffer(track_id)
        buf = self.person_buffers[track_id]
        now = time.monotonic()
        buf.append((now, features_flat))
        cutoff = now - TARGET_DURATION_SECONDS
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def _buffer_time_span(self, track_id):
        buf = self.person_buffers.get(track_id)
        if not buf or len(buf) < 2:
            return 0.0
        return buf[-1][0] - buf[0][0]

    def _buffer_ready(self, track_id):
        return self._buffer_time_span(track_id) >= MIN_INFERENCE_SECONDS

    def _resample_buffer(self, track_id):
        """Resample buffered (ts, features) pairs to exactly MODEL_INPUT_FRAMES
        evenly-spaced points across the buffer's current time span. Linear
        interpolation per feature dimension (vectorised). Returns (300, 1629)
        ndarray, or None if the buffer is too short."""
        buf = self.person_buffers.get(track_id)
        if not buf or len(buf) < 2:
            return None
        ts = np.fromiter((t for t, _ in buf), dtype=np.float64, count=len(buf))
        span = ts[-1] - ts[0]
        if span < 1e-3:
            return None
        feats = np.stack([f for _, f in buf]).astype(np.float32, copy=False)
        target_ts = np.linspace(ts[0], ts[-1], MODEL_INPUT_FRAMES)
        right = np.searchsorted(ts, target_ts, side='left').clip(1, len(ts) - 1)
        left = right - 1
        w = ((target_ts - ts[left]) / (ts[right] - ts[left] + 1e-9)).reshape(-1, 1).astype(np.float32)
        return feats[left] * (1.0 - w) + feats[right] * w

    def _update_context_seconds(self):
        spans = [self._buffer_time_span(tid) for tid in self.person_buffers]
        self.current_context_seconds = min(max(spans), TARGET_DURATION_SECONDS) if spans else 0.0

    # --- Crowd-load throttling (roadmap 2.d) --------------------------------

    def _select_persons_for_extraction(self, ordered_track_ids):
        """Pick which persons get a fresh MediaPipe pass this frame.
        Budget = MEDIAPIPE_BUDGET_MS / EWMA(per-person extract cost), clamped
        to [MIN_PERSONS_PER_FRAME, N]. Round-robin so every person gets a turn.
        Returns a set of track_ids selected for full extraction this frame."""
        n = len(ordered_track_ids)
        self.last_total_count = n
        if n == 0:
            self.last_selected_count = 0
            return set()
        budget = max(MIN_PERSONS_PER_FRAME,
                     int(MEDIAPIPE_BUDGET_MS / max(self._mp_avg_ms, 1.0)))
        if budget >= n:
            self.last_selected_count = n
            return set(ordered_track_ids)
        start = self._rr_offset % n
        end = start + budget
        if end <= n:
            selected = ordered_track_ids[start:end]
        else:
            selected = ordered_track_ids[start:] + ordered_track_ids[:end - n]
        self._rr_offset = (self._rr_offset + budget) % n
        self.last_selected_count = len(selected)
        return set(selected)

    def _record_mp_time(self, elapsed_ms):
        self._mp_avg_ms = ((1.0 - MP_TIME_EWMA_ALPHA) * self._mp_avg_ms
                           + MP_TIME_EWMA_ALPHA * elapsed_ms)

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame_counter += 1
        
        # 1. YOLO Tracking — pass BGR frame (YOLO expects BGR, converts internally)
        # persist=True is crucial for ID tracking across frames
        results = self.yolo.track(frame, persist=True, verbose=False, classes=[0],
                                  conf=0.15, tracker=self.tracker_cfg)
        
        current_frame_data = [] # List of (bbox, track_id, score)
        gaze_people = {}        # tid -> {'geom','bbox','sig'} for the gaze rules engine
        
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            # Evict stale-track buffers BEFORE re-populating, to bound RAM and
            # per-frame scan cost. See _evict_stale_tracks() docstring.
            active_ids = set(int(t) for t in track_ids)
            self._evict_stale_tracks(active_ids)

            # Throttle MediaPipe to hold TARGET_FPS_FLOOR: pick a budget-sized
            # subset of persons this frame; unselected persons reuse cached score.
            ordered_ids = [int(t) for t in track_ids]
            # Registered-only compute: MediaPipe + model only run for enrolled
            # (registered) parents. Bystanders are still tracked by YOLO (cheap)
            # so they can be face-matched and enrolled, but skip the expensive
            # holistic + transformer entirely. This is the 5fps->12fps fix.
            if self.registered_only:
                extract_pool = [t for t in ordered_ids if t in self.registered_ids]
            else:
                extract_pool = ordered_ids
            selected_ids = self._select_persons_for_extraction(extract_pool)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                tid_int = int(track_id)

                # Bystander when registered-only: keep a fresh bbox for face ID /
                # enrolment but run no MediaPipe, no inference, no buffer.
                if self.registered_only and tid_int not in self.registered_ids:
                    gaze_people[tid_int] = {'geom': null_geometry((x1, y1, x2, y2)),
                                            'bbox': (x1, y1, x2, y2),
                                            'sig': appearance_sig(frame, (x1, y1, x2, y2))}
                    current_frame_data.append({
                        'bbox': (x1, y1, x2, y2),
                        'padded_bbox': (x1, y1, x2, y2),
                        'id': track_id,
                        'score': self.person_scores.get(track_id, 0.0),
                        'buffer_fill': 0.0,
                        'keypoints': None,
                    })
                    continue

                if tid_int in selected_ids:
                    # 2. Extract features and append with timestamp (auto-prunes >10s old)
                    mp_t0 = time.perf_counter()
                    features, padded_bbox = self.extract_features(frame_rgb, (x1, y1, x2, y2))
                    self._record_mp_time((time.perf_counter() - mp_t0) * 1000.0)
                    features_flat = features.flatten()
                    self._append_features(track_id, features_flat)

                    # Gaze geometry from the SAME keypoints the model consumes
                    gaze_geom = person_geometry(features, padded_bbox)

                    # 3. Inference (gated on wall-clock span, not frame count; resample
                    #    to MODEL_INPUT_FRAMES so the model always sees its trained shape)
                    if self._buffer_ready(track_id):
                        buf_array = self._resample_buffer(track_id)
                        if buf_array is not None:
                            input_tensor = torch.tensor(
                                buf_array,
                                dtype=self.tensor_dtype,
                                device=self.device
                            ).unsqueeze(0)
                            with torch.no_grad():
                                logits = self.model(input_tensor)
                                probs = torch.softmax(logits, dim=1)
                                # Engaged-class prob (ENGAGEMENT_IDX bound to model)
                                score = probs[0][ENGAGEMENT_IDX].item()
                                self.person_scores[track_id] = score
                else:
                    # Throttled this frame: skip MediaPipe + model, keep cached score.
                    # YOLO bbox is fresh so the overlay still tracks the person live.
                    self._ensure_buffer(track_id)
                    features = None
                    padded_bbox = (x1, y1, x2, y2)
                    gaze_geom = null_geometry((x1, y1, x2, y2))

                gaze_people[tid_int] = {'geom': gaze_geom, 'bbox': (x1, y1, x2, y2),
                                        'sig': appearance_sig(frame, (x1, y1, x2, y2))}

                current_frame_data.append({
                    'bbox': (x1, y1, x2, y2),
                    'padded_bbox': padded_bbox,
                    'id': track_id,
                    'score': self.person_scores[track_id],
                    'buffer_fill': min(self._buffer_time_span(track_id) / TARGET_DURATION_SECONDS, 1.0),
                    'keypoints': features,
                    'gaze_geom': gaze_geom,
                })
        
        # 3b. Gaze rules engine: crowd focal point, per-person gaze scores,
        # performer promotion / re-ID. Then FUSE with the action-model score:
        # gaze is the base, the model rescues confident pro cues and caps
        # confident anti cues. Performers are flagged and never scored.
        # Registered people are exempt from performer promotion (audience by
        # definition — walking must not silently unscore them), unless
        # exempt_registered is off (sensor sim: let performers promote out).
        self.gaze_engine.excluded_ids = (self.registered_ids
                                         if (self.registered_only and self.exempt_registered)
                                         else set())
        gaze_result = self.gaze_engine.update(gaze_people, self._frame_counter)
        self.last_gaze_result = gaze_result
        for d in current_frame_data:
            tid = int(d['id'])
            if tid in self.gaze_engine.performers:
                d['performer'] = True
                d['gazed_performer'] = tid in gaze_result['gazed_performers']
                continue
            # Only REGISTERED adults (face-ID-matched EmotiBit wearers) are
            # gaze-scored. Bystanders/infants carry null geometry (no rays,
            # no fusion); performers are handled above. In --no-face-id /
            # 360° mode registered_only is off and everyone is scored.
            if self.registered_only and tid not in self.registered_ids:
                continue
            stat = gaze_result['status'].get(tid)
            gaze_s = stat['score'] if stat else None
            if gaze_s is None:
                continue                    # PRE-SHOW / warming: pure model score
            d['gaze_reason'] = stat['reason']
            d['gaze_scored'] = True
            if self._buffer_ready(d['id']):
                # Presume-engaged base (gaze) MINUS a disengagement penalty from
                # the action head. score is P(engaged); p_dis = 1 - P(engaged).
                # The model can only pull the score DOWN, and only once it is
                # past ACTION_DIS_TAU confident of disengagement.
                action_engaged = d['score']
                p_dis = 1.0 - action_engaged
                penalty = max(0.0, (p_dis - ACTION_DIS_TAU) / (1.0 - ACTION_DIS_TAU))
                penalty = min(1.0, penalty) * ACTION_PENALTY_MAX
                fused = gaze_s - penalty
                d['score'] = max(0.0, min(1.0, fused))
            else:
                d['score'] = gaze_s                 # gaze-only until the model warms up

        # 4. Calculate Crowd Average (confidence-weighted)
        # Gaze-scored people count with full weight (the gaze score needs no
        # model warm-up); action-only people are weighted by buffer fill as
        # before. Performers are excluded from the crowd score by design.
        weighted_sum = 0.0
        weight_total = 0.0
        for d in current_frame_data:
            if d.get('performer'):
                continue
            if self.registered_only and int(d['id']) not in self.registered_ids:
                continue
            if d.get('gaze_scored'):
                w = 1.0
            elif self._buffer_ready(d['id']):
                w = d.get('buffer_fill', 0.0)
            else:
                continue          # no pose / not warmed up: unknown, not a zero
            d['scored'] = True
            weighted_sum += d['score'] * w
            weight_total += w
        # None (not 0.0) when nobody is scorable this frame, so a too-far or
        # empty crowd reads as "unknown" instead of dragging the average to zero.
        crowd_average = weighted_sum / weight_total if weight_total > 0 else None

        self._update_context_seconds()
            
        # 5. Publish to Redis (throttled to 1 Hz). Skip when crowd_average is
        # None (nobody scorable) so subscribers keep the last value instead of
        # receiving a misleading 0.
        current_time = time.time()
        if (self.redis_client and crowd_average is not None and self.publish_crowd
                and (current_time - self.last_publish_time) >= self.publish_interval):
            try:
                payload = f"{crowd_average:.4f}"
                self.redis_client.publish(REDIS_CHANNEL, payload)
                self.last_publish_time = current_time
                self._console_pub_count += 1
                if (self._console_pub_count <= self._console_pub_warmup
                        or (current_time - self._console_pub_last) >= self._console_pub_interval):
                    print(f"📡 Redis pub → {REDIS_CHANNEL}: {payload}")
                    self._console_pub_last = current_time
            except Exception as e:
                print(f"Redis Error: {e}")

        return current_frame_data, crowd_average

    def publish_individual(self, people_data):
        """Publish each registered participant's engagement on its own
        device:<serial>:engagement channel (Sowmya's physio namespace).
        confirmed=True is a live/coasting face ID (green in AR); confirmed=False
        is an appearance-inferred guess after track churn (red in AR)."""
        if self.redis_client is None:
            return
        now = time.time()
        if now - self.last_individual_publish_time < self.individual_publish_interval:
            return
        published = 0
        for p in people_data:
            serial = p.get('identified_as')
            if not serial or not p.get('scored'):
                continue
            src = p.get('id_source')
            payload = json.dumps({
                'device': serial,
                'engagement': round(float(p['score']), 4),
                'confirmed': src in ('face', 'coast'),
                'confidence': round(float(p.get('id_confidence') or 0.0), 3),
                'source': src,
                'timestamp': round(now, 3),
            })
            try:
                self.redis_client.publish(
                    INDIVIDUAL_CHANNEL_FMT.format(serial=serial), payload)
                published += 1
            except Exception as e:
                print(f"Redis Error (individual): {e}")
                break
        if published:
            self.last_individual_publish_time = now
            self._indiv_pub_count += 1
            if (self._indiv_pub_count <= self._console_pub_warmup
                    or (now - self._indiv_pub_last) >= self._console_pub_interval):
                print(f"📡 Redis pub → device:*:engagement ({published} registered)")
                self._indiv_pub_last = now

SIDEBAR_W = 280  # pixel width of the EmotiBit physio sidebar panel

# =============================================================================
# EMOTIBIT SIDEBAR
# =============================================================================

def draw_emotibit_sidebar(h, enrolled_names, emotibit_data, emotibit_lock,
                          focus_target, sidebar_radio_rects, sidebar_w,
                          reconnect_rect_out=None):
    """Return a (h, sidebar_w, 3) uint8 image for the physio sidebar.
    Mutates sidebar_radio_rects in place with (y_top, y_bot, serial) tuples.
    If reconnect_rect_out is given, it is filled with a single
    (x0, y0, x1, y1) tuple (sidebar-local coords) for the reconnect button."""
    sidebar_radio_rects.clear()

    # Snapshot connected device serials once under the lock
    with emotibit_lock:
        connected_serials = list(emotibit_data.keys())

    # Build unified row list: enrolled devices first (with radio button),
    # then any other connected-but-unassigned devices (plotted, no radio).
    # Only devices that have actually published data THIS session get a row:
    # face enrollments persist across sessions (.face_enrollments/), so an
    # enrolled-but-absent EmotiBit would otherwise draw a blank plot.
    enrolled_set = set(enrolled_names)
    connected_set = set(connected_serials)
    enrolled_connected = [s for s in enrolled_names if s in connected_set]
    unassigned = [s for s in connected_serials if s not in enrolled_set]
    rows = ([(s, True) for s in enrolled_connected]
            + [(s, False) for s in unassigned])
    n = len(rows)

    # Column layout scales with wearer count so up to ~12 EmotiBits stay
    # legible: 1 col ≤4, 2 cols ≤8, 3 cols beyond (concerts can run 11+ infant/
    # parent pairs). The image widens to sidebar_w * cols so each panel keeps
    # its width.
    cols = 3 if n > 8 else (2 if n > 4 else 1)
    total_w = sidebar_w * cols
    sidebar = np.full((h, total_w, 3), 28, dtype=np.uint8)
    cv2.putText(sidebar, "EmotiBit", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1, cv2.LINE_AA)

    # Reconnect button: a fixed strip at the bottom of the sidebar. Clicking it
    # tells the EmotiBit publisher to re-run discovery without restarting the
    # whole pipeline (see on_mouse). Drawn last (on top) but its geometry is
    # reserved up-front so device rows never overlap it.
    _BTN_H = 34
    _BTN_M = 8
    _btn_y0 = h - _BTN_H - _BTN_M
    _btn_y1 = h - _BTN_M
    _btn_x0 = 6
    _btn_x1 = total_w - 6

    def _draw_reconnect_button():
        cv2.rectangle(sidebar, (_btn_x0, _btn_y0), (_btn_x1, _btn_y1), (40, 95, 40), -1)
        cv2.rectangle(sidebar, (_btn_x0, _btn_y0), (_btn_x1, _btn_y1), (80, 210, 80), 1)
        cv2.putText(sidebar, "RECONNECT", (_btn_x0 + 12, _btn_y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 255, 190), 1, cv2.LINE_AA)
        cv2.putText(sidebar, "click to re-scan EmotiBits", (_btn_x0 + 12, _btn_y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140, 200, 140), 1, cv2.LINE_AA)
        if reconnect_rect_out is not None:
            reconnect_rect_out[:] = [(_btn_x0, _btn_y0, _btn_x1, _btn_y1)]

    if not rows:
        cv2.putText(sidebar, "No EmotiBits", (8, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(sidebar, "connected", (8, 73),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(sidebar, "(start EmotiBit publisher)", (8, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70, 70, 70), 1, cv2.LINE_AA)
        _draw_reconnect_button()
        return sidebar

    # Column layout: rows_per_col / row height derive from cols decided above.
    col_w = sidebar_w
    rows_per_col = (n + cols - 1) // cols
    available_h = h - 28 - (_BTN_H + 2 * _BTN_M)
    row_h = max(52, min(140, available_h // max(rows_per_col, 1)))
    for i, (serial, is_enrolled) in enumerate(rows):
        col = i // rows_per_col
        row_in_col = i % rows_per_col
        x_off = col * col_w
        y0 = 28 + row_in_col * row_h
        y1 = min(y0 + row_h - 2, h - 2)
        # Only enrolled rows are clickable focus targets
        if is_enrolled:
            sidebar_radio_rects.append((y0, y1, serial))
        selected = is_enrolled and (focus_target == serial)
        if selected:
            cv2.rectangle(sidebar, (x_off + 2, y0), (x_off + col_w - 2, y1), (45, 28, 45), -1)
        # Radio button (enrolled only — unassigned rows show a dim dash)
        rb_cx, rb_cy = x_off + 11, y0 + 12
        if is_enrolled:
            if selected:
                cv2.circle(sidebar, (rb_cx, rb_cy), 6, (255, 0, 255), -1)
            else:
                cv2.circle(sidebar, (rb_cx, rb_cy), 6, (140, 140, 140), 1)
        else:
            cv2.line(sidebar, (rb_cx - 5, rb_cy), (rb_cx + 5, rb_cy),
                     (80, 80, 80), 1, cv2.LINE_AA)
        # Serial label — truncate long IDs to fit. Unassigned dimmed.
        # Last 5 chars is enough to disambiguate wearers in operator view; the
        # full ID is in the CSV. Keeps the row header clear for STROC / HRSD.
        label = serial if len(serial) <= 5 else serial[-5:]
        label_col = (220, 220, 220) if is_enrolled else (150, 150, 150)
        cv2.putText(sidebar, f"#{label}", (x_off + 22, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, label_col, 1, cv2.LINE_AA)
        if not is_enrolled:
            cv2.putText(sidebar, "unassigned", (x_off + 22, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (110, 110, 110), 1, cv2.LINE_AA)
        # ------ Snapshot subscriber state under lock ------
        with emotibit_lock:
            d = emotibit_data.get(serial, {})
            snap_eda_z   = list(d.get('eda_z',  []))
            snap_hr_z    = list(d.get('hr_z',   []))
            snap_stroc_z = list(d.get('temperature_roc_z', []))
            snap_metrics = dict(d.get('metrics', {}))

        # ------ Unified per-wearer physio panel ------
        # All three traces are unitless per-wearer session-baseline z-scores
        # (this wearer's deviation from their own mean, in their own SD units),
        # so plotting HR / EDA / STROC on a single +-3 SD axis is meaningful.
        # The header shows live numeric values colour-keyed to each spline.
        # Trace colour DARKENS as |z| grows (more saturated near +-2 SD,
        # paler near 0) so the eye is drawn to excursions without losing the
        # quieter traces. Off-wrist suppresses everything and watermarks the
        # panel rather than faking a zero baseline reading.
        Z_SOFT = 1.5
        Z_HARD = 2.0
        Z_SAT  = 3.0  # axis saturation = full +-3 SD
        # Publisher emits one physio_metrics every 1.0s, so 10 samples == 10s window.
        PLOT_WINDOW_SAMPLES = 10
        # Base hues (BGR). Light variants are used near 0; full saturation past |z|=2.
        # NB: 'TEMP' label refers to skin-temp rate-of-change (temperature_roc_z)
        # -- shorter than STROC and matches operator vocabulary.
        TRACES = (
            # (label, source_list, last_metric_key, light_col, full_col)
            ('HR',   snap_hr_z,    'hr_z',              (180, 235, 180), ( 70, 220,  70)),
            ('EDA',  snap_eda_z,   'eda_z',             (220, 230, 190), (255, 200,  60)),
            ('TEMP', snap_stroc_z, 'temperature_roc_z', (220, 200, 230), (200, 100, 200)),
        )

        is_calibrating = bool(snap_metrics.get('calibrating', False))
        is_off_wrist   = bool(snap_metrics.get('off_wrist', False))

        # ----- Header: three small colour-keyed readouts, right-aligned.
        # Label (HR / EDA / TEMP) + number; colour matches the spline so it
        # doubles as the in-panel legend. When |z| >= Z_HARD the readout is
        # promoted: thicker stroke + a faint filled pill behind it, so the
        # operator's eye is pulled to the wearer even out of the corner of
        # the screen.
        header_y = y0 + 16
        cur_x    = x_off + col_w - 6
        font_sc  = 0.34
        for label, _hist, key, _light, full in reversed(TRACES):
            val = snap_metrics.get(key)
            if val is None:
                txt = f"{label} --"
                col = (80, 80, 80)
                hot = False
            else:
                txt = f"{label} {val:+.2f}"
                col = full
                hot = abs(val) >= Z_HARD
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_sc, 1)
            tx = max(x_off + 4, cur_x - tw)
            if hot:
                # Translucent pill in the trace's full colour at ~22% opacity.
                pad_x, pad_y = 3, 2
                px0, py0 = tx - pad_x, header_y - th - pad_y
                px1, py1 = tx + tw + pad_x, header_y + pad_y
                overlay = sidebar.copy()
                cv2.rectangle(overlay, (px0, py0), (px1, py1), col, -1)
                cv2.addWeighted(overlay, 0.22, sidebar, 0.78, 0, sidebar)
            cv2.putText(sidebar, txt, (tx, header_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_sc, col,
                        2 if hot else 1, cv2.LINE_AA)
            cur_x -= tw + 8

        # ----- Plot panel: fixed +-Z_SAT axis below the header.
        plot_x  = x_off + 4
        plot_y  = y0 + 24
        plot_pw = col_w - 8
        plot_ph = y1 - plot_y - 3
        if plot_ph <= 12:
            if i < n - 1:
                cv2.line(sidebar, (x_off + 4, y1), (x_off + col_w - 4, y1), (50, 50, 50), 1)
            continue

        cv2.rectangle(sidebar, (plot_x, plot_y),
                      (plot_x + plot_pw, plot_y + plot_ph), (42, 42, 42), -1)

        # Axis grid: zero baseline + +-Z_HARD reference lines only.
        # (Previously also drew +-Z_SOFT in grey -- removed as visual clutter;
        # the colour-saturation lerp already conveys 'getting noteworthy' and
        # the red-tinted +-2 lines mark the alert threshold cleanly.)
        rng    = 2.0 * Z_SAT
        def _y_for(z):
            return int(plot_y + plot_ph - 2 - ((float(z) + Z_SAT) / rng * (plot_ph - 4)))
        for z_ref, col_ref in ((0.0, (90, 90, 90)),
                               (+Z_HARD, (60, 60, 120)), (-Z_HARD, (60, 60, 120))):
            yref = _y_for(z_ref)
            cv2.line(sidebar, (plot_x + 1, yref), (plot_x + plot_pw - 2, yref),
                     col_ref, 1, cv2.LINE_AA)

        # Axis labels (top / mid / bottom) -- only place "SD" appears.
        lbl_col = (75, 120, 150)
        cv2.putText(sidebar, f"+{Z_SAT:.0f} SD", (plot_x + 2, plot_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, lbl_col, 1, cv2.LINE_AA)
        cv2.putText(sidebar, f"-{Z_SAT:.0f} SD",
                    (plot_x + 2, plot_y + plot_ph - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, lbl_col, 1, cv2.LINE_AA)

        if is_calibrating and not any((snap_hr_z, snap_eda_z, snap_stroc_z)):
            remaining = snap_metrics.get('calibration_remaining_s')
            msg = f"calibrating {remaining:.0f}s" if remaining is not None else "calibrating..."
            cv2.putText(sidebar, msg,
                        (plot_x + 4, plot_y + plot_ph // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110, 160, 110), 1, cv2.LINE_AA)
        else:
            any_drawn = False
            for label, hist, _key, light, full in TRACES:
                if len(hist) < 2:
                    continue
                any_drawn = True
                # Rolling 10s window: only show the tail. Older samples stay in
                # the deque (handy for debugging) but the plot stops compressing
                # an ever-growing history into the same panel width.
                tail = list(hist)[-PLOT_WINDOW_SAMPLES:]
                if len(tail) < 2:
                    continue
                arr = np.clip(np.array(tail, dtype=np.float32), -Z_SAT, Z_SAT)
                xs = np.linspace(plot_x + 1, plot_x + plot_pw - 2,
                                 len(arr)).round().astype(np.int32)
                ys = np.array([_y_for(v) for v in arr], dtype=np.int32)
                abs_z = np.abs(arr)
                # Per-segment colour: lerp light -> full as |z| goes 0 -> Z_HARD,
                # then clamp at full beyond Z_HARD. Vectorised would be nicer but
                # the per-trace length is bounded (deque maxlen=180) so this loop
                # is cheap enough.
                light_arr = np.array(light, dtype=np.float32)
                full_arr  = np.array(full,  dtype=np.float32)
                for k in range(len(xs) - 1):
                    zmax = float(max(abs_z[k], abs_z[k + 1]))
                    t = min(1.0, zmax / Z_HARD)
                    seg_col = tuple(int(c) for c in (light_arr * (1.0 - t) + full_arr * t))
                    # Thicker stroke once the segment crosses the alert band:
                    # the colour lerp alone is too subtle on a small panel.
                    seg_thick = 2 if zmax >= Z_HARD else 1
                    cv2.line(sidebar, (int(xs[k]), int(ys[k])),
                             (int(xs[k + 1]), int(ys[k + 1])),
                             seg_col, seg_thick, cv2.LINE_AA)
                # End-of-trace marker: small filled dot at the latest sample,
                # growing from r=2 to r=4 as |z| climbs to Z_HARD, then a thin
                # bright halo if we're in the alert band. Gives a stable focal
                # point for the eye and makes high-|z| wearers 'pop' from a
                # multi-row sidebar.
                last_abs = float(abs_z[-1])
                t_last   = min(1.0, last_abs / Z_HARD)
                dot_r    = int(round(2 + 2 * t_last))
                dot_col  = tuple(int(c) for c in (light_arr * (1.0 - t_last) + full_arr * t_last))
                cv2.circle(sidebar, (int(xs[-1]), int(ys[-1])), dot_r,
                           dot_col, -1, cv2.LINE_AA)
                if last_abs >= Z_HARD:
                    cv2.circle(sidebar, (int(xs[-1]), int(ys[-1])),
                               dot_r + 2, (255, 255, 255), 1, cv2.LINE_AA)
            if not any_drawn:
                cv2.putText(sidebar, "Physio: no signal",
                            (plot_x + 4, plot_y + plot_ph // 2 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (65, 65, 65), 1, cv2.LINE_AA)

        # Off-wrist watermark: drawn over everything so it's unambiguous that
        # the panel content is stale / suppressed and not a real "near baseline"
        # reading. The publisher already drops *_z keys when off_wrist=True, so
        # the splines will gap out within a few render frames as their deques
        # stop receiving samples; this label explains *why*.
        if is_off_wrist:
            wm = "OFF-WRIST"
            (ww, wh), _ = cv2.getTextSize(wm, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            wx = plot_x + (plot_pw - ww) // 2
            wy = plot_y + (plot_ph + wh) // 2
            # Faint dim overlay so traces are still visible but clearly subdued.
            overlay = sidebar.copy()
            cv2.rectangle(overlay, (plot_x, plot_y),
                          (plot_x + plot_pw, plot_y + plot_ph), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, sidebar, 0.55, 0, sidebar)
            cv2.putText(sidebar, wm, (wx, wy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 220), 1, cv2.LINE_AA)
        if i < n - 1:
            cv2.line(sidebar, (x_off + 4, y1), (x_off + col_w - 4, y1), (50, 50, 50), 1)
    _draw_reconnect_button()
    return sidebar


# =============================================================================
# DUAL-CAMERA SUPPORT
# =============================================================================

class CameraContext:
    """All per-camera pipeline state for one feed. Everything here is private
    to a single camera so two contexts can run concurrently (one worker thread
    each) without sharing mutable state. The face-enrollment store, EmotiBit
    sidebar and Redis publishing stay global and are merged/handled once on the
    main thread after both contexts finish a frame."""

    def __init__(self, index, cap, system, name=None):
        self.index = index
        self.cap = cap
        self.system = system
        self.name = name or f"CAM{index}"
        # Per-camera face-ID state (track ids are per-tracker, so these must NOT
        # be shared between cameras or a serial would flip between feeds).
        self.face_id_counter = 0
        self.face_id_cache = {}     # {track_id: {'name','similarity','miss','inferred'}}
        self.face_id_pending = {}   # {track_id: {'name','count'}}
        self.registered_profiles = {}  # {serial: {'sig','centre','seen'}}
        self.sensor_sim_ids = set()  # --max-people sticky registered ids
        # Latest frame read for this camera + results produced by the worker
        self._frame = None
        self.display_frame = None
        self.people_data = []
        self.crowd_avg = None


def merge_people_by_serial(context_people, fused_scores=None, ema_alpha=0.5):
    """Merge people lists from all cameras, de-duplicated by EmotiBit serial,
    and FUSE each participant's engagement into ONE canonical score.

    The same participant may appear on both feeds. Each camera runs its own
    tracker, temporal buffer and transformer pass, so it produces an
    INDEPENDENT engagement score for that person. Picking one feed's raw score
    (the old behaviour) makes the published/displayed value jump between feeds
    as the "winner" flips frame to frame. Instead we combine every scored
    sighting of a serial into a single reliability-weighted score — a fuller
    temporal buffer and a more confident identity count for more — then
    EMA-smooth it against the previous frame in ``fused_scores``. The result:
    one registered person yields exactly one stable engagement score, shown
    identically on both overlays and published once to Redis.

    Identity metadata (bbox, face size, confidence) still comes from the single
    best sighting so the highlight/label track the clearest view. Unidentified
    people are kept as-is (they never collide across cameras, not published).

    Returns (merged_people, crowd_average). Mutates ``fused_scores`` in place
    and rewrites the winning person dict's 'score' to the fused value.
    """
    if fused_scores is None:
        fused_scores = {}
    best_by_serial = {}
    sightings_by_serial = {}
    unidentified = []

    def _rank(p):
        # (confirmed?, face_px, id_confidence) — higher is better
        confirmed = 0 if p.get('id_source') == 'inferred' else 1
        return (confirmed, float(p.get('face_px') or 0.0), float(p.get('id_confidence') or 0.0))

    for people in context_people:
        for p in people:
            serial = p.get('identified_as')
            if not serial:
                unidentified.append(p)
                continue
            cur = best_by_serial.get(serial)
            if cur is None or _rank(p) > _rank(cur):
                best_by_serial[serial] = p
            if p.get('scored'):
                sightings_by_serial.setdefault(serial, []).append(p)

    # Fuse each serial's scored sightings into one canonical, smoothed score and
    # stamp it onto that serial's representative dict (used for publishing).
    for serial, rep in best_by_serial.items():
        sightings = sightings_by_serial.get(serial)
        if not sightings:
            continue  # nobody scorable this frame → keep the prior fused value
        w_sum = 0.0
        s_sum = 0.0
        for p in sightings:
            w = (max(0.05, float(p.get('buffer_fill', 1.0) or 0.0))
                 * max(0.1, float(p.get('id_confidence') or 0.0)))
            s_sum += w * float(p['score'])
            w_sum += w
        new_score = s_sum / w_sum if w_sum > 0 else float(rep['score'])
        prev = fused_scores.get(serial)
        fused = new_score if prev is None else ema_alpha * new_score + (1.0 - ema_alpha) * prev
        fused_scores[serial] = fused
        rep['score'] = fused
        rep['scored'] = True

    merged = list(best_by_serial.values()) + unidentified

    # Crowd average over the unique fused per-serial scores (each person once,
    # so a seam duplicate can't drag or inflate the average).
    scored = [fused_scores[s] for s in best_by_serial if s in sightings_by_serial]
    # Also count unidentified scored people (only relevant when face ID is off).
    scored += [p['score'] for p in unidentified if p.get('scored')]
    crowd_average = (sum(scored) / len(scored)) if scored else None
    return merged, crowd_average


# --- Face-ID + identity re-binding tuning (module level so the per-camera
# pipeline function and the main-loop key handlers share one definition) ---
FACE_ID_INTERVAL = 10   # frames between face matching runs
FACE_MIN_MATCH_PX = 16  # below this native face width, coast (never assign/evict)
FACE_TRUST_PX = 40      # at/above this, a single frame is trustworthy
FACE_VOTES_FAR = 2      # consistent frames required between MIN and TRUST
FACE_EVICT_MISSES = 6   # strong contradictions before dropping a cached identity
FACE_MISS_SIM = 0.25    # similarity below which a clear face is "someone else"
REBIND_SIG_MIN = 0.85   # appearance-only correlation to accept a re-bind guess
REBIND_SIG_POS = 0.70   # looser floor when last-known position corroborates
REBIND_POS_FRAC = 0.20  # position radius as a fraction of the frame diagonal
REBIND_MAX_GAP_S = 5.0  # ignore last-known positions older than this (s)
ENROLL_POSES = ['front', 'left', 'right']  # guided 3-pose capture order


def process_and_annotate(ctx, frame, face_id, face_id_lock, focus_target,
                         enroll_session, args, is_360=False, fused_scores=None):
    """Run one camera's full 2D pipeline and draw its overlays onto its own
    frame. All per-camera state lives on ``ctx`` so two contexts can run in
    parallel worker threads. The only shared resource touched is ``face_id``
    (read-only match), guarded by ``face_id_lock``. Publishing, the crowd bar,
    the physio sidebar, the window and key handling stay on the main thread and
    run once on the merged result.

    ``fused_scores`` (serial -> smoothed cross-camera engagement) is the shared
    dual-camera fusion state from the previous frame. When a drawn person is
    identified as a serial that already has a fused value, that value is shown
    (and colours the box) instead of this camera's local score, so a person in
    both feeds reads the SAME engagement number on both overlays. Empty/None in
    single-camera mode, where the local score is already the only score.

    Returns (display_frame, people_data, crowd_avg) and also stashes them on
    ``ctx`` for the caller.
    """
    system = ctx.system
    people_data, crowd_avg = system.process_frame(frame)
    display_frame = frame.copy()

    # =====================================================================
    # FACE IDENTIFICATION — match tracked people against enrolled faces.
    # (Identical policy to the single-camera build, but keyed on this
    # camera's own caches so track ids never collide across feeds.)
    # =====================================================================
    if face_id and face_id.has_enrollments and not is_360:
        face_id_cache = ctx.face_id_cache
        face_id_pending = ctx.face_id_pending
        registered_profiles = ctx.registered_profiles
        ctx.face_id_counter += 1
        run_face_id = (ctx.face_id_counter % FACE_ID_INTERVAL == 0)
        now_t = time.time()
        newly_face_matched = set()

        for person in people_data:
            tid = person['id']
            if run_face_id:
                hc = head_crop(frame, person['bbox'])
                with face_id_lock:
                    res = face_id.identify_crop(hc) if hc is not None else None
                if res is None:
                    # Close-range framing: retry on the full box when the head
                    # crop clamps off the face at the frame edge.
                    x1, y1, x2, y2 = person['bbox']
                    fh, fw = frame.shape[:2]
                    cx1, cy1 = max(0, x1), max(0, y1)
                    cx2, cy2 = min(fw, x2), min(fh, y2)
                    if cx2 > cx1 and cy2 > cy1:
                        with face_id_lock:
                            res = face_id.identify_crop(frame[cy1:cy2, cx1:cx2])
                cached = face_id_cache.get(tid)
                if isinstance(res, dict):
                    face_px = res.get('face_px')
                    if face_px is not None:
                        person['face_px'] = face_px
                    name = res.get('name')
                    trusted = face_px is not None and face_px >= FACE_TRUST_PX
                    matchable = face_px is None or face_px >= FACE_MIN_MATCH_PX
                    if name and matchable:
                        if cached and cached['name'] == name:
                            cached['similarity'] = res['similarity']
                            cached['miss'] = 0
                            cached.pop('inferred', None)
                            face_id_pending.pop(tid, None)
                            newly_face_matched.add(tid)
                        else:
                            need = 1 if trusted else FACE_VOTES_FAR
                            pend = face_id_pending.get(tid)
                            if pend and pend['name'] == name:
                                pend['count'] += 1
                            else:
                                pend = {'name': name, 'count': 1}
                                face_id_pending[tid] = pend
                            if pend['count'] >= need:
                                face_id_cache[tid] = {'name': name,
                                                      'similarity': res['similarity'],
                                                      'miss': 0}
                                face_id_pending.pop(tid, None)
                                newly_face_matched.add(tid)
                                log.info(f"[{ctx.name}] face-id: track {tid} = {name} "
                                         f"(sim={res['similarity']:.3f}, face={face_px:.0f}px)")
                    elif cached and trusted and res['similarity'] < FACE_MISS_SIM:
                        cached['miss'] = cached.get('miss', 0) + 1
                        if cached['miss'] >= FACE_EVICT_MISSES:
                            log.info(f"[{ctx.name}] face-id: track {tid} evicted "
                                     f"(was {cached['name']}, {cached['miss']} misses, "
                                     f"sim={res['similarity']:.3f})")
                            face_id_cache.pop(tid, None)
                            face_id_pending.pop(tid, None)
                    elif name is None and trusted and not cached:
                        if (ctx.face_id_counter // FACE_ID_INTERVAL) % 10 == 0:
                            log.info(f"[{ctx.name}] face-id: track {tid} unmatched "
                                     f"(best sim={res['similarity']:.3f}, face={face_px:.0f}px)")
                elif res is not None:
                    face_id_cache[tid] = {'name': res['name'],
                                          'similarity': res['similarity'],
                                          'miss': 0}
                    newly_face_matched.add(tid)

            # Apply cached result
            cached = face_id_cache.get(tid)
            if cached:
                serial = cached['name']
                person['identified_as'] = serial
                person['face_similarity'] = cached['similarity']
                if cached.get('inferred'):
                    person['id_source'] = 'inferred'
                elif tid in newly_face_matched:
                    person['id_source'] = 'face'
                else:
                    person['id_source'] = 'coast'
                person['id_confidence'] = cached['similarity']
                prof = registered_profiles.setdefault(serial, {})
                px1, py1, px2, py2 = person['bbox']
                prof['centre'] = (0.5 * (px1 + px2), 0.5 * (py1 + py2))
                prof['seen'] = now_t
                if tid in newly_face_matched:
                    _sig = appearance_sig(frame, person['bbox'])
                    if _sig is not None:
                        _old = prof.get('sig')
                        prof['sig'] = _sig if _old is None else cv2.addWeighted(_old, 0.8, _sig, 0.2, 0)

        # One live track per name: duplicate claims keep the highest similarity
        if run_face_id:
            live_tids = {p['id'] for p in people_data}
            by_name = {}
            for tid, entry in face_id_cache.items():
                if tid in live_tids:
                    by_name.setdefault(entry['name'], []).append((tid, entry))
            for name, claims in by_name.items():
                if len(claims) > 1:
                    claims.sort(key=lambda te: (0 if te[1].get('inferred') else 1,
                                                te[1]['similarity']), reverse=True)
                    for tid, _ in claims[1:]:
                        face_id_cache.pop(tid, None)
                        for person in people_data:
                            if person['id'] == tid:
                                person.pop('identified_as', None)
                                person.pop('face_similarity', None)
                                person.pop('id_source', None)
                                person.pop('id_confidence', None)

        # Re-bind churned tracks to a registered participant by appearance
        if registered_profiles:
            fh, fw = frame.shape[:2]
            radius = REBIND_POS_FRAC * math.hypot(fw, fh)
            live_tids = {p['id'] for p in people_data}
            claimed = {e['name'] for t, e in face_id_cache.items() if t in live_tids}
            for person in people_data:
                tid = person['id']
                if tid in face_id_cache or person.get('performer'):
                    continue
                sig = appearance_sig(frame, person['bbox'])
                if sig is None:
                    continue
                bx1, by1, bx2, by2 = person['bbox']
                centre = (0.5 * (bx1 + bx2), 0.5 * (by1 + by2))
                best_serial, best_c = None, 0.0
                for serial, prof in registered_profiles.items():
                    if serial in claimed:
                        continue
                    ref = prof.get('sig')
                    if ref is None:
                        continue
                    c = float(cv2.compareHist(sig, ref, cv2.HISTCMP_CORREL))
                    pc = prof.get('centre')
                    near = (pc is not None
                            and (now_t - prof.get('seen', 0.0)) <= REBIND_MAX_GAP_S
                            and math.hypot(centre[0] - pc[0], centre[1] - pc[1]) <= radius)
                    floor = REBIND_SIG_POS if near else REBIND_SIG_MIN
                    if c >= floor and c > best_c:
                        best_serial, best_c = serial, c
                if best_serial is not None:
                    face_id_cache[tid] = {'name': best_serial, 'similarity': best_c,
                                          'miss': 0, 'inferred': True}
                    person['identified_as'] = best_serial
                    person['face_similarity'] = best_c
                    person['id_source'] = 'inferred'
                    person['id_confidence'] = best_c
                    claimed.add(best_serial)
                    log.info(f"[{ctx.name}] re-bind: track {tid} ~= {best_serial} "
                             f"(corr={best_c:.2f}, inferred)")

    # ---- 3-pose enrollment banner (only on the active enrollment camera) ----
    if (enroll_session and face_id and not is_360
            and enroll_session.get('cam', 0) == ctx.index):
        es = enroll_session
        target = next((p for p in people_data if p['id'] == es['tid']), None)
        if target is None and people_data:
            target = max(people_data,
                         key=lambda p: (p['bbox'][2]-p['bbox'][0]) * (p['bbox'][3]-p['bbox'][1]))
            es['tid'] = target['id']
        es['target_bbox'] = target['bbox'] if target is not None else None
        pose = ENROLL_POSES[es['idx']]
        if target is None:
            banner = f"Enroll {es['serial']}: step into frame  (Esc=cancel)"
        else:
            banner = (f"Enroll {es['serial']} - face {pose.upper()} and press SPACE "
                      f"({es['idx']+1}/3, Esc=cancel)")
        (btw, _bth), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        bx = max(10, (display_frame.shape[1] - btw) // 2)
        cv2.putText(display_frame, banner, (bx, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(display_frame, banner, (bx, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, IDENTIFIED_COLOR, 2)
        if target is not None:
            tx1, ty1, tx2, ty2 = target['bbox']
            cv2.rectangle(display_frame, (tx1, ty1), (tx2, ty2), IDENTIFIED_COLOR, 2)

    # ---- Registered-only gating (per camera) ----
    if face_id and not is_360:
        if not face_id.has_enrollments and ctx.face_id_cache:
            ctx.face_id_cache.clear()
            ctx.face_id_pending.clear()
            ctx.registered_profiles.clear()
        system.registered_only = True
        system.registered_ids = {int(tid) for tid in ctx.face_id_cache}
        system.exempt_registered = True
    elif args.max_people and not is_360:
        performers = system.gaze_engine.performers
        ctx.sensor_sim_ids.difference_update(performers)
        for p in people_data:
            if len(ctx.sensor_sim_ids) >= args.max_people:
                break
            pid = int(p['id'])
            if pid in performers or p.get('performer'):
                continue
            ctx.sensor_sim_ids.add(pid)
        system.registered_only = True
        system.registered_ids = ctx.sensor_sim_ids
        system.exempt_registered = False
    else:
        system.registered_only = False
        system.registered_ids = set()

    # ---- Draw gaze rays + person boxes (2D only) ----
    if not is_360:
        # Scale overlay text/line weight with capture resolution so labels stay
        # legible at 1080p/4K (a fixed 0.5 scale is unreadable once a 4K feed is
        # shrunk into the window). Baselined at 720p.
        _fs = max(0.5, display_frame.shape[0] / 720.0)
        _ft = max(1, int(round(_fs)))
        _yo = int(10 * _fs)
        _gres = system.last_gaze_result
        if _gres is not None:
            _perf_boxes = [p['bbox'] for p in people_data if p.get('performer')]
            _ray_people = {int(p['id']): {'geom': p['gaze_geom']}
                           for p in people_data if p.get('gaze_geom') is not None}
            draw_gaze_rays(display_frame, _ray_people, _gres['focal'],
                           _perf_boxes, system.gaze_engine.performers)
        for person in people_data:
            x1, y1, x2, y2 = person['bbox']
            score = person['score']
            pid = person['id']
            bf = person.get('buffer_fill', 1.0)
            identified = person.get('identified_as')
            if person.get('performer'):
                continue
            # Dual-camera: show the ONE fused engagement for this serial so the
            # same person reads identically on both feeds (falls back to the
            # local score in single-camera mode or before the first fusion).
            if fused_scores and identified in fused_scores:
                score = fused_scores[identified]
            if system.registered_only and not identified and int(pid) not in system.registered_ids:
                continue
            fresh_mp = person.get('keypoints') is not None
            g_reason = person.get('gaze_reason', '')
            gaze_tag = f" [{g_reason}]" if g_reason in ('shift', 'off-focal') else ""
            _fpx = person.get('face_px')
            face_tag = f" f{_fpx:.0f}px" if _fpx else ""
            show_highlight = (identified and focus_target == identified)
            if show_highlight:
                color = IDENTIFIED_COLOR
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, _ft + 2)
                _sid = _short_identity(identified)
                if person.get('id_source') == 'inferred':
                    _sid = f"~{_sid}?"
                name_label = f"{_sid} {score:.0%}{gaze_tag}{face_tag}"
                cv2.putText(display_frame, name_label, (x1, y1 - _yo),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * _fs, color, _ft + 1)
            else:
                color = (0, int(255 * score), int(255 * (1-score)))
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, _ft + 1)
                if identified:
                    _sid = _short_identity(identified)
                    id_str = f"~{_sid}?" if person.get('id_source') == 'inferred' else _sid
                else:
                    id_str = system.display_label(pid)
                if bf < 0.9:
                    label = f"{id_str} {score:.0%}{gaze_tag}{face_tag} conf:{bf:.0%}"
                    cv2.putText(display_frame, label, (x1, y1 - _yo),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * _fs, color, _ft)
                    bar_w_px = x2 - x1
                    bar_y_top = y2 + 2
                    bar_y_bot = y2 + 2 + 4 * _ft
                    cv2.rectangle(display_frame, (x1, bar_y_top), (x2, bar_y_bot), (50, 50, 50), -1)
                    cv2.rectangle(display_frame, (x1, bar_y_top), (x1 + int(bar_w_px * bf), bar_y_bot), (255, 200, 0), -1)
                else:
                    cv2.putText(display_frame, f"{id_str} {score:.0%}{gaze_tag}{face_tag}", (x1, y1 - _yo),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * _fs, color, _ft)
            if fresh_mp:
                cv2.circle(display_frame, (x2 - 5, y1 + 5), 2 * _ft, (0, 255, 255), -1)

    ctx.display_frame = display_frame
    ctx.people_data = people_data
    ctx.crowd_avg = crowd_avg
    return display_frame, people_data, crowd_avg


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Person Engagement Detection with Redis Streaming')
    parser.add_argument('--camera', type=int, default=None,
                        help='Force specific camera index (skips auto-detection)')
    parser.add_argument('--camera2', type=int, default=None,
                        help='Second camera index — enables DUAL-CAMERA mode (2D only). '
                             'Each camera runs its own tracker/pipeline; people are merged '
                             'by EmotiBit serial so the same person seen on both feeds is '
                             'never double-counted. Position the two cameras to cover each '
                             'half of the audience with minimal overlap.')
    parser.add_argument('--select-camera', '-s', action='store_true', default=False,
                        help='Always show the camera-selection preview (ignores last-used cache). '
                             'Useful when an external webcam is plugged in and the user wants '
                             'to pick it instead of the built-in laptop camera.')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file (overrides --camera)')
    parser.add_argument('--start-time', type=str, default=None,
                        help='Seek video files to this position before processing '
                             '(SS, MM:SS or HH:MM:SS). Ignored for live cameras.')
    parser.add_argument('--max-people', type=int, default=None,
                        help='Simulate a fixed sensor budget: the first N tracked people '
                             'become the registered set for the whole session; everyone '
                             'else is an unscored bystander. (Concert: 8 EmotiBits.)')
    parser.add_argument('--redis-host', type=str,
                        default=os.environ.get('REDIS_HOST', 'auto'),
                        help='Redis server host/IP. Default "auto" finds the broker over the '
                             'network via mDNS (hub must run 1_START_REDIS.bat, which also '
                             'advertises), then falls back to localhost. Pass an explicit IP '
                             'to skip discovery.')
    parser.add_argument('--redis-port', type=int, default=DEFAULT_REDIS_PORT,
                        help=f'Redis server port (default: {DEFAULT_REDIS_PORT})')
    parser.add_argument('--no-discover', action='store_true',
                        help='Disable mDNS auto-discovery; use localhost when no host is given.')
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
    
    # Resolve the Redis broker: explicit host wins; "auto" discovers the hub
    # over mDNS (advertiser runs alongside Redis on the hub) then falls back to
    # localhost, so a networked broker needs no hardcoded IP.
    if args.redis_host == 'auto':
        found = None
        if not args.no_discover:
            print("🔎 Looking for the Redis broker on the network (mDNS)...")
            try:
                from redis_discovery import discover_redis_broker
                found = discover_redis_broker(timeout=5.0)
            except Exception as e:
                print(f"   Discovery unavailable ({e}); falling back to localhost.")
        if found:
            args.redis_host, args.redis_port = found
            print(f"✅ Discovered Redis broker at {args.redis_host}:{args.redis_port}")
        else:
            args.redis_host = 'localhost'
            if not args.no_discover:
                print("ℹ️  No advertised broker found; using localhost "
                      "(pass --redis-host <ip> to target another device).")

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
    mirror_console_to_log()  # tee console output into the session log
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
            face_id = create_face_identifier(actual_device)
        except Exception as e:
            print(f"⚠️  FaceIdentifier init failed: {e}  — running without face ID")
            face_id = None
    
    # Registration mode state
    registration_mode = False
    registration_flash_until = 0  # timestamp for on-screen flash message
    registration_flash_msg = ''
    # Reconnect button flash (separate from registration so they don't clobber)
    reconnect_flash_until = 0.0
    reconnect_flash_msg = ''
    
    # Face-ID throttle constants, per-track identity caches, long-range policy
    # thresholds and re-bind thresholds now live at module level and per camera
    # (CameraContext.face_id_cache / face_id_pending / registered_profiles /
    # sensor_sim_ids), so a second feed never shares identity state with the
    # first. The main loop only keeps the shared enrollment/focus UI state.

    # 3-pose enrollment session (guided FRONT/LEFT/RIGHT captures at ~1m).
    # Each pose is captured MANUALLY with SPACE so the subject has time to
    # turn their head; Esc cancels. None when idle. 'cam' pins the capture to
    # the active feed so the banner and SPACE use that context's frame.
    enroll_session = None
    
    # Focus mode: which identified person(s) get magenta highlight
    # Values: 'all' | 'none' | '<emotibit serial>'
    # Cycle with F key: all → none → serial1 → serial2 → ... → all
    focus_target = 'all'

    # EmotiBit physio data — populated by subscriber thread
    emotibit_data = {}        # {serial: {'eda': deque, 'hr': deque, 'metrics': dict}}
    emotibit_lock = threading.Lock()
    sidebar_radio_rects = []  # [(y_top, y_bot, serial), ...] updated each frame
    reconnect_btn_rect = []   # [(x0, y0, x1, y1)] sidebar-local, updated each frame
    video_w_box    = [0]      # video pixel width, set on first frame
    _mouse_cb_set  = [False]  # set mouse callback once after first imshow

    def on_mouse(event, x, y, flags, param):
        nonlocal focus_target, reconnect_flash_until, reconnect_flash_msg
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        vw = video_w_box[0]
        if vw == 0 or x < vw:   # click is inside the video, not the sidebar
            return
        lx = x - vw             # x relative to the sidebar's left edge
        # Reconnect button (sidebar-local coords) takes priority over radio rows.
        if reconnect_btn_rect:
            bx0, by0, bx1, by1 = reconnect_btn_rect[0]
            if bx0 <= lx <= bx1 and by0 <= y <= by1:
                ok = False
                try:
                    if system.redis_client is not None:
                        system.redis_client.publish('emotibit:reconnect', '1')
                        ok = True
                except Exception as e:
                    print(f"\u26a0\ufe0f  reconnect publish failed: {e}")
                reconnect_flash_msg = ('Reconnecting EmotiBits...' if ok
                                       else 'Redis unavailable - cannot reconnect')
                reconnect_flash_until = time.time() + 2.5
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
                                # Three z-score history buffers powering the unified
                                # ±3 SD spline panel. 180 samples ≈ 3 min at the
                                # publisher's 1Hz emit cadence for *_z values.
                                'eda_z':  deque(maxlen=180),
                                'hr_z':   deque(maxlen=180),
                                'temperature_roc_z': deque(maxlen=180),
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
                            # Per-wearer z-scores (post-calibration). The publisher omits any
                            # *_z key while calibrating OR while off-wrist, and omits
                            # individual ones whose underlying channel currently has no
                            # Welford samples. Mirror that: only stash a value when
                            # present, never carry a stale one.
                            calib = bool(data.get('calibrating', False))
                            emotibit_data[serial]['metrics']['calibrating'] = calib
                            emotibit_data[serial]['metrics']['off_wrist'] = bool(data.get('off_wrist', False))
                            if 'calibration_remaining_s' in data:
                                emotibit_data[serial]['metrics']['calibration_remaining_s'] = float(data['calibration_remaining_s'])
                            for zk in ('hr_z', 'eda_z', 'ibi_z', 'temperature_roc_z', 'scr_frequency_z'):
                                if zk in data and data[zk] is not None:
                                    try:
                                        emotibit_data[serial]['metrics'][zk] = float(data[zk])
                                    except (TypeError, ValueError):
                                        pass
                                else:
                                    # Drop a stale value once the publisher stops sending it,
                                    # so the GUI text falls back to '--' instead of freezing.
                                    emotibit_data[serial]['metrics'].pop(zk, None)
                            # Spline history: only push during a valid (non-calibrating,
                            # on-wrist) reading. The plot then naturally gaps when contact
                            # is lost rather than freezing on the last good value.
                            if not calib:
                                for zk, deque_key in (
                                    ('hr_z', 'hr_z'),
                                    ('eda_z', 'eda_z'),
                                    ('temperature_roc_z', 'temperature_roc_z'),
                                ):
                                    if zk in data and data[zk] is not None:
                                        try:
                                            emotibit_data[serial][deque_key].append(float(data[zk]))
                                        except (TypeError, ValueError):
                                            pass
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
    MSMF_CACHE_FILE = Path(__file__).parent / '.last_camera_msmf'
    cap = None
    video_source_name = ""
    
    def is_real_camera(cap, num_test_frames=3):
        """
        Check if a camera produces real, changing video content.
        Virtual cameras (OBS, NDI) often return blank or static frames.
        """
        # Warm-up: real webcams often return the same buffered frame on the
        # first few reads (and need a moment for auto-exposure to settle), so
        # discard some initial frames before sampling. Without this a genuine
        # camera gets misclassified as virtual/inactive by the frame-diff check.
        for _ in range(5):
            cap.read()

        frames = []
        for _ in range(num_test_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                return False, None
            frames.append(frame)
            # Give the sensor time to advance so successive frames differ.
            time.sleep(0.05)
        
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
                # DSHOW couldn't open this index at all — some cameras only
                # enumerate on MSMF, so fall back before giving up.
                c.release()
                if try_msmf_fallback:
                    m = cv2.VideoCapture(index)  # MSMF
                    if m.isOpened():
                        return m
                    m.release()
                return cv2.VideoCapture()  # closed cap — no camera at this index
        # Non-Windows: default backend
        return cv2.VideoCapture(index)

    def _open_camera_pinned(index, target_w, target_h):
        """Open one specific camera index at target MJPG resolution WITHOUT the
        cross-index MSMF scan _ensure_high_res_capture performs. Dual mode needs
        each feed pinned to its own device — otherwise both cameras' scans pick
        the same 'best' index and one panel just mirrors the other. Tries DSHOW
        then MSMF on the SAME index; accepts any real (even dim) frame so a dark
        venue camera isn't rejected. Returns a closed capture if none stream."""
        backends = ((cv2.CAP_DSHOW, cv2.CAP_MSMF)
                    if platform.system() == 'Windows' else (None,))
        for backend in backends:
            c = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
            if not c.isOpened():
                c.release()
                continue
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            # A camera on a contended USB bus routinely drops its first few reads
            # after a mode switch; retry briefly before rejecting the index so a
            # slow-to-wake second camera isn't misreported as absent.
            for _ in range(8):
                ret, frame = c.read()
                if ret and frame is not None:
                    return c
                time.sleep(0.03)
            c.release()
        return cv2.VideoCapture()  # closed — nothing streams at this index

    def _ensure_high_res_capture(cap, index, target_w=None, target_h=None):
        """Renegotiate the live camera to its highest usable resolution.

        A 7m face is ~24px at 1080p and ~10px at 480p — capture resolution is
        the distance budget, so by default we drive the camera to the top mode
        it actually supports rather than a fixed 1080p target. The ceiling is
        discovered by requesting an oversized frame with MJPG and reading back
        what the driver clamps to; capture then walks down a ladder from that
        ceiling to the first mode that still streams >=10fps (a 4K sensor that
        can only manage a few fps at its top mode steps down instead of
        collapsing to the 640x480 default).

        Windows needs a clean reopen with MJPG requested BEFORE the first read:
        both DSHOW and MSMF lock their format once streaming, and uncompressed
        YUY2 is USB-capped to ~1-5fps at high resolutions. Pass target_w/target_h
        to pin a specific mode instead of auto-selecting the ceiling. 360°
        (equirect) cameras keep their native mode."""
        ret, probe = cap.read()
        if not ret or probe is None:
            return cap
        h, w = probe.shape[:2]
        if 1.9 <= w / h <= 2.1:
            return cap  # 360° camera — leave the native equirect stream alone

        def _measure_fps(c, secs=0.7):
            t0 = time.time()
            n = 0
            while time.time() - t0 < secs:
                n += 1 if c.read()[0] else 0
            return n / secs

        def _res_ladder(max_w, max_h):
            """Highest supported mode first, then common modes below it."""
            ladder = [(max_w, max_h)]
            for cw, ch in ((3840, 2160), (2560, 1440), (1920, 1080), (1280, 720)):
                if cw < max_w and (cw, ch) not in ladder:
                    ladder.append((cw, ch))
            return ladder

        auto = target_w is None or target_h is None
        cur_fps = _measure_fps(cap)
        _already_hi = w >= (1920 if auto else target_w)
        if _already_hi and cur_fps >= 15:
            print(f"📷 Capture: {w}x{h} @ ~{cur_fps:.0f}fps (native)")
            log.info(f"capture: {w}x{h} @ {cur_fps:.1f}fps native")
            return cap

        if platform.system() == 'Windows':
            cap.release()
            # DSHOW and MSMF enumerate devices differently (the C930e is
            # DSHOW index 4 but MSMF index 0 on this rig), so scan MSMF
            # indices rather than trusting the DSHOW index. Failed MSMF
            # opens time out slowly, so the last known-good index (cached)
            # is tried first. The right device delivers a real frame wider
            # than the default mode; black/static streams and virtual cams
            # fail the content + width checks.
            cached_msmf = None
            if MSMF_CACHE_FILE.exists():
                try:
                    cached_msmf = int(MSMF_CACHE_FILE.read_text().strip())
                except (ValueError, OSError):
                    cached_msmf = None
            msmf_candidates = []
            for i in ([cached_msmf] if cached_msmf is not None else []) + [index] + list(range(6)):
                if i not in msmf_candidates:
                    msmf_candidates.append(i)

            # Phase 1: find the right MSMF index and probe its resolution ceiling.
            found_idx, max_w, max_h = None, 0, 0
            for midx in msmf_candidates:
                probe_cap = cv2.VideoCapture(midx, cv2.CAP_MSMF)
                if not probe_cap.isOpened():
                    probe_cap.release()
                    continue
                probe_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                req_w, req_h = (7680, 4320) if auto else (target_w, target_h)
                probe_cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
                probe_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
                ret2, probe2 = probe_cap.read()
                pw = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ph = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                probe_cap.release()
                if (ret2 and probe2 is not None and pw > w
                        and np.mean(probe2) > 5 and np.std(probe2) > 10):
                    found_idx, max_w, max_h = midx, pw, ph
                    break

            if found_idx is not None:
                # Phase 2: walk the ladder down from the ceiling until a mode
                # sustains a usable frame rate. MSMF locks its format once
                # streaming, so each rung gets a clean reopen.
                targets = _res_ladder(max_w, max_h) if auto else [(target_w, target_h)]
                for req_w, req_h in targets:
                    new = cv2.VideoCapture(found_idx, cv2.CAP_MSMF)
                    if not new.isOpened():
                        new.release()
                        continue
                    new.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    new.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
                    new.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
                    new.set(cv2.CAP_PROP_FPS, 30)
                    ret2, probe2 = new.read()
                    if (not ret2 or probe2 is None
                            or np.mean(probe2) <= 5 or np.std(probe2) <= 10):
                        new.release()
                        continue
                    nh, nw = probe2.shape[:2]
                    if nw < req_w * 0.9:
                        new.release()  # driver didn't honour this mode
                        continue
                    new_fps = _measure_fps(new)
                    if new_fps >= 10:
                        print(f"📷 Capture boosted: {w}x{h}@~{cur_fps:.0f}fps → "
                              f"{nw}x{nh}@~{new_fps:.0f}fps (MSMF idx {found_idx}, MJPG)")
                        log.info(f"capture: boosted {w}x{h}@{cur_fps:.1f} -> "
                                 f"{nw}x{nh}@{new_fps:.1f} via MSMF idx {found_idx}")
                        try:
                            MSMF_CACHE_FILE.write_text(str(found_idx))
                        except OSError:
                            pass
                        return new
                    new.release()
            print(f"⚠️  High-res renegotiation FAILED — reverting to default camera mode. "
                  f"Face range will be reduced.")
            log.warning(f"capture: high-res renegotiation failed, default mode kept (was {w}x{h})")
            return _open_camera(index)

        # Non-Windows: in-place request is honoured by V4L2/AVFoundation.
        # An oversized request clamps to the sensor's top mode.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w if not auto else 7680)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h if not auto else 4320)
        return cap

    def _measure_pair_fps(caps, secs=0.8):
        """Read every cap in lockstep and return the MIN sustained fps across
        them. Shared-USB-bus starvation only shows when both cameras stream at
        once, so the pair must be measured together, not one at a time."""
        counts = [0] * len(caps)
        t0 = time.time()
        while time.time() - t0 < secs:
            for i, c in enumerate(caps):
                if c.read()[0]:
                    counts[i] += 1
        dt = max(1e-3, time.time() - t0)
        return min(counts) / dt

    def _negotiate_dual_capture(idx_a, idx_b, min_fps=10.0):
        """Pick the HIGHEST resolution both cameras can stream simultaneously.

        Two USB cameras on one controller share the bus, so a mode each can do
        alone (e.g. 4K) can starve to ~0 fps when run together. We walk a shared
        ladder from 4K down, open both feeds pinned to their own index (no
        cross-index scan, so neither mirrors the other), warm up, and measure the
        pair's concurrent fps. The first (highest) rung the pair sustains at
        >= min_fps wins — there is NO 1080p cap, so a fast PC on a USB3 bus keeps
        4K while a bandwidth-starved USB2 bus steps down and warns. Returns
        (cap_a, cap_b, (w, h), fps) or None if the pair never opens."""
        ladder = [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720)]
        last_opened = None  # lowest rung that at least opened, as a last resort
        for w, h in ladder:
            ca = _open_camera_pinned(idx_a, w, h)
            cb = _open_camera_pinned(idx_b, w, h)
            if not (ca.isOpened() and cb.isOpened()):
                ca.release(); cb.release()
                continue
            wa = int(ca.get(cv2.CAP_PROP_FRAME_WIDTH))
            wb = int(cb.get(cv2.CAP_PROP_FRAME_WIDTH))
            for _ in range(5):  # discard the first post-switch frames
                ca.read(); cb.read()
            fps = _measure_pair_fps([ca, cb])
            honoured = wa >= w * 0.9 and wb >= w * 0.9
            if honoured and fps >= min_fps:
                if last_opened is not None:
                    last_opened[0].release(); last_opened[1].release()
                print(f"🎥🎥 Dual capture: {wa}x{h} @ ~{fps:.0f}fps/cam "
                      f"(shared-bus negotiated, no 1080p cap)")
                log.info(f"dual capture negotiated {wa}x{h}@{fps:.1f}fps/cam")
                return ca, cb, (wa, h), fps
            if last_opened is not None:
                last_opened[0].release(); last_opened[1].release()
            last_opened = (ca, cb, wa, h, fps)
        if last_opened is not None:
            fa, fb, fw, fh, ffps = last_opened
            print(f"⚠️  Dual capture: the shared USB bus can't sustain >= {min_fps:.0f}fps for "
                  f"both cameras above {fw}x{fh} (~{ffps:.0f}fps). Using {fw}x{fh}.\n"
                  f"    For higher resolution give each camera its OWN USB controller: plug them "
                  f"into ports on different physical buses (e.g. one front + one rear header, a "
                  f"USB PCIe/ExpressCard add-in card, or one on a USB-C / Thunderbolt port). A "
                  f"powered USB3 hub does NOT help — it still shares one upstream bus.")
            log.warning(f"dual capture bus-limited to {fw}x{fh}@{ffps:.1f}fps/cam")
            return fa, fb, (fw, fh), ffps
        return None

    def select_camera_interactively(detected_cameras, default_camera, force=False):
        """
        Show a live tiled preview of all detected cameras and let the user
        choose one via a keypress.

        Key bindings:
            1 / 2 / 3 ...  — select camera N (1-based)
            Enter / Space  — accept highlighted default (360° preferred)
            Q / Escape     — quit application

        Returns (primary_camera, second_camera_or_None). The second camera is
        only offered when the chosen primary is 2D and at least one other 2D
        camera exists (dual mode is 2D-only); it enables DUAL-CAMERA mode.
        When force=False (default) the selector is skipped for a single camera;
        set force=True (via --select-camera) to always show it.
        """
        if len(detected_cameras) == 1 and not force:
            return detected_cameras[0], None

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

        # Dual mode (2D-only) is offered when the picked primary is 2D and at
        # least one OTHER 2D camera exists.
        def _dual_possible(primary_i):
            if detected_cameras[primary_i]['is_360']:
                return False
            return any(j != primary_i and not detected_cameras[j]['is_360']
                       for j in range(len(detected_cameras)))

        primary_number = None
        second_number = None
        phase = 'primary'
        _announced_second = False
        while True:
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
                is_primary = (phase == 'second' and (i + 1) == primary_number)
                # In the second phase a 360-degree camera can't join a dual rig.
                disabled = (phase == 'second' and not is_primary and cam['is_360'])
                if is_primary:
                    label_bg = (130, 90, 0)      # amber = locked-in primary
                elif is_default and phase == 'primary':
                    label_bg = (0, 130, 0)       # green = default
                elif disabled:
                    label_bg = (30, 30, 30)
                else:
                    label_bg = (50, 50, 50)
                cv2.rectangle(canvas,
                              (lx0, ly0), (lx0 + TILE_W, ly0 + LABEL_H),
                              label_bg, -1)
                tag_360     = "  360\u00b0" if cam['is_360'] else ""
                tag_last    = "  last used" if cam.get('last_used') else ""
                default_tag = "  [default]" if (is_default and phase == 'primary') else ""
                role_tag = ""
                if is_primary:
                    role_tag = "  [PRIMARY]"
                elif disabled:
                    role_tag = "  [2D only]"
                label = f"[{i+1}] cam{cam['index']}  {cam['resolution']}{tag_360}{tag_last}{default_tag}{role_tag}"
                cv2.putText(canvas, label, (lx0 + 6, ly0 + 19),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1,
                            cv2.LINE_AA)

            if phase == 'primary':
                cv2.setWindowTitle(win_name,
                    "Select PRIMARY camera - 1/2/3... or Enter for default, Q to quit")
            else:
                cv2.setWindowTitle(win_name,
                    "Add SECOND camera (dual) - pick a 2D cam number, or Enter/S = single, Q to quit")
                if not _announced_second:
                    print("\nAdd a SECOND 2D camera for DUAL mode? "
                          "Press its number, or Enter/S to stay single-camera.")
                    _announced_second = True

            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(30) & 0xFF

            # Number keys 1-9 select the PRIMARY (and, in phase 2, the SECOND) camera
            if phase == 'primary' and ord('1') <= key <= ord('9'):
                choice = key - ord('0')  # 1-based
                if 1 <= choice <= len(detected_cameras):
                    primary_number = choice
                    if _dual_possible(primary_number - 1):
                        phase = 'second'
                    else:
                        break
            elif phase == 'second' and ord('1') <= key <= ord('9'):
                choice = key - ord('0')
                if (1 <= choice <= len(detected_cameras) and choice != primary_number
                        and not detected_cameras[choice - 1]['is_360']):
                    second_number = choice
                    break
            elif phase == 'primary' and key in (13, 32):   # Enter/Space -> default
                primary_number = default_idx_in_list + 1
                if _dual_possible(primary_number - 1):
                    phase = 'second'
                else:
                    break
            elif phase == 'second' and key in (13, 32, ord('s'), ord('S')):  # skip -> single
                break
            elif key in (ord('q'), ord('Q'), 27):          # Q/Esc -> quit
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

        chosen = detected_cameras[primary_number - 1]
        chosen2 = detected_cameras[second_number - 1] if second_number else None
        fmt_tag = "360\u00b0" if chosen['is_360'] else "2D"
        print(f"\u2705 Selected camera {chosen['index']} ({chosen['resolution']}, {fmt_tag})")
        if chosen2 is not None:
            print(f"\u2705 Second camera {chosen2['index']} ({chosen2['resolution']}, 2D) - DUAL mode")
        return chosen, chosen2

    if args.video:
        # Video file specified
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file {args.video}")
            return
        video_source_name = args.video
        _cam_index = None
        print(f"📁 Opened video file: {args.video}")
    elif args.camera is not None:
        # Manual camera index specified - skip auto-detection
        cap = _open_camera(args.camera)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {args.camera}")
            return
        _cam_index = args.camera
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
            selected, selected2 = select_camera_interactively(
                detected_cameras, default_camera, force=args.select_camera,
            )
            # GUI dual-camera pick feeds the existing --camera2 setup path.
            if selected2 is not None and args.camera2 is None:
                args.camera2 = selected2['index']
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
        _cam_index = selected['index']
        video_source_name = f"Camera {selected['index']}"

    # =========================================================================
    # VIDEO FORMAT DETECTION (2D vs 360°)
    # =========================================================================
    if not args.video and _cam_index is not None:
        if args.camera2 is not None:
            # Dual mode: just grab a frame for format detection here. The real
            # capture resolution is negotiated later against the shared USB bus,
            # once both camera indices are known (see _negotiate_dual_capture),
            # so 4K is kept when the bus can carry it. Pin to this exact index
            # (no cross-index MSMF scan) so the two feeds can't converge on the
            # same physical camera.
            cap.release()
            _pinned = _open_camera_pinned(_cam_index, 1280, 720)
            cap = _pinned if _pinned.isOpened() else _open_camera(_cam_index)
        else:
            cap = _ensure_high_res_capture(cap, _cam_index)
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
    
    # Reset video to beginning / requested start position (for video files)
    video_start_ms = 0.0
    if args.video and args.start_time:
        parts = [float(p) for p in args.start_time.split(':')]
        secs = 0.0
        for p in parts:
            secs = secs * 60 + p
        video_start_ms = secs * 1000.0
        print(f"⏩ Starting video at {args.start_time} ({secs:.0f}s)")
    if args.video:
        if video_start_ms > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, video_start_ms)
        else:
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
    print(f"📡 Per-participant channels: 'device:<serial>:engagement'")
    if is_360:
        print(f"🌐 360° Mode: Processing {NUM_360_VIEWS} views per frame")
    if logger:
        print(f"📊 Data Logging: ENABLED")
    print("Press 'q' to quit.")
    if face_id:
        print("Press 'r' to register, 'f' to cycle focus, 'c' to clear.")
    if args.camera2 is not None and not is_360:
        print("Press 'v' to switch which camera the next enrollment targets.")
    
    # Live FPS tracking
    fps_frame_times = deque(maxlen=30)  # Rolling window for FPS calculation
    last_frame_time = time.time()
    live_fps = measured_fps  # Start with measured value
    last_adaptation_time = time.time()
    adaptation_interval = 2.0  # Adapt every 2 seconds
    
    # Use prefetcher for 360° mode (large frame decode benefits from overlap)
    prefetcher = FramePrefetcher(cap) if is_360 else None

    # --- Camera contexts: 1 (single) or 2 (dual). Each context owns its own
    # tracker/MediaPipe/face-caches so two feeds never collide on track ids;
    # the face-enrollment repo, physio sidebar and Redis publishing stay shared
    # and run once on the merged result. Dual mode is 2D-only. ---
    from concurrent.futures import ThreadPoolExecutor
    face_id_lock = threading.Lock()
    contexts = [CameraContext(0, cap, system, name='CAM0')]
    cam_pool = None
    active_enroll_cam = 0  # which feed the R/SPACE enrollment captures from
    if args.camera2 is not None and not is_360:
        cap2 = _open_camera(args.camera2)
        if not cap2.isOpened():
            # Diagnose why the explicitly-picked second camera won't open: probe
            # both backends directly so we can tell a dark-frame rejection (a
            # backend reads OK here but _open_camera discarded it) from a truly
            # busy/absent index (both False → the laptop has one usable camera).
            _d = cv2.VideoCapture(args.camera2, cv2.CAP_DSHOW)
            _d_ok = _d.isOpened() and bool(_d.read()[0])
            _d.release()
            _m = cv2.VideoCapture(args.camera2)  # MSMF
            _m_ok = _m.isOpened() and bool(_m.read()[0])
            _m.release()
            print(f"⚠️  --camera2 {args.camera2}: could not open — continuing single-camera "
                  f"(DSHOW reads={_d_ok}, MSMF reads={_m_ok})")
            # Give the surviving single feed the full high-res treatment.
            cap.release()
            cap = _ensure_high_res_capture(_open_camera(_cam_index), _cam_index)
            contexts[0].cap = cap
            cap2 = cv2.VideoCapture()  # closed → single-camera path below
        else:
            # Both cameras open. Free the primary + probe handles, then negotiate
            # the highest resolution the shared USB bus can sustain for BOTH at
            # once (no 1080p cap — 4K is kept when the bus can carry it).
            cap2.release()
            cap.release()
            _neg = _negotiate_dual_capture(_cam_index, args.camera2)
            if _neg is not None:
                cap, cap2, _dmode, _dfps = _neg
                contexts[0].cap = cap  # CAM0 context was built with the old handle
            else:
                print("⚠️  Could not bring both cameras up together — continuing single-camera.")
                cap = _ensure_high_res_capture(_open_camera(_cam_index), _cam_index)
                contexts[0].cap = cap
                cap2 = cv2.VideoCapture()
        if cap2.isOpened():
            system2 = MultiPersonEngagementSystem(
                MODEL_PATH, device=args.device,
                redis_host=args.redis_host, redis_port=args.redis_port)
            system2.sequence_length = calculated_seq_length
            # One merged crowd publish from the main loop instead of each
            # camera racing to overwrite engagement_score.
            system.publish_crowd = False
            system2.publish_crowd = False
            contexts.append(CameraContext(1, cap2, system2, name='CAM1'))
            cam_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='cam')
            print(f"🎥🎥 DUAL-CAMERA mode enabled (cam {args.camera2} as CAM1)")

    # Name of the main display window. The window itself is created lazily on
    # the first imshow below (with WINDOW_NORMAL) so that any earlier OpenCV
    # windows (e.g. the --select-camera preview) finish cleanly first and we
    # don't leave a grey placeholder window on screen.
    _WIN_NAME = 'Concert Engagement System'

    # Warm up every capture before the loop. Freshly-opened cameras — and the
    # second camera in a bandwidth-shared dual rig — often return empty on
    # their first reads, which would otherwise end the session with 0 frames.
    for c in contexts:
        for _ in range(10):
            wr, wf = c.cap.read()
            if wr and wf is not None:
                break
            time.sleep(0.05)

    _read_fail_since = None            # wall-clock of the first of a failure run
    LIVE_READ_GRACE_S = 5.0            # end the session only after this long with no frames

    # Dual-camera engagement fusion: serial -> smoothed cross-camera score.
    # Persists across frames so a person in both feeds gets ONE stable score on
    # both overlays and in Redis (see merge_people_by_serial). Stays empty in
    # single-camera mode, where each serial already has a single score.
    fused_scores = {}

    # Release the cameras on EVERY exit path. 'q' reaches the cleanup below, but
    # Ctrl+C (SIGINT), SIGTERM and unhandled exceptions previously skipped it and
    # left the capture devices held. Idempotent so the atexit/signal/normal
    # paths can all call it without double-closing.
    import atexit
    _cleaned_up = False
    def _cleanup():
        nonlocal _cleaned_up
        if _cleaned_up:
            return
        _cleaned_up = True
        if prefetcher:
            prefetcher.stop()
        if cam_pool is not None:
            cam_pool.shutdown(wait=False)
        for _c in contexts:
            try:
                _c.cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        if logger:
            logger.close()
    atexit.register(_cleanup)

    def _signal_shutdown(signum=None, frame=None):
        _cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, _signal_shutdown)
    signal.signal(signal.SIGTERM, _signal_shutdown)

    while True:
        frame_start_time = time.time()
        
        if prefetcher:
            ret, frame = prefetcher.read()
            if ret:
                contexts[0]._frame = frame
        else:
            ret = True
            failed_cam = None
            for c in contexts:
                r, fr = c.cap.read()
                if not r:
                    ret = False
                    failed_cam = c.name
                    break
                c._frame = fr
            frame = contexts[0]._frame if ret else None
        if not ret:
            if args.video:
                # Loop video file — reset capture and tracker state
                if video_start_ms > 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, video_start_ms)
                else:
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
            # Live camera: a single dropped frame must not end the session
            # (two USB cameras sharing bandwidth can stall briefly). Tolerate
            # gaps up to LIVE_READ_GRACE_S, naming the offending camera.
            now = time.time()
            if _read_fail_since is None:
                _read_fail_since = now
                print(f"\u26a0\ufe0f  {failed_cam or 'camera'} returned no frame — retrying...")
            elif now - _read_fail_since > LIVE_READ_GRACE_S:
                print(f"\u274c {failed_cam or 'camera'} delivered no frames for "
                      f"{LIVE_READ_GRACE_S:.0f}s — ending session. In dual mode this is "
                      f"usually USB bandwidth: try the cameras on separate USB controllers.")
                break
            time.sleep(0.03)
            continue
        _read_fail_since = None
        
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
            
            
            # Aggregate crowd average across all views (None when nobody is
            # scorable, so unknown frames don't read as 0% engagement)
            if all_scores:
                crowd_avg = sum(all_scores) / len(all_scores)
            else:
                crowd_avg = None
            
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
                            plabel = f"{system.display_label(pid)} {score:.0%} conf:{bf:.0%}"
                        else:
                            plabel = f"{system.display_label(pid)} {score:.0%}"
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
            # Standard 2D processing — each camera context runs its own full
            # pipeline (tracker, MediaPipe, transformer, face-ID, overlays). One
            # context is a normal single-camera run; two contexts run in parallel
            # worker threads and are merged by EmotiBit serial below.
            if len(contexts) == 1:
                c0 = contexts[0]
                process_and_annotate(c0, c0._frame, face_id, face_id_lock,
                                     focus_target, enroll_session, args)
                people_data = c0.people_data
                crowd_avg = c0.crowd_avg
                display_frame = c0.display_frame
            else:
                _futs = [cam_pool.submit(process_and_annotate, c, c._frame, face_id,
                                         face_id_lock, focus_target, enroll_session, args,
                                         False, fused_scores)
                         for c in contexts]
                for _f in _futs:
                    _f.result()
                people_data, crowd_avg = merge_people_by_serial(
                    [c.people_data for c in contexts], fused_scores)
                _frames = [c.display_frame for c in contexts]
                _h0 = _frames[0].shape[0]
                _frames = [f if f.shape[0] == _h0 else
                           cv2.resize(f, (int(f.shape[1] * _h0 / f.shape[0]), _h0))
                           for f in _frames]
                display_frame = np.hstack(_frames)
        
        # =================================================================
        # MERGED PUBLISH — face-ID, registered-only gating, enrollment banner
        # and per-person overlays now happen inside process_and_annotate() per
        # camera context (above). Here we publish ONCE on the merged result:
        #   * single camera: process_frame already published the crowd score
        #     internally (publish_crowd=True); we only add per-participant.
        #   * dual camera: both cameras' internal crowd publish is disabled, so
        #     we publish the merged crowd average and the serial-deduplicated
        #     per-participant channels once (a seam-straddling parent is one row).
        # =================================================================
        if not is_360:
            if len(contexts) > 1:
                _sys0 = contexts[0].system
                _now_pub = time.time()
                if (_sys0.redis_client and crowd_avg is not None
                        and (_now_pub - _sys0.last_publish_time) >= _sys0.publish_interval):
                    try:
                        _sys0.redis_client.publish(REDIS_CHANNEL, f"{crowd_avg:.4f}")
                        _sys0.last_publish_time = _now_pub
                    except Exception as _e:
                        print(f"Redis Error: {_e}")
            # Per-participant device:<serial>:engagement channels (AR
            # pattern-subscribes device:*). Merged people_data is already
            # serial-deduplicated, so each parent publishes once.
            contexts[0].system.publish_individual(people_data)
        
        # Calculate live FPS
        frame_duration = time.time() - frame_start_time
        fps_frame_times.append(frame_duration)
        if len(fps_frame_times) > 0:
            avg_frame_time = sum(fps_frame_times) / len(fps_frame_times)
            live_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Time-windowed buffer (roadmap 2.c) makes per-FPS sequence-length
        # adaptation obsolete: the buffer always spans TARGET_DURATION_SECONDS
        # of wall-clock time and is resampled to MODEL_INPUT_FRAMES at every
        # model call, so the model input shape and effective time window are
        # both invariant to live FPS. The old block adapted self.sequence_length
        # against live FPS and was clamped to [60,300] frames, which could not
        # represent 10s at FPS outside [6, 30].
        current_time = time.time()
        
        # --- VISUALIZATION (common for both modes) ---
        h, w = display_frame.shape[:2]
        
        # Draw Crowd Average Bar (Top Center)
        bar_w = min(400, w - 40)
        bar_h = 30
        bar_x = (w - bar_w) // 2
        bar_y = 20
        
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        # Fill Bar (leave grey + show "--" when unknown / no scorable audience)
        mode_str = "360" if is_360 else "2D"
        if crowd_avg is None:
            text = f"CROWD ENGAGEMENT ({mode_str}): --"
        else:
            fill_w = int(bar_w * crowd_avg)
            avg_color = (0, int(255 * crowd_avg), int(255 * (1-crowd_avg)))
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), avg_color, -1)
            text = f"CROWD ENGAGEMENT ({mode_str}): {crowd_avg:.1%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx = (w - tw) // 2
        ty = bar_y + bar_h + 25
        
        cv2.putText(display_frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)
        cv2.putText(display_frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FPS and Info (Bottom Left)
        platform_str = f"{platform_info['os']} | {actual_device.upper()}"
        people_count = len(people_data)
        context_seconds = system.current_context_seconds
        throttle_str = ""
        if system.last_total_count > 0 and system.last_selected_count < system.last_total_count:
            throttle_str = (f" | MP: {system.last_selected_count}/{system.last_total_count}"
                            f" rr{system._rr_offset}")
        gaze_str = ""
        if not is_360 and system.last_gaze_result is not None:
            _gr = system.last_gaze_result
            _n_perf = len([p for p in people_data if p.get('performer')])
            gaze_str = f" | Gaze: {'LIVE' if _gr['started'] else 'PRE-SHOW'}"
            if _n_perf:
                gaze_str += f" perf:{_n_perf}"
            if _gr['distraction']:
                gaze_str += " SHIFT!"
        fps_text = f"{platform_str} | FPS: {live_fps:.1f} | People: {people_count} | Context: {context_seconds:.1f}s{throttle_str}{gaze_str}"
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

        # Reconnect flash message (green, below centre so it doesn't collide
        # with the registration flash)
        if reconnect_flash_until > time.time():
            rf_text = reconnect_flash_msg
            (rtw, rth), _ = cv2.getTextSize(rf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            rx = (w - rtw) // 2
            ry = h // 2 + 50
            cv2.putText(display_frame, rf_text, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(display_frame, rf_text, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 2)

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
            reconnect_btn_rect,
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
            # but cap at 1600px so it never opens larger than typical laptop
            # screens. Uses the real sidebar width (it widens to 2–3 columns
            # with more wearers) so the aspect ratio stays correct.
            _full_w = w + _sidebar.shape[1]
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
                if detected_serials:
                    # All real EmotiBits assigned — nothing to offer
                    registration_flash_msg = "All detected EmotiBits already assigned"
                    registration_flash_until = time.time() + 2.5
                else:
                    # No EmotiBits running (e.g. distance testing without
                    # physio) — offer generic names so face ID still works
                    available = [n for n in (f"Participant-{i}" for i in range(1, 6))
                                 if n not in already_enrolled]
            if available:
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
                        reconnect_btn_rect,
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
                _enr_people = contexts[active_enroll_cam].people_data
                if _enr_people:
                    unidentified = [p for p in _enr_people if not p.get('identified_as')]
                    candidates = unidentified if unidentified else _enr_people
                    best = max(candidates,
                               key=lambda p: (p['bbox'][2]-p['bbox'][0]) * (p['bbox'][3]-p['bbox'][1]))
                    # Kick off the guided 3-pose capture; SPACE captures each
                    # pose in the main loop (subject stays ~1m away). 'cam' pins
                    # the capture to the active feed in dual-camera mode.
                    enroll_session = {'serial': chosen_serial, 'tid': best['id'],
                                      'idx': 0, 'target_bbox': None,
                                      'cam': active_enroll_cam}
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

        # V = Toggle which live feed the next R/SPACE enrollment targets
        # (dual-camera only; a no-op with a single feed).
        elif key == ord('v') and len(contexts) > 1:
            active_enroll_cam = (active_enroll_cam + 1) % len(contexts)
            registration_flash_msg = f"Enroll target: {contexts[active_enroll_cam].name}"
            registration_flash_until = time.time() + 1.5
        
        # C = Clear all enrollments
        elif key == ord('c') and face_id:
            face_id.clear_enrollments()
            for _c in contexts:
                _c.face_id_cache.clear()   # drop stale track→name matches immediately
                _c.face_id_pending.clear()
                _c.registered_profiles.clear()
            enroll_session = None
            focus_target = 'all'
            registration_flash_msg = "All enrollments cleared"
            registration_flash_until = time.time() + 2.0
        
        # Esc = cancel an in-progress enrollment session
        elif key == 27 and enroll_session:
            enroll_session = None
            registration_flash_msg = "Enrollment cancelled"
            registration_flash_until = time.time() + 1.5
        
        # SPACE = capture current enrollment pose
        elif key == 32 and enroll_session and face_id and not is_360:
            es = enroll_session
            bbox = es.get('target_bbox')
            _ectx = contexts[es.get('cam', 0)]
            _eframe = _ectx._frame
            if bbox is None:
                registration_flash_msg = "No person tracked — step into frame"
                registration_flash_until = time.time() + 1.5
            else:
                pose = ENROLL_POSES[es['idx']]
                hc = head_crop(_eframe, bbox)
                ok = False
                for crop in (hc,
                             _eframe[max(0, bbox[1]):min(_eframe.shape[0], bbox[3]),
                                   max(0, bbox[0]):min(_eframe.shape[1], bbox[2])]):
                    if crop is None or crop.size == 0:
                        continue
                    try:
                        ok = face_id.register_from_crop(crop, name=es['serial'], angle=pose)
                    except TypeError:
                        ok = face_id.register_from_crop(crop, name=es['serial'])  # legacy backend
                    if ok:
                        break
                if ok:
                    es['idx'] += 1
                    if es['idx'] >= len(ENROLL_POSES):
                        _ectx.face_id_cache[es['tid']] = {'name': es['serial'],
                                                    'similarity': 1.0, 'miss': 0}
                        _ectx.face_id_pending.pop(es['tid'], None)
                        registration_flash_msg = f"Enrolled: {es['serial']} (3 poses, ID:{es['tid']})"
                        registration_flash_until = time.time() + 2.5
                        log.info(f"face-id: enrolled {es['serial']} (3 poses, track {es['tid']})")
                        enroll_session = None
                    else:
                        registration_flash_msg = f"Captured {pose.upper()} — now face {ENROLL_POSES[es['idx']].upper()}"
                        registration_flash_until = time.time() + 1.5
                else:
                    registration_flash_msg = f"No face detected ({pose}) — try again"
                    registration_flash_until = time.time() + 1.5
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    _cleanup()

if __name__ == "__main__":
    import faulthandler
    import traceback

    # Write any crash (Python exception OR hard native crash) to a log file next
    # to this script, so a silent exit always leaves evidence to send back.
    _crash_log_path = Path(__file__).parent / 'crash_last_run.log'
    try:
        _crash_log = open(_crash_log_path, 'w', encoding='utf-8')
    except OSError:
        _crash_log = None
    if _crash_log is not None:
        # Dumps a C-level stack on an access violation / segfault before the
        # process dies (catches native OpenCV / CUDA / MediaPipe crashes).
        faulthandler.enable(file=_crash_log, all_threads=True)

    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        _restore_native_stderr()  # un-hide the traceback if stderr is still silenced
        _tb = traceback.format_exc()
        _stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
        sys.stderr.write("\n" + "=" * 68 + "\n")
        sys.stderr.write(f"  FATAL: engagement app crashed at {_stamp}\n")
        sys.stderr.write("=" * 68 + "\n")
        sys.stderr.write(_tb + "\n")
        sys.stderr.flush()
        if _crash_log is not None:
            _crash_log.write(f"\nFATAL {_stamp}\n{_tb}\n")
            _crash_log.flush()
        print(f"\n\U0001f4c4 Full error written to: {_crash_log_path}")
        sys.exit(1)
    finally:
        if _crash_log is not None:
            _crash_log.close()