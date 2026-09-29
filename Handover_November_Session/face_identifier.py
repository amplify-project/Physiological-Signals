"""
Face Identification Module
==========================
Encapsulates face detection, embedding extraction, and identity matching
using facenet-pytorch (MTCNN + InceptionResnetV1).

Baby-step v1: Single-person registration via live webcam capture.
Future: 3-angle registration, pre-saved image files, up to 5 people.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN


# Cosine similarity threshold for a positive match
MATCH_THRESHOLD = 0.65

# Magenta colour (BGR) for identified participants
IDENTIFIED_COLOR = (255, 0, 255)

# Where to persist enrolled embeddings between sessions
ENROLLMENTS_DIR = Path(__file__).parent.resolve() / '.face_enrollments'

# Face embedding weights needed by InceptionResnetV1 for identification.
FACE_WEIGHTS_FILENAME = '20180402-114759-vggface2.pt'
PROJECT_FACE_WEIGHTS = Path(__file__).parent.resolve() / 'model' / FACE_WEIGHTS_FILENAME


def _torch_checkpoints_dir():
    """Return the torch checkpoint cache directory used by facenet-pytorch."""
    torch_home = os.getenv('TORCH_HOME')
    if torch_home:
        return Path(torch_home).expanduser() / 'checkpoints'

    xdg_cache_home = os.getenv('XDG_CACHE_HOME')
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / 'torch' / 'checkpoints'

    return Path.home() / '.cache' / 'torch' / 'checkpoints'


def _candidate_weights_paths():
    """Search project-local weights first, then the shared torch cache."""
    candidates = [PROJECT_FACE_WEIGHTS, _torch_checkpoints_dir() / FACE_WEIGHTS_FILENAME]
    seen = set()
    ordered = []
    for path in candidates:
        resolved = str(path)
        if resolved not in seen:
            ordered.append(path)
            seen.add(resolved)
    return ordered


class FaceIdentifier:
    """Detects, embeds, and matches faces against enrolled identities.
    
    Usage:
        fi = FaceIdentifier(device='cuda')
        fi.register_from_crop(bgr_crop, name='Eoghan')
        match = fi.identify(bgr_crop)
        # match = {'name': 'Eoghan', 'similarity': 0.87} or None
    """

    def __init__(self, device='cuda'):
        requested_device = self._resolve_device(device)
        init_attempts = []
        self.weights_path = None

        for candidate_device in self._candidate_devices(requested_device):
            try:
                self.mtcnn = MTCNN(
                    image_size=160,
                    margin=20,
                    keep_all=True,          # Return all faces in frame
                    min_face_size=40,
                    thresholds=[0.6, 0.7, 0.7],
                    device=candidate_device,
                    post_process=True,      # Normalise pixel values
                )
                self.resnet, self.weights_path = self._build_resnet(candidate_device)
                self.device = candidate_device
                if candidate_device != requested_device:
                    last_error = init_attempts[-1][1]
                    print(f"WARNING: FaceIdentifier falling back to CPU after {requested_device}: {last_error}")
                break
            except Exception as e:
                init_attempts.append((candidate_device, e))
        else:
            details = "; ".join(f"{dev}: {err}" for dev, err in init_attempts)
            raise RuntimeError(f"FaceIdentifier could not initialize ({details})")
        
        # Enrolled identities: list of {name: str, embedding: torch.Tensor}
        self.enrolled = []
        
        # Load persisted enrollments if they exist
        self._load_enrollments()
        
        print(f"FaceIdentifier ready on {self.device} "
              f"({len(self.enrolled)} enrolled, weights={self.weights_path})")

    def _resolve_device(self, requested_device):
        """Map 'auto' / unavailable accelerators to a concrete torch device."""
        if isinstance(requested_device, torch.device):
            return requested_device

        requested = str(requested_device).lower()
        if requested == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')

        if requested == 'cuda':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if requested == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        return torch.device('cpu')

    def _candidate_devices(self, preferred_device):
        """Try the requested accelerator first, then fall back to CPU if needed."""
        devices = [preferred_device]
        if preferred_device.type != 'cpu':
            devices.append(torch.device('cpu'))
        return devices

    def _build_resnet(self, device):
        """Build the embedding model from a local state_dict instead of downloading at runtime."""
        weights_path = self._find_weights_path()
        try:
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location='cpu')

        model = InceptionResnetV1(classify=False).eval()
        incompatible = model.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in incompatible.unexpected_keys if k not in {'logits.weight', 'logits.bias'}]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                f"Unexpected face-ID weights in '{weights_path}' "
                f"(missing={incompatible.missing_keys}, unexpected={unexpected})"
            )

        return model.to(device), weights_path

    def _find_weights_path(self):
        """Resolve the pretrained embedding weights from the repo or torch cache."""
        for path in _candidate_weights_paths():
            if path.exists():
                return path

        searched = ', '.join(str(p) for p in _candidate_weights_paths())
        raise FileNotFoundError(
            f"Missing face-ID weights '{FACE_WEIGHTS_FILENAME}'. "
            f"Rerun SETUP.bat or place the file in '{PROJECT_FACE_WEIGHTS.parent}'. "
            f"Searched: {searched}"
        )
    
    # -----------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------
    def register_from_crop(self, bgr_crop, name='Participant 1'):
        """Register a face from a BGR image crop.
        
        Args:
            bgr_crop: BGR image (numpy array) containing exactly one face
            name: Display name for this person
            
        Returns:
            bool: True if registration succeeded, False otherwise
        """
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        
        # Detect + align faces (MTCNN can crash on tiny crops)
        try:
            faces, probs = self.mtcnn(rgb, return_prob=True)
        except RuntimeError:
            print("ERROR: No face detected in crop - registration failed")
            return False
        
        if faces is None or len(faces) == 0:
            print("ERROR: No face detected in crop - registration failed")
            return False
        
        # Take the highest-confidence face
        best_idx = int(np.argmax(probs))
        face_tensor = faces[best_idx].unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.resnet(face_tensor).cpu()
        
        # Check for duplicate name — update if exists
        for entry in self.enrolled:
            if entry['name'] == name:
                entry['embedding'] = embedding
                print(f"Updated enrollment for '{name}' (prob={probs[best_idx]:.3f})")
                self._save_enrollments()
                return True
        
        self.enrolled.append({'name': name, 'embedding': embedding})
        print(f"Enrolled '{name}' (prob={probs[best_idx]:.3f}, "
              f"total enrolled: {len(self.enrolled)})")
        self._save_enrollments()
        return True
    
    # -----------------------------------------------------------------
    # Identification
    # -----------------------------------------------------------------
    def identify_crop(self, bgr_crop):
        """Try to identify a face in a BGR image crop.
        
        Args:
            bgr_crop: BGR image region (e.g. person bbox from YOLO)
            
        Returns:
            dict or None: {'name': str, 'similarity': float} if matched,
                          None if no face detected or no match above threshold
        """
        if not self.enrolled:
            return None
        
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        
        # Detect + align (MTCNN can crash on tiny crops with no proposals)
        try:
            faces, probs = self.mtcnn(rgb, return_prob=True)
        except RuntimeError:
            return None
        
        if faces is None or len(faces) == 0:
            return None
        
        # Take highest-confidence face
        best_idx = int(np.argmax(probs))
        face_tensor = faces[best_idx].unsqueeze(0).to(self.device)
        
        # Embed
        with torch.no_grad():
            embedding = self.resnet(face_tensor).cpu()
        
        # Compare against all enrolled
        best_match = None
        best_sim = -1.0
        
        for entry in self.enrolled:
            sim = torch.nn.functional.cosine_similarity(
                embedding, entry['embedding']
            ).item()
            if sim > best_sim:
                best_sim = sim
                best_match = entry['name']
        
        if best_sim >= MATCH_THRESHOLD:
            return {'name': best_match, 'similarity': best_sim}
        
        return None
    
    def detect_faces_in_frame(self, frame_rgb):
        """Detect all face bounding boxes in a full frame.
        
        Args:
            frame_rgb: RGB image (numpy array)
            
        Returns:
            list of (x1, y1, x2, y2) face bounding boxes, or empty list
        """
        try:
            boxes, probs = self.mtcnn.detect(frame_rgb)
        except RuntimeError:
            return []
        if boxes is None:
            return []
        return [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) 
                for b, p in zip(boxes, probs) if p > 0.5]
    
    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    def _save_enrollments(self):
        """Persist enrolled embeddings to disk."""
        ENROLLMENTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = ENROLLMENTS_DIR / 'enrollments.pt'
        data = [{'name': e['name'], 'embedding': e['embedding']} 
                for e in self.enrolled]
        torch.save(data, save_path)
    
    def _load_enrollments(self):
        """Load persisted enrollments from disk."""
        save_path = ENROLLMENTS_DIR / 'enrollments.pt'
        if save_path.exists():
            try:
                data = torch.load(save_path, map_location='cpu', weights_only=True)
                self.enrolled = data
            except Exception as e:
                print(f"WARNING: Could not load enrollments: {e}")
                self.enrolled = []
    
    def clear_enrollments(self):
        """Remove all enrolled identities."""
        self.enrolled = []
        save_path = ENROLLMENTS_DIR / 'enrollments.pt'
        if save_path.exists():
            save_path.unlink()
        print("All enrollments cleared")
    
    @property
    def has_enrollments(self):
        return len(self.enrolled) > 0
    
    @property
    def enrolled_names(self):
        return [e['name'] for e in self.enrolled]


# =============================================================================
# LONG-RANGE ONNX BACKEND — YuNet detector + SFace recognizer (pure OpenCV)
# =============================================================================
# MTCNN loses faces below ~30px, which caps recognition range around 2-3m on a
# 1080p webcam. YuNet (WIDER-FACE-trained) detects down to ~12px and returns
# 5 landmarks for alignment; SFace embeds the aligned 112x112 crop. Both run
# as OpenCV DNN ONNX models — no new dependencies, CPU-cheap (~5ms/face).

YUNET_FILENAME = 'face_detection_yunet_2023mar.onnx'
SFACE_FILENAME = 'face_recognition_sface_2021dec.onnx'
FACE_MODEL_DIR = Path(__file__).parent.resolve() / 'model'

# Official SFace cosine threshold is 0.363; slightly relaxed for large faces
# because the live pipeline adds multi-frame voting on top. Small faces carry
# less identity information and blur-converge (measured impostor sim 0.41 at
# 24px vs 0.30 at 60px), so below SFACE_FAR_PX the bar is raised.
SFACE_MATCH_THRESHOLD = 0.34
SFACE_FAR_THRESHOLD = 0.45
SFACE_FAR_PX = 40

DETECT_MIN_SIDE = 320    # upscale small head crops so YuNet has enough pixels
DETECT_MAX_SCALE = 4.0
LOWRES_SIM_WIDTH = 24    # enrollment augmentation: simulate a ~7m face


def head_crop(frame, bbox, top_frac=0.32, width_frac=0.76):
    """Head region of a person bbox (top ~third, centre-weighted width).

    Detecting inside an upscaled head crop instead of the full frame is an
    effective free zoom at distance, and keeps neighbours' faces out."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    cx = (x1 + x2) / 2.0
    half_w = bw * width_frac / 2.0
    nx1 = int(max(0, cx - half_w))
    nx2 = int(min(w, cx + half_w))
    ny1 = int(max(0, y1 - 0.05 * bh))
    ny2 = int(min(h, y1 + top_frac * bh))
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return None
    return frame[ny1:ny2, nx1:nx2]


class ONNXFaceIdentifier:
    """Long-range drop-in replacement for FaceIdentifier.

    Differences that matter at distance:
      - tiny-face detection (YuNet) with automatic crop upscaling
      - multi-angle enrollment: several embeddings per name (front/left/right),
        each also stored as a simulated low-resolution variant so gallery and
        7m probe live in the same blur domain; match = max cosine over all
      - identify_crop() reports the *native* face width in pixels so the
        caller can gate trust by distance
    """

    def __init__(self, device='cpu'):
        yunet_path = FACE_MODEL_DIR / YUNET_FILENAME
        sface_path = FACE_MODEL_DIR / SFACE_FILENAME
        missing = [p.name for p in (yunet_path, sface_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing ONNX face models in '{FACE_MODEL_DIR}': {', '.join(missing)} "
                f"(rerun SETUP.bat to download)")
        if not hasattr(cv2, 'FaceDetectorYN'):
            raise RuntimeError("OpenCV build lacks FaceDetectorYN (need >= 4.5.4)")

        self.detector = cv2.FaceDetectorYN.create(
            str(yunet_path), "", (320, 320),
            score_threshold=0.6, nms_threshold=0.3, top_k=50)
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
        self.device = 'cpu/opencv-dnn'

        # [{'name': str, 'angle': str, 'embedding': np.ndarray (128,) L2-normed}]
        self.enrolled = []
        self._load_enrollments()
        print(f"ONNXFaceIdentifier ready (YuNet+SFace, "
              f"{len(self.enrolled_names)} people / {len(self.enrolled)} embeddings)")

    # -----------------------------------------------------------------
    # Detection / embedding internals
    # -----------------------------------------------------------------
    def _detect_best(self, bgr):
        """Largest face in (possibly upscaled) crop.

        Returns (face_row, scale, img_used) — face_row is None if no face.
        face_row format: [x, y, w, h, 5x(lm_x, lm_y), score] in img_used coords.
        """
        h, w = bgr.shape[:2]
        if h < 12 or w < 12:
            return None, 1.0, bgr
        scale = 1.0
        short = min(h, w)
        if short < DETECT_MIN_SIDE:
            scale = min(DETECT_MAX_SCALE, DETECT_MIN_SIDE / short)
        if scale != 1.0:
            img = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        else:
            img = bgr
        self.detector.setInputSize((img.shape[1], img.shape[0]))
        try:
            _, faces = self.detector.detect(img)
        except cv2.error:
            return None, scale, img
        if faces is None or len(faces) == 0:
            return None, scale, img
        best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        return best, scale, img

    def _embed_aligned(self, aligned_bgr):
        feat = self.recognizer.feature(aligned_bgr).flatten().astype(np.float32)
        n = np.linalg.norm(feat)
        return feat / n if n > 0 else feat

    def _embed(self, img, face_row):
        aligned = self.recognizer.alignCrop(img, face_row)
        return self._embed_aligned(aligned), aligned

    # -----------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------
    def register_from_crop(self, bgr_crop, name='Participant 1', angle='front'):
        """Register one pose. Stores the embedding plus a low-res-degraded
        variant (gallery then matches distant, blurry probes far better)."""
        face, scale, img = self._detect_best(bgr_crop)
        if face is None:
            print(f"ERROR: No face detected in crop - registration failed ({name}/{angle})")
            return False

        emb, aligned = self._embed(img, face)
        small = cv2.resize(aligned, (LOWRES_SIM_WIDTH, LOWRES_SIM_WIDTH),
                           interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(small, (aligned.shape[1], aligned.shape[0]),
                              interpolation=cv2.INTER_CUBIC)
        emb_lr = self._embed_aligned(degraded)

        for ang, e in ((angle, emb), (f"{angle}-lowres", emb_lr)):
            entry = next((x for x in self.enrolled
                          if x['name'] == name and x['angle'] == ang), None)
            if entry is not None:
                entry['embedding'] = e
            else:
                self.enrolled.append({'name': name, 'angle': ang, 'embedding': e})

        self._save_enrollments()
        print(f"Enrolled '{name}' pose={angle} (score={float(face[-1]):.3f}, "
              f"{len(self.enrolled)} embeddings total)")
        return True

    # -----------------------------------------------------------------
    # Identification
    # -----------------------------------------------------------------
    def identify_crop(self, bgr_crop):
        """Identify the largest face in a (head) crop.

        Returns:
            None                        — no face detected at all
            {'name': None, 'similarity', 'face_px'}   — face seen, nobody matched
            {'name': str,  'similarity', 'face_px'}   — matched
        face_px = face width in NATIVE (pre-upscale) pixels: the caller's
        distance/trust gate.
        """
        face, scale, img = self._detect_best(bgr_crop)
        if face is None:
            return None
        face_px = float(face[2]) / scale
        if not self.enrolled:
            return {'name': None, 'similarity': 0.0, 'face_px': face_px}

        emb, _ = self._embed(img, face)
        best_name, best_sim = None, -1.0
        for entry in self.enrolled:
            sim = float(np.dot(emb, entry['embedding']))
            if sim > best_sim:
                best_sim = sim
                best_name = entry['name']

        threshold = SFACE_MATCH_THRESHOLD if face_px >= SFACE_FAR_PX else SFACE_FAR_THRESHOLD
        if best_sim >= threshold:
            return {'name': best_name, 'similarity': best_sim, 'face_px': face_px}
        return {'name': None, 'similarity': best_sim, 'face_px': face_px}

    def detect_faces_in_frame(self, frame_rgb):
        """All face boxes in a full frame (RGB in, to match old API)."""
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self.detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        try:
            _, faces = self.detector.detect(bgr)
        except cv2.error:
            return []
        if faces is None:
            return []
        return [(int(f[0]), int(f[1]), int(f[0] + f[2]), int(f[1] + f[3]))
                for f in faces]

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    _SAVE_PATH = ENROLLMENTS_DIR / 'enrollments_sface.npz'

    def _save_enrollments(self):
        ENROLLMENTS_DIR.mkdir(parents=True, exist_ok=True)
        if not self.enrolled:
            if self._SAVE_PATH.exists():
                self._SAVE_PATH.unlink()
            return
        np.savez(self._SAVE_PATH,
                 names=np.array([e['name'] for e in self.enrolled]),
                 angles=np.array([e['angle'] for e in self.enrolled]),
                 embeddings=np.stack([e['embedding'] for e in self.enrolled]))

    def _load_enrollments(self):
        if not self._SAVE_PATH.exists():
            return
        try:
            data = np.load(self._SAVE_PATH)
            self.enrolled = [
                {'name': str(n), 'angle': str(a), 'embedding': e.astype(np.float32)}
                for n, a, e in zip(data['names'], data['angles'], data['embeddings'])
            ]
        except Exception as e:
            print(f"WARNING: Could not load SFace enrollments: {e}")
            self.enrolled = []

    def clear_enrollments(self):
        self.enrolled = []
        if self._SAVE_PATH.exists():
            self._SAVE_PATH.unlink()
        print("All enrollments cleared")

    @property
    def has_enrollments(self):
        return len(self.enrolled) > 0

    @property
    def enrolled_names(self):
        seen = []
        for e in self.enrolled:
            if e['name'] not in seen:
                seen.append(e['name'])
        return seen


def create_face_identifier(device='auto'):
    """Prefer the long-range YuNet+SFace backend; fall back to MTCNN/FaceNet."""
    try:
        return ONNXFaceIdentifier(device)
    except Exception as e:
        print(f"WARNING: ONNX face backend unavailable ({e}) — falling back to MTCNN/FaceNet")
        return FaceIdentifier(device=device)
