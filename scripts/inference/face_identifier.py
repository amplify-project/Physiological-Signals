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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_FACE_WEIGHTS = PROJECT_ROOT / 'models' / FACE_WEIGHTS_FILENAME


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
            f"Rerun setup.ps1/setup.sh or place the file in '{PROJECT_FACE_WEIGHTS.parent}'. "
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
