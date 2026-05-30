"""Visualise saved keypoints from an NPZ session file.

Usage:
    python scripts/analysis/view_keypoints.py data/sessions/<session>/keypoints.npz
    python scripts/analysis/view_keypoints.py data/sessions/<session>/keypoints.npz --frame 50 --track 1
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# MediaPipe landmark layout within the 543-point array
POSE_SLICE   = slice(0, 33)      # 33 pose landmarks
FACE_SLICE   = slice(33, 501)    # 468 face mesh landmarks
LHAND_SLICE  = slice(501, 522)   # 21 left hand landmarks
RHAND_SLICE  = slice(522, 543)   # 21 right hand landmarks

# Pose skeleton connections (MediaPipe Pose)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),    # head right
    (0,4),(4,5),(5,6),(6,8),    # head left
    (9,10),                      # mouth
    (11,12),                     # shoulders
    (11,13),(13,15),             # left arm
    (12,14),(14,16),             # right arm
    (15,17),(15,19),(15,21),     # left hand connects
    (16,18),(16,20),(16,22),     # right hand connects
    (11,23),(12,24),             # torso
    (23,24),                     # hips
    (23,25),(25,27),(27,29),(27,31),(29,31),  # left leg
    (24,26),(26,28),(28,30),(28,32),(30,32),  # right leg
]

# Hand skeleton connections (MediaPipe Hands)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),          # palm
]


def plot_frame(keypoints, title="Keypoints"):
    """Plot pose, face, and hand keypoints for one person-frame."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # --- Full body view (pose + face outline) ---
    ax = axes[0]
    ax.set_title("Pose + Face", fontsize=12)
    
    pose = keypoints[POSE_SLICE]    # (33, 3)
    face = keypoints[FACE_SLICE]    # (468, 3)
    
    # Face mesh as point cloud
    ax.scatter(face[:, 0], face[:, 1], s=0.3, c='lightblue', alpha=0.6, label='face')
    
    # Pose skeleton
    ax.scatter(pose[:, 0], pose[:, 1], s=20, c='red', zorder=5, label='pose')
    for i, j in POSE_CONNECTIONS:
        ax.plot([pose[i, 0], pose[j, 0]], [pose[i, 1], pose[j, 1]],
                'r-', linewidth=1.5, alpha=0.7)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # flip Y — MediaPipe uses top-left origin
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    
    # --- Face zoom ---
    ax = axes[1]
    ax.set_title("Face Mesh", fontsize=12)
    ax.scatter(face[:, 0], face[:, 1], s=1, c='deepskyblue')
    
    # Zoom to face bounding box with padding
    fx_min, fx_max = face[:, 0].min(), face[:, 0].max()
    fy_min, fy_max = face[:, 1].min(), face[:, 1].max()
    pad = 0.02
    ax.set_xlim(fx_min - pad, fx_max + pad)
    ax.set_ylim(fy_max + pad, fy_min - pad)  # flip Y
    ax.set_aspect('equal')
    
    # --- Hands ---
    ax = axes[2]
    ax.set_title("Hands", fontsize=12)
    
    lhand = keypoints[LHAND_SLICE]  # (21, 3)
    rhand = keypoints[RHAND_SLICE]  # (21, 3)
    
    for hand, color, label in [(lhand, 'green', 'left'), (rhand, 'orange', 'right')]:
        # Only plot if hand was actually detected (not all zeros)
        if np.any(hand[:, :2] > 0):
            ax.scatter(hand[:, 0], hand[:, 1], s=15, c=color, zorder=5, label=label)
            for i, j in HAND_CONNECTIONS:
                ax.plot([hand[i, 0], hand[j, 0]], [hand[i, 1], hand[j, 1]],
                        color=color, linewidth=1.2, alpha=0.7)
    
    ax.set_ylim(ax.get_ylim()[::-1])  # flip Y
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualise saved keypoints")
    parser.add_argument("npz_path", type=str, help="Path to keypoints.npz")
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame number to plot (default: middle frame)")
    parser.add_argument("--track", type=int, default=None,
                        help="Track ID to plot (default: most frequent)")
    args = parser.parse_args()
    
    data = np.load(args.npz_path)
    frames = data['frames']
    track_ids = data['track_ids']
    keypoints = data['keypoints']  # (N, 543, 3)
    
    print(f"Loaded: {keypoints.shape[0]} person-frames, "
          f"dtype={keypoints.dtype}, "
          f"frames {frames.min()}-{frames.max()}, "
          f"track IDs: {np.unique(track_ids)}")
    
    # Pick track ID
    if args.track is not None:
        tid = args.track
    else:
        # Most frequent track
        unique, counts = np.unique(track_ids, return_counts=True)
        tid = unique[counts.argmax()]
    
    mask = track_ids == tid
    tid_frames = frames[mask]
    tid_kps = keypoints[mask]
    
    if len(tid_frames) == 0:
        print(f"No data for track_id={tid}")
        return
    
    # Pick frame
    if args.frame is not None:
        idx = np.where(tid_frames == args.frame)[0]
        if len(idx) == 0:
            print(f"Track {tid} not in frame {args.frame}. "
                  f"Available: {tid_frames[0]}-{tid_frames[-1]}")
            return
        idx = idx[0]
    else:
        idx = len(tid_frames) // 2  # middle
    
    kp = tid_kps[idx].astype(np.float32)  # upcast for plotting
    title = f"Track {tid}, Frame {tid_frames[idx]}"
    print(f"Plotting: {title}")
    plot_frame(kp, title)


if __name__ == "__main__":
    main()
