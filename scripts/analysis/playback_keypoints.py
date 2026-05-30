"""Animated playback of saved keypoints from an NPZ session file.

Plays back pose, face mesh, and hand keypoints frame-by-frame like a video.
Supports single-track or multi-person (--all-tracks) mode.

Controls:
    Space       — Play / Pause
    Right arrow  — Next frame (when paused)
    Left arrow   — Previous frame (when paused)
    Up arrow     — Speed up (fewer ms per frame)
    Down arrow   — Slow down (more ms per frame)
    Home         — Jump to first frame
    End          — Jump to last frame
    Q / Escape   — Quit

Usage:
    python scripts/analysis/playback_keypoints.py data/sessions/<session>/keypoints.npz
    python scripts/analysis/playback_keypoints.py <path> --all-tracks    # all people
    python scripts/analysis/playback_keypoints.py <path> --track 1
    python scripts/analysis/playback_keypoints.py <path> --interval 50   # 50ms per frame
    python scripts/analysis/playback_keypoints.py <path> --start 100     # start at frame 100
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# ── MediaPipe landmark layout within the 543-point array ──
POSE_SLICE  = slice(0, 33)
FACE_SLICE  = slice(33, 501)
LHAND_SLICE = slice(501, 522)
RHAND_SLICE = slice(522, 543)

# Pose skeleton connections
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16),
    (15,17),(15,19),(15,21),
    (16,18),(16,20),(16,22),
    (11,23),(12,24),
    (23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]

# Hand skeleton connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Face mesh triangulation edges (subset for contour visibility)
# Canonical contour indices for jaw, eyes, brows, nose, lips
FACE_CONTOUR_CONNECTIONS = [
    # Jaw contour
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),
    (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
    (379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),
    (149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),
    (234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10),
    # Left eye
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),
    (155,133),(133,173),(173,157),(157,158),(158,159),(159,160),(160,161),
    (161,246),(246,33),
    # Right eye
    (362,382),(382,381),(381,380),(380,374),(374,373),(373,390),(390,249),
    (249,263),(263,466),(466,388),(388,387),(387,386),(386,385),(385,384),
    (384,398),(398,362),
    # Left eyebrow
    (46,53),(53,52),(52,65),(65,55),(55,70),
    # Right eyebrow
    (276,283),(283,282),(282,295),(295,285),(285,300),
    # Nose bridge
    (168,6),(6,197),(197,195),(195,5),
    # Outer lips
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),
    (405,321),(321,375),(375,291),(291,409),(409,270),(270,269),(269,267),
    (267,0),(0,37),(37,39),(39,40),(40,185),(185,61),
    # Inner lips
    (78,95),(95,88),(88,178),(178,87),(87,14),(14,317),(317,402),
    (402,318),(318,324),(324,308),(308,415),(415,310),(310,311),(311,312),
    (312,13),(13,82),(82,81),(81,80),(80,191),(191,78),
]


class KeypointPlayer:
    """Interactive keypoint playback with matplotlib."""

    # Distinct colors for multi-person mode
    TRACK_COLORS = [
        ("#e6194b", "#fab0b0"),  # red / light red
        ("#3cb44b", "#a9dfab"),  # green / light green
        ("#4363d8", "#a3b5e8"),  # blue / light blue  
        ("#f58231", "#fcc89b"),  # orange / light orange
        ("#911eb4", "#d4a5e6"),  # purple / light purple
        ("#42d4f4", "#a8ecf7"),  # cyan / light cyan
    ]

    def __init__(self, npz_path, track_id=None, interval=66, start_frame=0,
                 all_tracks=False, smooth=False):
        # ── Load data ──
        data = np.load(npz_path)
        self.all_frames = data["frames"]
        self.all_track_ids = data["track_ids"]
        self.all_keypoints = data["keypoints"]  # (N, 543, 3)
        self.all_tracks_mode = all_tracks

        # Bounding boxes & frame size (new format — may be absent in old NPZ)
        self.has_bboxes = "bboxes" in data
        if self.has_bboxes:
            self.all_bboxes = data["bboxes"]  # (N, 4) int32  x1,y1,x2,y2
            if "frame_size" in data:
                self.frame_w, self.frame_h = data["frame_size"]  # (w, h)
            else:
                self.frame_w, self.frame_h = None, None
        else:
            # Fallback: try loading bboxes from companion JSONL
            self.all_bboxes, self.frame_w, self.frame_h = self._load_bboxes_from_jsonl(
                npz_path, self.all_frames, self.all_track_ids
            )
            self.has_bboxes = self.all_bboxes is not None

        unique_tracks = np.unique(self.all_track_ids)

        if all_tracks:
            # Multi-person mode: index by video frame number
            self.unique_video_frames = np.unique(self.all_frames)
            self.n_frames = len(self.unique_video_frames)
            # Build per-frame lookup: {frame_num: {track_id: {'kp': keypoints, 'bbox': bbox}}}
            self._frame_lookup = {}
            for i in range(len(self.all_frames)):
                fn = self.all_frames[i]
                tid = self.all_track_ids[i]
                if fn not in self._frame_lookup:
                    self._frame_lookup[fn] = {}
                entry = {'kp': self.all_keypoints[i].astype(np.float32)}
                if self.has_bboxes:
                    entry['bbox'] = self.all_bboxes[i]
                self._frame_lookup[fn][tid] = entry
            # Assign colors to tracks
            self._track_colors = {}
            for ci, t in enumerate(sorted(unique_tracks)):
                self._track_colors[t] = self.TRACK_COLORS[ci % len(self.TRACK_COLORS)]
            self.track_id = None
            self.frames = self.unique_video_frames
            print(f"All tracks mode: {len(unique_tracks)} tracks, "
                  f"{self.n_frames} video frames "
                  f"({self.frames.min()}–{self.frames.max()})")
            for t in sorted(unique_tracks):
                n = int(np.sum(self.all_track_ids == t))
                c = self._track_colors[t][0]
                print(f"  Track {t}: {n} frames  ({c})")
            if self.has_bboxes:
                source = "NPZ" if "bboxes" in data else "JSONL fallback"
                print(f"  Bboxes: YES ({source}) → frame-space coordinates"
                      + (f" ({self.frame_w}x{self.frame_h})" if self.frame_w else ""))
            else:
                print("  Bboxes: NO → crop-space coordinates (no bboxes or JSONL)")
            # Apply smoothing if requested
            if smooth:
                self._apply_smoothing_all_tracks()
        else:
            # Single-track mode (original behaviour)
            if track_id is not None:
                assert track_id in unique_tracks, f"Track {track_id} not found. Available: {unique_tracks}"
                self.track_id = track_id
            else:
                counts = [(t, np.sum(self.all_track_ids == t)) for t in unique_tracks]
                self.track_id = max(counts, key=lambda x: x[1])[0]

            mask = self.all_track_ids == self.track_id
            self.frames = self.all_frames[mask]
            self.keypoints = self.all_keypoints[mask].astype(np.float32)
            self.n_frames = len(self.frames)
            print(f"Track {self.track_id}: {self.n_frames} frames "
                  f"({self.frames.min()}–{self.frames.max()}), "
                  f"{len(unique_tracks)} track(s) total")
            # Apply smoothing if requested
            if smooth:
                self._apply_smoothing_single_track()
        # Playback state
        self.idx = max(0, min(start_frame, self.n_frames - 1))
        self.playing = True
        self.interval = interval  # ms per frame
        self.anim = None

        # ── Build figure ──
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("Keypoint Playback")

        # Single full-width panel for pose + face
        gs = self.fig.add_gridspec(6, 1, hspace=0.45,
                                    left=0.06, right=0.96, top=0.92, bottom=0.13)
        self.ax_pose = self.fig.add_subplot(gs[0:5, 0])

        # Slider & buttons
        ax_slider = self.fig.add_axes([0.15, 0.04, 0.55, 0.03])
        self.slider = Slider(ax_slider, "Frame", 0, self.n_frames - 1,
                             valinit=self.idx, valstep=1, valfmt="%d")
        self.slider.on_changed(self._on_slider)

        ax_play = self.fig.add_axes([0.78, 0.03, 0.07, 0.04])
        self.btn_play = Button(ax_play, "⏸ Pause")
        self.btn_play.on_clicked(self._toggle_play)

        ax_speed_lbl = self.fig.add_axes([0.87, 0.03, 0.11, 0.04])
        ax_speed_lbl.set_axis_off()
        self.speed_text = ax_speed_lbl.text(0.0, 0.4, self._speed_label(),
                                             fontsize=9, va="center")

        # Title
        self.title = self.fig.suptitle("", fontsize=13, fontweight="bold")

        # Key bindings
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Draw first frame then start animation
        self._draw_frame(self.idx)
        self.anim = FuncAnimation(self.fig, self._animate, interval=self.interval,
                                  blit=False, cache_frame_data=False)

    # ── Smoothing / jitter filter ─────────────────────────────
    def _frame_centroid(self, kp, bbox=None):
        """Mean x,y of visible pose landmarks, in frame-space when bbox available.

        Using frame-space avoids false positives from crop-window drift:
        crop-space centroids shift even when the person is stationary because
        the YOLO bounding box moves between frames.
        """
        pose = kp[POSE_SLICE]
        visible = pose[:, 2] > 0.01
        if np.sum(visible) < 3:
            return None
        cx = pose[visible, 0].mean()
        cy = pose[visible, 1].mean()
        # Transform to frame-space if bbox available
        if bbox is not None and self.frame_w:
            x1, y1, x2, y2 = bbox
            cx = (cx * (x2 - x1) + x1) / float(self.frame_w)
            cy = (cy * (y2 - y1) + y1) / float(self.frame_h)
        return np.array([cx, cy])

    def _apply_smoothing_all_tracks(self, outlier_radius=5, outlier_thresh=0.12,
                                     window=5):
        """Median-based outlier rejection + temporal smoothing for all tracks.

        Pass 1 (per track): For each frame, compute the frame-space centroid
        and compare against the *median* centroid of a local neighbourhood.
        Frames whose centroid deviates more than ``outlier_thresh`` from the
        local median are replaced with the median keypoints/bboxes of that
        neighbourhood.  This avoids the cascade problem of consecutive-frame
        comparison and correctly handles genuine movement.

        Pass 2: Temporal moving-average over keypoints AND bboxes.
        """
        sorted_frames = sorted(self._frame_lookup.keys())

        # Build per-track ordered lists: tid → [(fn, entry), ...]
        track_frames = {}
        for fn in sorted_frames:
            for tid, entry in self._frame_lookup[fn].items():
                track_frames.setdefault(tid, []).append((fn, entry))

        replaced = 0
        total = 0
        half_r = outlier_radius // 2

        # Pass 1: Median-based outlier replacement (per track)
        for tid, frame_list in track_frames.items():
            n = len(frame_list)
            # Pre-compute frame-space centroids
            centroids = []
            for _fn, entry in frame_list:
                centroids.append(
                    self._frame_centroid(entry['kp'], entry.get('bbox')))
            for i in range(n):
                total += 1
                if centroids[i] is None:
                    # No valid centroid — replace with nearest valid
                    for offset in range(1, n):
                        if i - offset >= 0 and centroids[i - offset] is not None:
                            fn_i = frame_list[i][0]
                            self._frame_lookup[fn_i][tid] = {
                                k: (v.copy() if isinstance(v, np.ndarray) else v)
                                for k, v in frame_list[i - offset][1].items()}
                            frame_list[i] = (fn_i, self._frame_lookup[fn_i][tid])
                            replaced += 1
                            break
                    continue
                # Local neighbourhood
                lo = max(0, i - half_r)
                hi = min(n, i + half_r + 1)
                neighbours = [centroids[j] for j in range(lo, hi)
                              if centroids[j] is not None]
                if len(neighbours) < 2:
                    continue
                median_c = np.median(neighbours, axis=0)
                dist = np.linalg.norm(centroids[i] - median_c)
                if dist > outlier_thresh:
                    # Replace with median keypoints/bboxes from neighbourhood
                    valid_kps = [frame_list[j][1]['kp']
                                 for j in range(lo, hi) if centroids[j] is not None
                                 and j != i]
                    if valid_kps:
                        fn_i = frame_list[i][0]
                        entry_i = self._frame_lookup[fn_i][tid]
                        entry_i['kp'] = np.median(valid_kps, axis=0).astype(np.float32)
                        if self.has_bboxes and 'bbox' in entry_i:
                            valid_bbs = [frame_list[j][1]['bbox'].astype(np.float64)
                                         for j in range(lo, hi)
                                         if centroids[j] is not None and j != i
                                         and 'bbox' in frame_list[j][1]]
                            if valid_bbs:
                                entry_i['bbox'] = np.median(
                                    valid_bbs, axis=0).astype(np.int32)
                        frame_list[i] = (fn_i, entry_i)
                        replaced += 1

        # Pass 2: Temporal smoothing — moving average over keypoints AND bboxes
        half_w = window // 2
        for tid, frame_list in track_frames.items():
            n = len(frame_list)
            if n < window:
                continue
            smoothed_kp = []
            smoothed_bbox = [] if self.has_bboxes else None
            for i in range(n):
                lo = max(0, i - half_w)
                hi = min(n, i + half_w + 1)
                avg_kp = np.mean(
                    [frame_list[j][1]['kp'] for j in range(lo, hi)], axis=0)
                smoothed_kp.append(avg_kp)
                if smoothed_bbox is not None and 'bbox' in frame_list[i][1]:
                    avg_bb = np.mean(
                        [frame_list[j][1]['bbox'].astype(np.float64)
                         for j in range(lo, hi)], axis=0)
                    smoothed_bbox.append(avg_bb.astype(np.int32))
            # Write back
            for i, (fn, _entry) in enumerate(frame_list):
                self._frame_lookup[fn][tid]['kp'] = smoothed_kp[i]
                if smoothed_bbox is not None and i < len(smoothed_bbox):
                    self._frame_lookup[fn][tid]['bbox'] = smoothed_bbox[i]

        print(f"  Smoothing: replaced {replaced}/{total} outliers, "
              f"window={window} frames")

    def _apply_smoothing_single_track(self, outlier_radius=5,
                                      outlier_thresh=0.12, window=5):
        """Median-based outlier rejection + temporal smoothing for single track."""
        n = len(self.keypoints)
        half_r = outlier_radius // 2
        replaced = 0

        # Pre-compute centroids (crop-space only for single track)
        centroids = [self._frame_centroid(self.keypoints[i]) for i in range(n)]

        # Pass 1: Median-based outlier replacement
        kp_copy = self.keypoints.copy()
        for i in range(n):
            if centroids[i] is None:
                # Replace with nearest valid
                for offset in range(1, n):
                    if i - offset >= 0 and centroids[i - offset] is not None:
                        kp_copy[i] = kp_copy[i - offset]
                        replaced += 1
                        break
                    if i + offset < n and centroids[i + offset] is not None:
                        kp_copy[i] = self.keypoints[i + offset]
                        replaced += 1
                        break
                continue
            lo = max(0, i - half_r)
            hi = min(n, i + half_r + 1)
            neighbours = [centroids[j] for j in range(lo, hi)
                          if centroids[j] is not None]
            if len(neighbours) < 2:
                continue
            median_c = np.median(neighbours, axis=0)
            dist = np.linalg.norm(centroids[i] - median_c)
            if dist > outlier_thresh:
                valid_kps = [self.keypoints[j] for j in range(lo, hi)
                             if centroids[j] is not None and j != i]
                if valid_kps:
                    kp_copy[i] = np.median(valid_kps, axis=0).astype(np.float32)
                    replaced += 1
        self.keypoints = kp_copy

        # Pass 2: Temporal smoothing
        half = window // 2
        if n >= window:
            smoothed = np.copy(self.keypoints)
            for i in range(n):
                lo = max(0, i - half)
                hi = min(n, i + half + 1)
                smoothed[i] = np.mean(self.keypoints[lo:hi], axis=0)
            self.keypoints = smoothed

        print(f"  Smoothing: replaced {replaced}/{n} outlier frames, "
              f"window={window} frames")

    # ── JSONL bbox fallback ─────────────────────────────────────
    @staticmethod
    def _load_bboxes_from_jsonl(npz_path, all_frames, all_track_ids):
        """Load bboxes from a companion engagement_data.jsonl when NPZ lacks them.
        
        Applies the same 10% padding used during capture to reconstruct
        the padded bbox that MediaPipe landmarks are relative to.
        
        Returns:
            (bboxes, frame_w, frame_h) or (None, None, None) if no JSONL found.
        """
        jsonl_path = Path(npz_path).parent / "engagement_data.jsonl"
        if not jsonl_path.exists():
            return None, None, None
        
        print(f"  Loading bboxes from companion JSONL: {jsonl_path.name}")
        
        # Build lookup: (frame, track_id) → [x1, y1, x2, y2]
        bbox_lookup = {}
        max_x, max_y = 0, 0
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec['frame'], rec['track_id'])
                bb = rec['bbox']
                bbox_lookup[key] = bb
                max_x = max(max_x, bb[2])
                max_y = max(max_y, bb[3])
        
        # Estimate frame size from bbox extents
        # Common resolutions: snap to nearest standard
        STANDARD_W = [640, 1280, 1920, 2560, 3840]
        STANDARD_H = [480, 720, 1080, 1440, 2160]
        frame_w = min((w for w in STANDARD_W if w >= max_x), default=max_x)
        frame_h = min((h for h in STANDARD_H if h >= max_y), default=max_y)
        
        # Build padded bbox array aligned with NPZ entries
        n = len(all_frames)
        bboxes = np.zeros((n, 4), dtype=np.int32)
        matched = 0
        for i in range(n):
            key = (int(all_frames[i]), int(all_track_ids[i]))
            bb = bbox_lookup.get(key)
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            # Apply 10% padding (same as extract_features)
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(frame_w, x2 + pad_x)
            y2 = min(frame_h, y2 + pad_y)
            bboxes[i] = [x1, y1, x2, y2]
            matched += 1
        
        print(f"  Matched {matched}/{n} entries, estimated frame: {frame_w}x{frame_h}")
        
        if matched == 0:
            return None, None, None
        
        return bboxes, frame_w, frame_h

    # ── Drawing ───────────────────────────────────────────────
    def _draw_frame(self, idx):
        if self.all_tracks_mode:
            self._draw_frame_all_tracks(idx)
        else:
            self._draw_frame_single(idx)

    def _draw_frame_all_tracks(self, idx):
        """Draw all people on the same canvas for this video frame.
        
        When bboxes are available in the NPZ, keypoint x/y are transformed
        from crop-relative [0,1] to frame-relative [0,1] so that each person
        appears at their real position in the scene.
        """
        frame_num = self.frames[idx]
        people = self._frame_lookup.get(frame_num, {})

        # ── Pose + Face (all people) ──
        ax = self.ax_pose
        ax.clear()
        ax.set_title(f"Pose + Face  ({len(people)} people)", fontsize=11)
        for tid, entry in sorted(people.items()):
            pose_c, face_c = self._track_colors[tid]
            kp = entry['kp'].copy()  # (543, 3) — don't mutate original

            # Transform crop-relative → frame-relative coordinates
            if 'bbox' in entry and self.frame_w is not None:
                x1, y1, x2, y2 = entry['bbox']
                fw, fh = float(self.frame_w), float(self.frame_h)
                kp[:, 0] = (kp[:, 0] * (x2 - x1) + x1) / fw
                kp[:, 1] = (kp[:, 1] * (y2 - y1) + y1) / fh

            pose = kp[POSE_SLICE]
            face = kp[FACE_SLICE]
            # Face cloud
            ax.scatter(face[:, 0], face[:, 1], s=0.3, c=face_c, alpha=0.4)
            # Pose skeleton
            ax.scatter(pose[:, 0], pose[:, 1], s=18, c=pose_c, zorder=5)
            for i, j in POSE_CONNECTIONS:
                ax.plot([pose[i, 0], pose[j, 0]], [pose[i, 1], pose[j, 1]],
                        color=pose_c, linewidth=1.4, alpha=0.7)
            # Label
            head_y = pose[0, 1] - 0.03 if pose[0, 1] > 0.05 else pose[0, 1] + 0.05
            ax.text(pose[0, 0], head_y, f"T{tid}", fontsize=8, fontweight="bold",
                    color=pose_c, ha="center", va="bottom")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")

        # Title
        status = "▶" if self.playing else "⏸"
        track_list = ", ".join(f"T{t}" for t in sorted(people.keys()))
        coords = "frame-space" if self.has_bboxes else "crop-space (no bboxes)"
        self.title.set_text(
            f"All Tracks [{track_list}]  |  Frame {frame_num}  "
            f"({idx + 1}/{self.n_frames})  |  {status}  {1000 / self.interval:.0f} fps  |  {coords}"
        )

    def _draw_frame_single(self, idx):
        kp = self.keypoints[idx]
        frame_num = self.frames[idx]

        pose = kp[POSE_SLICE]
        face = kp[FACE_SLICE]

        # ── Pose + Face ──
        ax = self.ax_pose
        ax.clear()
        ax.set_title("Pose + Face", fontsize=11)
        # Face point cloud
        ax.scatter(face[:, 0], face[:, 1], s=0.3, c="lightblue", alpha=0.5)
        # Pose skeleton
        ax.scatter(pose[:, 0], pose[:, 1], s=18, c="red", zorder=5)
        for i, j in POSE_CONNECTIONS:
            ax.plot([pose[i, 0], pose[j, 0]], [pose[i, 1], pose[j, 1]],
                    "r-", linewidth=1.4, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")

        # Title
        status = "▶" if self.playing else "⏸"
        self.title.set_text(
            f"Track {self.track_id}  |  Frame {frame_num}  "
            f"({idx + 1}/{self.n_frames})  |  {status}  {1000 / self.interval:.0f} fps"
        )

    # ── Animation callback ────────────────────────────────────
    def _animate(self, _frame_unused):
        if not self.playing:
            return
        self.idx = (self.idx + 1) % self.n_frames
        self.slider.set_val(self.idx)  # triggers _on_slider → _draw_frame

    # ── Controls ──────────────────────────────────────────────
    def _on_slider(self, val):
        self.idx = int(val)
        self._draw_frame(self.idx)

    def _toggle_play(self, _event=None):
        self.playing = not self.playing
        self.btn_play.label.set_text("▶ Play" if not self.playing else "⏸ Pause")
        self._update_title_status()

    def _on_key(self, event):
        if event.key == " ":
            self._toggle_play()
        elif event.key == "right" and not self.playing:
            self.idx = min(self.idx + 1, self.n_frames - 1)
            self.slider.set_val(self.idx)
        elif event.key == "left" and not self.playing:
            self.idx = max(self.idx - 1, 0)
            self.slider.set_val(self.idx)
        elif event.key == "up":
            self.interval = max(10, self.interval - 10)
            self._update_speed()
        elif event.key == "down":
            self.interval = min(500, self.interval + 10)
            self._update_speed()
        elif event.key == "home":
            self.idx = 0
            self.slider.set_val(self.idx)
        elif event.key == "end":
            self.idx = self.n_frames - 1
            self.slider.set_val(self.idx)
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    def _update_speed(self):
        if self.anim is not None:
            self.anim.event_source.interval = self.interval
        self.speed_text.set_text(self._speed_label())
        self._update_title_status()

    def _update_title_status(self):
        status = "▶" if self.playing else "⏸"
        if self.all_tracks_mode:
            self._draw_frame(self.idx)  # redraws with updated status
        else:
            self.title.set_text(
                f"Track {self.track_id}  |  Frame {self.frames[self.idx]}  "
                f"({self.idx + 1}/{self.n_frames})  |  {status}  {1000 / self.interval:.0f} fps"
            )

    def _speed_label(self):
        return f"{1000 / self.interval:.0f} fps ({self.interval} ms)"

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Animated playback of saved keypoints (NPZ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Controls: Space=play/pause, ←→=step, ↑↓=speed, Home/End=jump, Q=quit",
    )
    parser.add_argument("npz_path", type=str, help="Path to keypoints.npz")
    parser.add_argument("--track", type=int, default=None,
                        help="Track ID to display (default: most frequent)")
    parser.add_argument("--all-tracks", action="store_true", default=False,
                        help="Show all people simultaneously with different colors")
    parser.add_argument("--smooth", action="store_true", default=False,
                        help="Filter erratic tracking glitches and apply temporal smoothing")
    parser.add_argument("--interval", type=int, default=66,
                        help="Milliseconds per frame (default: 66 ≈ 15 fps)")
    parser.add_argument("--start", type=int, default=0,
                        help="Starting frame index (default: 0)")
    args = parser.parse_args()

    player = KeypointPlayer(args.npz_path, track_id=args.track,
                            interval=args.interval, start_frame=args.start,
                            all_tracks=args.all_tracks, smooth=args.smooth)
    player.show()


if __name__ == "__main__":
    main()
