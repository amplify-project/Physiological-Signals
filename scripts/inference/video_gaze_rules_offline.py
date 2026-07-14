#!/usr/bin/env python3
"""Offline gaze-rules engagement overlay (Option A - no learned model).

Implements the 2D gaze-first engagement rules from ROADMAP.md
("Gaze-first retrain (observations from '4th lab video.mp4')"):

  * Per-person 2D gaze vector = straight vector out from the front of the face
    (ear-midpoint -> nose in image space; single-ear profile fallback).
  * Common focal point = least-squares intersection of audience gaze rays,
    valid only when a MAJORITY (> QUORUM_FRAC) of the seated audience members
    with a measurable gaze ray converge on it (absolute floor MIN_RAYS).
  * Engagement rules (presume engaged, subtract on anti-cues):
      - gaze departs from the common focal point            -> disengaged
      - staring at feet / down at the floor                 -> disengaged
      - sudden synchronized gaze shift of several people    -> distraction,
        shifted people disengaged for a cooldown
      - no common focal point                               -> concert not
        started yet (PRE-SHOW; nobody is scored)
      - body movement is NEVER penalised while gaze stays on the focal point
  * Standing adults become PERFORMERS (sticky: once a performer, always a
    performer, even after sitting). Performers are excluded from audience
    scoring and from the focal-point estimate. A performer holding the crowd
    focal point gets an ORANGE bounding box.

Overlay: per-person gaze lines projected out to the common focal point so the
convergence is visible, orange performer boxes, green/red audience boxes,
crowd banner.

Pipeline (harness mirrors scripts/inference/video_engagement_offline.py):
  YOLO person track -> per-person crop (10% pad) -> MediaPipe Holistic ->
  head/leg geometry -> gaze rules engine -> overlay.

No torch / no transformer: pure geometry + rules.

Usage (PowerShell):
  python scripts\inference\video_gaze_rules_offline.py `
      --video "Family Lab Videos\4th lab video.mp4" --show --no-save
"""

import argparse
import math
import os
import time
from collections import deque, defaultdict

import cv2
import numpy as np


# =============================================================================
# CONFIG (tunable rule thresholds)
# =============================================================================
# --- gaze geometry ---
VIS_THRESH = 0.3                 # min MediaPipe landmark visibility (posture)
HEAD_VIS_THRESH = 0.6            # stricter bar for head landmarks used for gaze
HEAD_MARGIN = 0.05               # nose/ears/eyes may exceed the crop by this fraction before
                                 # the gaze is rejected as a bad detection (stray rays)
HEAD_TOP_FRAC = 0.45             # nose must sit in the top fraction of the crop (head region)
AMBIG_FRAC = 0.15                # |nose-earmid| < this * ear_dist => facing camera/away (no 2D dir)
FLOOR_GAZE_DY = 0.80             # unit-dir y-component above this => staring at floor
FLOOR_PITCH_FRAC = 0.80          # nose below ear-line by this * head_size also => floor gaze

# --- focal point ---
MIN_RAYS = 3                     # absolute floor of valid audience rays for a focal point
QUORUM_FRAC = 0.5                # focal point needs > this fraction of the seated audience
                                 # (members with a measurable gaze ray) to converge on it
MAX_MEAN_DEV_DEG = 30.0          # mean ray->focal angular deviation above this => "no focal point"
FOCAL_EMA = 0.25                 # smoothing of the focal point position

# --- per-person engagement rules ---
THETA_OK_DEG = 25.0              # gaze within this angle of the focal point => engaged
SCORE_EMA = 0.15                 # smoothing of the displayed 0..1 score
ENGAGED_THRESH = 0.5

# --- synchronized gaze-shift (off-stage distraction) ---
SHIFT_WINDOW = 5                 # processed frames over which a direction change is measured
SHIFT_DEG = 30.0                 # per-person direction change counting as a "shift"
SHIFT_MIN_PEOPLE = 3             # at least this many people ...
SHIFT_MIN_FRAC = 0.4             # ... and this fraction of valid audience must shift together
SHIFT_COOLDOWN = 20              # processed frames the distraction event lasts

# --- performer detection (standing adults; sticky) ---
LEG_TORSO_RATIO = 1.15           # (hip->ankle) > ratio * (shoulder->hip) => legs extended
KNEE_DROP_FRAC = 0.55            # (hip->knee) > frac * torso => not folded (seated knees ~ hip level)
PERFORMER_STAND_FRAMES = 10      # consecutive standing frames before promotion (sticky)
PERFORMER_BOX_PAD = 0.20         # focal point within performer bbox padded by this => "gazed at"

# --- harness ---
NUM_KEYPOINTS = 543
STALE_TRACK_FRAMES = 45

# MediaPipe pose landmark indices (within the 543 layout, pose block is 0..32)
NOSE, L_EYE, R_EYE = 0, 2, 5
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# colors (BGR)
COL_ENGAGED = (0, 200, 0)
COL_DISENGAGED = (0, 0, 230)
COL_WARMING = (160, 160, 160)
COL_PERFORMER = (0, 140, 255)        # orange
COL_GAZE = (255, 220, 0)             # cyan-ish gaze rays


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================
def unit(v):
    n = float(np.hypot(v[0], v[1]))
    return (v / n, n) if n > 1e-6 else (v, 0.0)


def ang_between(d1, d2):
    """Angle in degrees between two 2D unit vectors."""
    dot = float(np.clip(d1[0] * d2[0] + d1[1] * d2[1], -1.0, 1.0))
    return math.degrees(math.acos(dot))


def least_squares_focal(points, dirs):
    """Least-squares closest point to a set of 2D rays (p_i, d_i unit).

    Solves sum_i (I - d d^T)(x - p) = 0. Discards rays whose focal point lies
    behind them (t < 0) and re-solves once. Returns (focal, contributors) or
    (None, []).
    """
    idx = list(range(len(points)))
    for _ in range(2):
        A = np.zeros((2, 2))
        b = np.zeros(2)
        for i in idx:
            d = dirs[i]
            M = np.eye(2) - np.outer(d, d)
            A += M
            b += M @ points[i]
        if abs(np.linalg.det(A)) < 1e-9:
            return None, []
        x = np.linalg.solve(A, b)
        infront = [i for i in idx if np.dot(dirs[i], x - points[i]) > 0]
        if len(infront) == len(idx) or len(infront) < MIN_RAYS:
            return x, infront
        idx = infront
    return x, idx


# =============================================================================
# PER-PERSON GAZE / POSTURE EXTRACTION
# =============================================================================
def person_geometry(kp, pbox):
    """Derive gaze + posture from a (543,3) crop-normalized keypoint array.

    pbox = padded crop (x1, y1, x2, y2) in frame pixels, used to map
    crop-normalized landmarks into frame coordinates.

    Returns dict with:
      anchor    (x, y) frame px of the nose (gaze origin), or None
      gaze      unit 2D direction in frame px, or None (ambiguous / no face)
      head_size head scale in frame px (for drawing / pitch checks)
      floor     True if staring down at the floor
      standing  True if legs extended below torso (standing adult)
    """
    x1, y1, x2, y2 = pbox
    cw, ch = max(1, x2 - x1), max(1, y2 - y1)

    def pt(i):
        return np.array([x1 + kp[i, 0] * cw, y1 + kp[i, 1] * ch])

    def vis(i):
        return kp[i, 2]

    out = {'anchor': None, 'gaze': None, 'head_size': 0.25 * cw,
           'floor': False, 'standing': False}

    # MediaPipe can return landmarks outside the crop (normalized coords beyond
    # [0,1]) on poor detections, and in dense crowds a padded crop may contain a
    # NEIGHBOUR's face, which MediaPipe latches onto - both produce stray gaze
    # rays. Require head landmarks well-visible, inside the crop (+margin), the
    # nose in the upper part of the box, and an eye corroborating each ear.
    def head_ok(i):
        nx, ny = kp[i, 0], kp[i, 1]
        return (kp[i, 2] >= HEAD_VIS_THRESH and
                -HEAD_MARGIN <= nx <= 1.0 + HEAD_MARGIN and
                -HEAD_MARGIN <= ny <= 1.0 + HEAD_MARGIN)

    # ---- gaze: ear-midpoint -> nose (or single-ear profile fallback) ----
    if head_ok(NOSE) and 0.0 <= kp[NOSE, 0] <= 1.0 and kp[NOSE, 1] <= HEAD_TOP_FRAC:
        nose = pt(NOSE)
        out['anchor'] = (float(nose[0]), float(nose[1]))
        # each ear must be corroborated by the eye on the same side, otherwise
        # a lone misdetected ear yields a wild gaze direction
        l_ok = head_ok(L_EAR) and head_ok(L_EYE)
        r_ok = head_ok(R_EAR) and head_ok(R_EYE)
        ear_mid = ear_dist = None
        if l_ok and r_ok:
            le, re = pt(L_EAR), pt(R_EAR)
            ear_mid = 0.5 * (le + re)
            ear_dist = float(np.hypot(*(le - re)))
        elif l_ok or r_ok:
            ear_mid = pt(L_EAR if l_ok else R_EAR)
            ear_dist = 0.5 * out['head_size']     # profile: scale from bbox
        if ear_mid is not None:
            out['head_size'] = max(ear_dist, 1.0)
            d, n = unit(nose - ear_mid)
            if n >= AMBIG_FRAC * max(ear_dist, 1.0):
                out['gaze'] = (float(d[0]), float(d[1]))
                # floor gaze: steep downward ray, or nose well below ear line
                pitch_drop = (nose[1] - ear_mid[1]) / max(ear_dist, 1.0)
                out['floor'] = d[1] > FLOOR_GAZE_DY or pitch_drop > FLOOR_PITCH_FRAC

    # ---- posture: standing vs seated ----
    need = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
    if all(vis(i) >= VIS_THRESH for i in need):
        sho_y = 0.5 * (pt(L_SHOULDER)[1] + pt(R_SHOULDER)[1])
        hip_y = 0.5 * (pt(L_HIP)[1] + pt(R_HIP)[1])
        knee_y = 0.5 * (pt(L_KNEE)[1] + pt(R_KNEE)[1])
        ank_y = 0.5 * (pt(L_ANKLE)[1] + pt(R_ANKLE)[1])
        torso = hip_y - sho_y
        if torso > 1.0:
            legs_extended = (ank_y - hip_y) > LEG_TORSO_RATIO * torso
            knees_dropped = (knee_y - hip_y) > KNEE_DROP_FRAC * torso
            out['standing'] = bool(legs_extended and knees_dropped)
    return out


# =============================================================================
# GAZE RULES ENGINE
# =============================================================================
class GazeRulesEngine:
    """Population-level gaze rules: focal point, sync-shift, performer set,
    per-person rule-based engagement scores."""

    def __init__(self):
        self.performers = set()                 # sticky performer track ids
        self.stand_streak = defaultdict(int)    # tid -> consecutive standing frames
        self.dir_hist = defaultdict(lambda: deque(maxlen=SHIFT_WINDOW + 1))
        self.score = {}                         # tid -> EMA engagement 0..1
        self.focal = None                       # smoothed focal point (np.array)
        self.focal_valid = False
        self.shift_until = -1                   # distraction cooldown deadline
        self.shifted_tids = set()

    def evict(self, tid):
        self.stand_streak.pop(tid, None)
        self.dir_hist.pop(tid, None)
        self.score.pop(tid, None)
        self.shifted_tids.discard(tid)
        # NOTE: performer stickiness intentionally survives eviction only while
        # the track id is stable; a re-detected performer gets a fresh id and
        # must be re-promoted. Acceptable for v1.
        self.performers.discard(tid)

    def update(self, people, proc_idx):
        """people: {tid: {'geom': ..., 'bbox': (x1,y1,x2,y2)}}.

        Returns dict with focal point, per-tid status, gazed performer ids and
        the distraction flag.
        """
        # ---- 1. performer promotion (standing adults; sticky) ----
        for tid, p in people.items():
            if p['geom']['standing']:
                self.stand_streak[tid] += 1
                if self.stand_streak[tid] >= PERFORMER_STAND_FRAMES:
                    self.performers.add(tid)
            else:
                self.stand_streak[tid] = 0

        audience = {tid: p for tid, p in people.items() if tid not in self.performers}

        # ---- 2. common focal point from audience gaze rays ----
        pts, dirs, ray_tids = [], [], []
        for tid, p in audience.items():
            g = p['geom']
            if g['gaze'] is not None and g['anchor'] is not None and not g['floor']:
                pts.append(np.array(g['anchor']))
                dirs.append(np.array(g['gaze']))
                ray_tids.append(tid)

        raw_focal, contributors = (None, [])
        quorum = max(MIN_RAYS, int(math.ceil(QUORUM_FRAC * len(pts)))) if pts else MIN_RAYS
        if len(pts) >= MIN_RAYS:
            raw_focal, contributors = least_squares_focal(pts, dirs)
            if raw_focal is not None and len(contributors) >= quorum:
                devs = [ang_between(dirs[i], unit(raw_focal - pts[i])[0])
                        for i in contributors]
                if float(np.mean(devs)) > MAX_MEAN_DEV_DEG:
                    raw_focal = None          # rays don't converge
            else:
                raw_focal = None              # below majority quorum

        if raw_focal is not None:
            self.focal = (raw_focal if self.focal is None
                          else FOCAL_EMA * raw_focal + (1 - FOCAL_EMA) * self.focal)
            self.focal_valid = True
        else:
            self.focal_valid = False          # PRE-SHOW / lost convergence

        # ---- 3. synchronized gaze-shift detection ----
        shifted_now = set()
        n_valid = 0
        for tid, p in audience.items():
            g = p['geom']['gaze']
            if g is None:
                continue
            n_valid += 1
            hist = self.dir_hist[tid]
            hist.append(np.array(g))
            if len(hist) > SHIFT_WINDOW and ang_between(hist[0], hist[-1]) > SHIFT_DEG:
                shifted_now.add(tid)
        if (len(shifted_now) >= SHIFT_MIN_PEOPLE and n_valid > 0 and
                len(shifted_now) / n_valid >= SHIFT_MIN_FRAC):
            # NOTE v1: we don't yet verify the NEW directions re-converge on a
            # common off-stage point; any mass shift counts as a distraction.
            self.shift_until = proc_idx + SHIFT_COOLDOWN
            self.shifted_tids = set(shifted_now)
        distraction = proc_idx <= self.shift_until

        # ---- 4. per-person rules -> instantaneous engaged/neutral/disengaged ----
        status = {}
        for tid, p in audience.items():
            g = p['geom']
            inst = None                        # None = neutral, don't move score
            reason = ''
            if g['floor']:
                inst, reason = 0.0, 'floor'
            elif distraction and tid in self.shifted_tids:
                inst, reason = 0.0, 'shift'
            elif self.focal_valid and g['gaze'] is not None and g['anchor'] is not None:
                to_focal, dist = unit(self.focal - np.array(g['anchor']))
                if dist < 2.0 * g['head_size']:
                    inst, reason = 1.0, 'at-focal'   # person effectively AT the focal point
                else:
                    dev = ang_between(np.array(g['gaze']), to_focal)
                    inst = 1.0 if dev <= THETA_OK_DEG else 0.0
                    reason = 'on-focal' if inst else 'off-focal'
            # movement is never penalised: no motion term anywhere.
            if inst is not None:
                prev = self.score.get(tid, inst)
                self.score[tid] = SCORE_EMA * inst + (1 - SCORE_EMA) * prev
            status[tid] = {'score': self.score.get(tid), 'reason': reason,
                           'contributed': tid in [ray_tids[i] for i in contributors]}

        # ---- 5. which performer(s) hold the focal point ----
        gazed = set()
        if self.focal_valid and self.focal is not None:
            fx, fy = self.focal
            for tid in self.performers:
                if tid not in people:
                    continue
                x1, y1, x2, y2 = people[tid]['bbox']
                px = PERFORMER_BOX_PAD * (x2 - x1)
                py = PERFORMER_BOX_PAD * (y2 - y1)
                if x1 - px <= fx <= x2 + px and y1 - py <= fy <= y2 + py:
                    gazed.add(tid)

        return {
            'focal': tuple(self.focal) if (self.focal_valid and self.focal is not None) else None,
            'started': self.focal_valid,
            'distraction': distraction,
            'status': status,
            'gazed_performers': gazed,
        }


# =============================================================================
# FEATURE EXTRACTION (MediaPipe on padded person crop)
# =============================================================================
def extract_keypoints(holistic, frame_rgb, bbox):
    """Return ((543,3) crop-normalized keypoints, padded bbox) or (None, None)."""
    x1, y1, x2, y2 = bbox
    h, w, _ = frame_rgb.shape
    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None, None

    results = holistic.process(frame_rgb[y1:y2, x1:x2])
    kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            kp[i] = [lm.x, lm.y, lm.visibility if hasattr(lm, 'visibility') else 1.0]
    else:
        return None, None
    return kp, (x1, y1, x2, y2)


# =============================================================================
# OVERLAY DRAWING
# =============================================================================
def draw_gaze_rays(frame, people, focal):
    """Plain gaze lines (no arrowheads). When a common focal point exists,
    every ray is projected out to its closest approach to that point so the
    convergence is visible; otherwise a short stub is drawn."""
    for tid, p in people.items():
        g = p['geom']
        if g['gaze'] is None or g['anchor'] is None:
            continue
        a = np.array(g['anchor'], dtype=float)
        d = np.array(g['gaze'], dtype=float)
        L = 2.2 * g['head_size']                  # fallback: short stub
        if focal is not None:
            t_focal = float(np.dot(np.array(focal) - a, d))
            if t_focal > 0:
                L = t_focal
        end = (int(a[0] + d[0] * L), int(a[1] + d[1] * L))
        cv2.line(frame, (int(a[0]), int(a[1])), end, COL_GAZE, 2, cv2.LINE_AA)


def draw_person(frame, tid, bbox, geom, is_performer, is_gazed, stat):
    x1, y1, x2, y2 = bbox
    if is_performer:
        color = COL_PERFORMER
        thick = 4 if is_gazed else 2
        label = f"ID{tid} PERFORMER" + (" <<" if is_gazed else "")
    elif stat is None or stat['score'] is None:
        color, thick = COL_WARMING, 2
        label = f"ID{tid} ..."
    else:
        s = stat['score']
        engaged = s >= ENGAGED_THRESH
        color = COL_ENGAGED if engaged else COL_DISENGAGED
        thick = 2
        label = f"ID{tid} {'ENG' if engaged else 'DIS'} {s*100:.0f}%"
        if stat['reason'] in ('floor', 'shift', 'off-focal'):
            label += f" [{stat['reason']}]"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_global(frame, result, n_scored, src_w):
    banner_h = 46
    cv2.rectangle(frame, (0, 0), (src_w, banner_h), (30, 30, 30), -1)
    if not result['started']:
        txt, col = "PRE-SHOW: no common gaze focal point", (200, 200, 200)
        crowd = None
    else:
        scores = [s['score'] for s in result['status'].values() if s['score'] is not None]
        crowd = float(np.mean(scores)) if scores else 0.0
        col = COL_ENGAGED if crowd >= ENGAGED_THRESH else (0, 120, 230)
        txt = f"CROWD ENGAGEMENT (gaze rules): {crowd*100:.0f}%  |  scored: {n_scored}"
        if result['distraction']:
            txt += "  |  SYNC GAZE SHIFT!"
            col = (0, 165, 255)
    cv2.putText(frame, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    if crowd is not None:
        meter_x = src_w - 260
        cv2.rectangle(frame, (meter_x, 14), (meter_x + 220, 32), (80, 80, 80), 1)
        cv2.rectangle(frame, (meter_x, 14), (meter_x + int(220 * crowd), 32), col, -1)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Offline gaze-rules engagement overlay")
    ap.add_argument('--video', required=True, help='input video path')
    ap.add_argument('--output', default=None,
                    help='annotated output path (default: <video>_gaze_rules.mp4)')
    ap.add_argument('--yolo', default='yolo11n.pt')
    ap.add_argument('--conf', type=float, default=0.15, help='YOLO person confidence')
    ap.add_argument('--frame-stride', type=int, default=2, help='process every Nth frame')
    ap.add_argument('--start-seconds', type=float, default=0.0)
    ap.add_argument('--max-seconds', type=float, default=0.0, help='0 = whole video')
    ap.add_argument('--max-persons', type=int, default=25, help='cap MediaPipe passes per frame')
    ap.add_argument('--show', action='store_true', help='live pop-up window')
    ap.add_argument('--window-scale', type=float, default=0.6)
    ap.add_argument('--no-save', action='store_true', help='do not write an output file')
    args = ap.parse_args()

    video = args.video
    if not os.path.isfile(video):
        raise SystemExit(f"Video not found: {video}")
    out_path = args.output or os.path.splitext(video)[0] + '_gaze_rules.mp4'

    from ultralytics import YOLO
    import mediapipe as mp
    print(f"Loading YOLO: {args.yolo}")
    yolo = YOLO(args.yolo)
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000.0)
    end_ms = (args.start_seconds + args.max_seconds) * 1000.0 if args.max_seconds > 0 else None

    out_fps = max(1.0, src_fps / max(1, args.frame_stride))
    writer = None
    if not args.no_save:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 out_fps, (src_w, src_h))

    win = 'Gaze rules engagement (q=quit, space=pause)'
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, int(src_w * args.window_scale), int(src_h * args.window_scale))

    print(f"Input : {src_w}x{src_h} @ {src_fps:.1f}fps, {total_frames} frames")
    print(f"Output: {'(not saving)' if writer is None else out_path}")

    engine = GazeRulesEngine()
    last_seen = {}
    raw_idx = proc_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if end_ms is not None and cap.get(cv2.CAP_PROP_POS_MSEC) > end_ms:
            break
        if raw_idx % args.frame_stride != 0:
            raw_idx += 1
            continue
        raw_idx += 1
        proc_idx += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- YOLO person tracking ---
        results = yolo.track(frame, persist=True, verbose=False, classes=[0], conf=args.conf)
        detections = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box.astype(int)
                detections.append((int(tid), (x1, y1, x2, y2)))
        detections.sort(key=lambda d: (d[1][2] - d[1][0]) * (d[1][3] - d[1][1]), reverse=True)
        detections = detections[:args.max_persons]

        # --- per-person geometry ---
        people = {}
        for tid, bbox in detections:
            last_seen[tid] = proc_idx
            kp, pbox = extract_keypoints(holistic, frame_rgb, bbox)
            if kp is None:
                continue
            people[tid] = {'geom': person_geometry(kp, pbox), 'bbox': bbox}

        # --- rules engine ---
        result = engine.update(people, proc_idx)

        # --- evict stale tracks ---
        for tid in list(last_seen.keys()):
            if proc_idx - last_seen[tid] > STALE_TRACK_FRAMES:
                engine.evict(tid)
                last_seen.pop(tid, None)

        # --- overlay ---
        n_scored = sum(1 for s in result['status'].values() if s['score'] is not None)
        draw_gaze_rays(frame, people, result['focal'])
        for tid, p in people.items():
            draw_person(frame, tid, p['bbox'], p['geom'],
                        tid in engine.performers,
                        tid in result['gazed_performers'],
                        result['status'].get(tid))
        draw_global(frame, result, n_scored, src_w)

        if writer is not None:
            writer.write(frame)

        if args.show:
            disp = frame
            if args.window_scale != 1.0:
                disp = cv2.resize(frame, None, fx=args.window_scale, fy=args.window_scale,
                                  interpolation=cv2.INTER_AREA)
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopped by user (q).")
                break
            if key == ord(' '):
                while True:
                    k2 = cv2.waitKey(50) & 0xFF
                    if k2 in (ord(' '), ord('q')):
                        key = k2
                        break
                if key == ord('q'):
                    print("\nStopped by user (q).")
                    break

        if proc_idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"  processed {proc_idx} (raw {raw_idx}/{total_frames}) "
                  f"| {proc_idx/elapsed:.1f} proc-fps | people {len(people)} "
                  f"| performers {len(engine.performers)} "
                  f"| {'LIVE' if result['started'] else 'PRE-SHOW'}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nDone. Wrote {out_path}")
    else:
        print("\nDone (no file saved).")
    if args.show:
        cv2.destroyAllWindows()
    holistic.close()
    dt = time.time() - t0
    print(f"Processed {proc_idx} frames in {dt/60:.1f} min ({proc_idx/max(dt,1e-9):.1f} proc-fps)")


if __name__ == '__main__':
    main()
