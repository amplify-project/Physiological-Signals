#!/usr/bin/env python3
"""Offline video engagement inference (young-families binary model).

Runs the trained young-families TemporalTransformer over a recorded video and
writes an annotated copy showing, for every detected audience adult:
  * tracked bounding box (green = engaged, red = disengaged)
  * that person's individual engagement score (EMA-smoothed)
and, across the top of the frame, the AVERAGE crowd engagement over all
currently-active people.

This is a TEST harness: unlike the live app (which only scores registered
adults), every YOLO-detected person is scored here so you can eyeball how the
whole audience looks.

Pipeline (mirrors scripts/inference/live_multiperson_binary_v2.py):
  YOLO person track -> per-person crop (10% pad) -> MediaPipe Holistic 543 kpts
  [x, y, visibility] -> flatten to 1629 -> buffer per track -> resample to
  SEQUENCE_LENGTH -> TemporalTransformer -> softmax.

IMPORTANT: for the young-families model, engagement is class INDEX 0
(idx_to_action = {0: 'engagement', 1: 'disengagement'}), so the engagement
probability is probs[0][0].  (The standard model used index 1 - do not copy
that here.)

Usage (PowerShell):
  & "C:\\Program Files\\Python311\\python.exe" scripts\\inference\\video_engagement_offline.py \
      --video "Family Lab Videos\\4th lab video.mp4" \
      --output "Family Lab Videos\\4th lab video_engagement.mp4"
"""

import argparse
import os
import time
from collections import deque, defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# CONFIG
# =============================================================================
NUM_KEYPOINTS = 543
FEATURE_DIM = NUM_KEYPOINTS * 3          # 1629
SEQUENCE_LENGTH = 64                     # young-families model was trained at 64
MIN_FRAMES_FOR_PRED = 24                 # need this many samples before scoring
PREDICT_EVERY = 6                        # run the transformer every N processed frames per track
BUFFER_MAXLEN = 128                      # rolling window of recent feature vectors
EMA_ALPHA = 0.3                          # smoothing for displayed per-person score
STALE_TRACK_FRAMES = 45                  # drop a track unseen for this many processed frames

# engagement is class index 0 for the young-families model
ENGAGEMENT_IDX = 0

# MediaPipe landmark offsets
OFF_POSE = 0
OFF_FACE = 33
OFF_LHAND = 33 + 468        # 501
OFF_RHAND = 33 + 468 + 21   # 522

# graded overlay bands (green/amber/red) - mirrors the gaze-rules overlay policy
PERSON_GREEN = 0.50   # >= green (engaged); between = amber; < PERSON_RED = red
PERSON_RED = 0.30
CROWD_GREEN = 0.75    # crowd banner: green >= 0.75, amber, red < CROWD_RED
CROWD_RED = 0.30

GREEN = (0, 200, 0)
AMBER = (0, 165, 255)
RED = (0, 0, 230)
GREY = (160, 160, 160)


def grade_color(score, green_at=PERSON_GREEN, red_at=PERSON_RED):
    """BGR colour on a green->amber->red gradient (no hard engaged/disengaged flip)."""
    if score >= green_at:
        return GREEN
    if score < red_at:
        return RED
    return AMBER


# =============================================================================
# MODEL (identical architecture to the live/training script)
# =============================================================================
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, num_classes=2, d_model=256, nhead=8,
                 num_layers=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        _, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def load_model(model_path, device):
    model = TemporalTransformer(num_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    clean = {}
    for k, v in state_dict.items():
        clean[k[7:] if k.startswith('module.') else k] = v
    model.load_state_dict(clean)
    model.eval()
    return model


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
def extract_features(holistic, frame_rgb, bbox):
    """Return (543, 3) keypoints from a padded person crop, or zeros."""
    x1, y1, x2, y2 = bbox
    h, w, _ = frame_rgb.shape
    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

    crop = frame_rgb[y1:y2, x1:x2]
    results = holistic.process(crop)
    kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

    def fill(landmarks, offset):
        if landmarks:
            for i, lm in enumerate(landmarks.landmark):
                kp[offset + i] = [lm.x, lm.y,
                                  lm.visibility if hasattr(lm, 'visibility') else 1.0]

    fill(results.pose_landmarks, OFF_POSE)
    fill(results.face_landmarks, OFF_FACE)
    fill(results.left_hand_landmarks, OFF_LHAND)
    fill(results.right_hand_landmarks, OFF_RHAND)
    return kp


def resample(buf, target_len):
    """Linear-interpolate a list of (target_len)-spaced feature vectors."""
    feats = np.stack(buf).astype(np.float32)          # (n, 1629)
    n = feats.shape[0]
    if n == target_len:
        return feats
    src = np.linspace(0.0, 1.0, n)
    tgt = np.linspace(0.0, 1.0, target_len)
    right = np.searchsorted(src, tgt, side='left').clip(1, n - 1)
    left = right - 1
    w = ((tgt - src[left]) / (src[right] - src[left] + 1e-9)).reshape(-1, 1).astype(np.float32)
    return feats[left] * (1.0 - w) + feats[right] * w


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Offline engagement overlay video")
    ap.add_argument('--video', required=True, help='input video path')
    ap.add_argument('--output', default=None, help='annotated output path (default: <video>_engagement.mp4)')
    ap.add_argument('--model', default='models/action_transformer_young_families_v0/best_model.pth')
    ap.add_argument('--yolo', default='yolo11n.pt')
    ap.add_argument('--conf', type=float, default=0.15, help='YOLO person confidence')
    ap.add_argument('--frame-stride', type=int, default=2, help='process every Nth frame (speed)')
    ap.add_argument('--start-seconds', type=float, default=0.0)
    ap.add_argument('--max-seconds', type=float, default=0.0, help='0 = whole video')
    ap.add_argument('--max-persons', type=int, default=25, help='cap MediaPipe passes per frame')
    ap.add_argument('--display-scale', type=float, default=1.0, help='scale output resolution')
    ap.add_argument('--show', action='store_true',
                    help='show a live pop-up window with the overlays for manual validation')
    ap.add_argument('--window-scale', type=float, default=0.6,
                    help='scale of the live pop-up window only (does not affect the saved file)')
    ap.add_argument('--no-save', action='store_true',
                    help='do not write an output file (use with --show for validation only)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    video = args.video
    if not os.path.isfile(video):
        raise SystemExit(f"Video not found: {video}")
    out_path = args.output or os.path.splitext(video)[0] + '_engagement.mp4'

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")
    model = load_model(args.model, device)

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

    out_w = int(src_w * args.display_scale)
    out_h = int(src_h * args.display_scale)
    out_fps = max(1.0, src_fps / max(1, args.frame_stride))
    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))

    win = 'Young Families engagement (q=quit, space=pause)'
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, int(src_w * args.window_scale), int(src_h * args.window_scale))

    print(f"Input : {src_w}x{src_h} @ {src_fps:.1f}fps, {total_frames} frames")
    if writer is not None:
        print(f"Output: {out_w}x{out_h} @ {out_fps:.1f}fps -> {out_path}")
    else:
        print("Output: (not saving; --show validation only)")
    print(f"Stride: every {args.frame_stride} frame(s); engagement_idx={ENGAGEMENT_IDX}")
    if args.show:
        print("Live window: press 'q' to quit, SPACE to pause/resume.")

    # per-track state
    buffers = defaultdict(lambda: deque(maxlen=BUFFER_MAXLEN))
    ema_score = {}            # track_id -> smoothed engagement prob
    last_pred_frame = {}      # track_id -> processed-frame index of last inference
    last_seen = {}            # track_id -> processed-frame index last detected

    raw_idx = 0               # raw source frame counter
    proc_idx = 0              # processed (strided) frame counter
    crowd_history = []        # crowd engagement per scored frame (for end summary)
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
        detections = []  # (track_id, (x1,y1,x2,y2))
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box.astype(int)
                detections.append((int(tid), (x1, y1, x2, y2)))

        # cap number of MediaPipe passes (largest boxes first = closest people)
        detections.sort(key=lambda d: (d[1][2] - d[1][0]) * (d[1][3] - d[1][1]), reverse=True)
        detections = detections[:args.max_persons]

        active_scores = []
        for tid, bbox in detections:
            last_seen[tid] = proc_idx
            kp = extract_features(holistic, frame_rgb, bbox)
            buffers[tid].append(kp.flatten())

            # run inference on a cadence once we have enough samples
            if (len(buffers[tid]) >= MIN_FRAMES_FOR_PRED and
                    proc_idx - last_pred_frame.get(tid, -10**9) >= PREDICT_EVERY):
                seq = resample(buffers[tid], SEQUENCE_LENGTH)
                with torch.no_grad():
                    x = torch.from_numpy(seq).unsqueeze(0).to(device)
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    score = float(probs[0, ENGAGEMENT_IDX].item())
                prev = ema_score.get(tid, score)
                ema_score[tid] = EMA_ALPHA * score + (1.0 - EMA_ALPHA) * prev
                last_pred_frame[tid] = proc_idx

            if tid in ema_score:
                active_scores.append(ema_score[tid])

            # --- draw overlay ---
            x1, y1, x2, y2 = bbox
            if tid in ema_score:
                s = ema_score[tid]
                color = grade_color(s)
                label = f"ID{tid} {s*100:.0f}%"
            else:
                color = GREY
                label = f"ID{tid} ..."
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # evict stale tracks
        for tid in list(last_seen.keys()):
            if proc_idx - last_seen[tid] > STALE_TRACK_FRAMES:
                buffers.pop(tid, None)
                ema_score.pop(tid, None)
                last_pred_frame.pop(tid, None)
                last_seen.pop(tid, None)

        # --- crowd average banner ---
        n_active = len(active_scores)
        crowd = float(np.mean(active_scores)) if active_scores else 0.0
        if n_active:
            crowd_history.append(crowd)
        banner_h = 46
        cv2.rectangle(frame, (0, 0), (src_w, banner_h), (30, 30, 30), -1)
        if n_active:
            bcol = grade_color(crowd, green_at=CROWD_GREEN, red_at=CROWD_RED)
            btxt = f"CROWD ENGAGEMENT: {crowd*100:.0f}%  |  scored adults: {n_active}"
        else:
            bcol = (200, 200, 200)
            btxt = "CROWD ENGAGEMENT: warming up..."
        cv2.putText(frame, btxt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bcol, 2, cv2.LINE_AA)
        # crowd meter
        meter_x = src_w - 260
        if n_active:
            cv2.rectangle(frame, (meter_x, 14), (meter_x + 220, 32), (80, 80, 80), 1)
            cv2.rectangle(frame, (meter_x, 14), (meter_x + int(220 * crowd), 32), bcol, -1)

        if args.display_scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
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
                        if k2 == ord('q'):
                            key = ord('q')
                        break
                if key == ord('q'):
                    print("\nStopped by user (q).")
                    break

        if proc_idx % 50 == 0:
            elapsed = time.time() - t0
            fps_proc = proc_idx / elapsed if elapsed else 0
            print(f"  processed {proc_idx} frames (raw {raw_idx}/{total_frames}) "
                  f"| {fps_proc:.1f} proc-fps | active {n_active} | crowd {crowd*100:.0f}%")

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    holistic.close()
    dt = time.time() - t0
    if writer is not None:
        print(f"\nDone. Wrote {out_path}")
    else:
        print("\nDone (no file saved).")
    print(f"Processed {proc_idx} frames in {dt/60:.1f} min ({proc_idx/dt:.1f} proc-fps)")

    # --- engagement summary (smoke-test verification) ---
    if crowd_history:
        ch = np.array(crowd_history, dtype=np.float32)
        pct_engaged_frames = float(np.mean(ch >= 0.5) * 100.0)
        print("\n=== ENGAGEMENT SUMMARY ===")
        print(f"model            : {args.model}")
        print(f"scored frames    : {len(ch)}")
        print(f"mean crowd eng   : {ch.mean()*100:.1f}%")
        print(f"median crowd eng : {float(np.median(ch))*100:.1f}%")
        print(f"min / max        : {ch.min()*100:.1f}% / {ch.max()*100:.1f}%")
        print(f"frames >= 50%    : {pct_engaged_frames:.1f}% (majority-engaged share)")
        if ema_score:
            per = np.array(list(ema_score.values()), dtype=np.float32)
            print(f"final per-person : n={len(per)} mean={per.mean()*100:.1f}% "
                  f"engaged={int((per >= 0.5).sum())}/{len(per)}")
    else:
        print("\n=== ENGAGEMENT SUMMARY ===\n(no frames scored - no people detected/tracked long enough)")


if __name__ == '__main__':
    main()
