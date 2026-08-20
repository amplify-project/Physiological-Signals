#!/usr/bin/env python3
"""Offline gaze-rules engagement overlay (Option A - no learned model).

Harness around the shared gaze_rules module (see gaze_rules.py for the full
rule set and tunable thresholds): YOLO person track -> per-person crop
(10% pad) -> MediaPipe Holistic -> head/leg geometry -> gaze rules engine ->
overlay. No torch / no transformer: pure geometry + rules.

Usage (PowerShell):
  python scripts\video_gaze_rules_offline.py `
      --video "Family Lab Videos\4th lab video.mp4" --show --no-save
"""

import argparse
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

import gaze_rules as gr
from gaze_rules import (
    NUM_KEYPOINTS, STALE_TRACK_FRAMES, ENGAGED_THRESH, RED_THRESH,
    COL_ENGAGED, COL_PARTIAL, COL_LOW, COL_WARMING, COL_PERFORMER,
    GazeRulesEngine, person_geometry, appearance_sig, draw_gaze_rays,
)


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
        if s >= ENGAGED_THRESH:
            color = COL_ENGAGED
        elif s >= RED_THRESH:
            color = COL_PARTIAL          # partially engaged, not "disengaged"
        else:
            color = COL_LOW
        thick = 2
        label = f"ID{tid} {s*100:.0f}%"
        if stat['reason'] and any(k in stat['reason'] for k in ('shift', 'off-focal', 'act')):
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
        if crowd >= 0.75:
            col = COL_ENGAGED
        elif crowd >= RED_THRESH:
            col = COL_PARTIAL
        else:
            col = COL_LOW
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
    ap.add_argument('--imgsz', type=int, default=1280,
                    help='YOLO inference size; 640 downscales 1080p wide shots so far-away '
                         'seated people (and performers) drop out of detection entirely')
    ap.add_argument('--tracker', default=str(Path(__file__).parent / 'botsort_gaze.yaml'),
                    help='Ultralytics tracker config (default: strided-video BoT-SORT tune '
                         'with a long lost-track buffer to curb id inflation)')
    ap.add_argument('--frame-stride', type=int, default=2, help='process every Nth frame')
    ap.add_argument('--start-seconds', type=float, default=0.0)
    ap.add_argument('--max-seconds', type=float, default=0.0, help='0 = whole video')
    ap.add_argument('--max-persons', type=int, default=25, help='cap MediaPipe passes per frame')
    ap.add_argument('--show', action='store_true', help='live pop-up window')
    ap.add_argument('--window-scale', type=float, default=0.6)
    ap.add_argument('--quorum', type=float, default=None,
                    help=f'override QUORUM_FRAC (default {gr.QUORUM_FRAC}): fraction of audience '
                         'gaze rays that must converge for a valid focal point')
    ap.add_argument('--no-save', action='store_true', help='do not write an output file')
    ap.add_argument('--action-model', default=None,
                    help='optional action-transformer checkpoint; its DISENGAGEMENT probability '
                         'is subtracted from the gaze (presume-engaged) score - the action head '
                         'acts purely as a distraction/distress detector on top of the gaze base')
    ap.add_argument('--dis-tau', type=float, default=0.60,
                    help='disengagement prob must exceed this before ANY action penalty applies')
    ap.add_argument('--action-penalty-max', type=float, default=0.7,
                    help='max score the action head can subtract (reached at disengagement prob 1.0)')
    ap.add_argument('--device', default=None, help='torch device for the action model (auto)')
    args = ap.parse_args()

    if args.quorum is not None:
        gr.QUORUM_FRAC = args.quorum

    video = args.video
    if not os.path.isfile(video):
        raise SystemExit(f"Video not found: {video}")
    out_path = args.output or os.path.splitext(video)[0] + '_gaze_rules.mp4'

    from ultralytics import YOLO
    import mediapipe as mp
    print(f"Loading YOLO: {args.yolo}")
    yolo = YOLO(args.yolo)
    holistic = mp.solutions.holistic.Holistic(
        # static_image_mode=True: we call process() on DIFFERENT people's crops
        # back-to-back (and skip frames between rounds). Tracking mode
        # (static_image_mode=False) carries landmark state from the previous
        # call - i.e. from a DIFFERENT person - producing stray gaze rays.
        static_image_mode=True, model_complexity=1,
        min_detection_confidence=0.5,
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

    # optional action head = a DISENGAGEMENT detector; it only ever SUBTRACTS
    # from the gaze presume-engaged base (it never sets the baseline).
    action_model = a_device = None
    a_buffers = a_penalty = a_last_pred = a_cfg = None
    if args.action_model:
        import torch
        from video_engagement_offline import (
            load_model, resample, ENGAGEMENT_IDX, SEQUENCE_LENGTH,
            MIN_FRAMES_FOR_PRED, PREDICT_EVERY, BUFFER_MAXLEN,
        )
        a_device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Loading action model: {args.action_model} (device {a_device})")
        action_model = load_model(args.action_model, a_device)
        a_buffers = defaultdict(lambda: deque(maxlen=BUFFER_MAXLEN))
        a_penalty = {}
        a_last_pred = {}
        a_cfg = dict(torch=torch, resample=resample, dis_idx=1 - ENGAGEMENT_IDX,
                     seq=SEQUENCE_LENGTH, min_frames=MIN_FRAMES_FOR_PRED, every=PREDICT_EVERY)

    last_seen = {}
    raw_idx = proc_idx = 0
    crowd_history = []
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
        results = yolo.track(frame, persist=True, verbose=False, classes=[0],
                             conf=args.conf, imgsz=args.imgsz, tracker=args.tracker)
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
            people[tid] = {'geom': person_geometry(kp, pbox), 'bbox': bbox,
                           'sig': appearance_sig(frame, bbox)}
            if action_model is not None:
                a_buffers[tid].append(kp.flatten())

        # --- rules engine ---
        result = engine.update(people, proc_idx)

        # --- evict stale tracks ---
        for tid in list(last_seen.keys()):
            if proc_idx - last_seen[tid] > STALE_TRACK_FRAMES:
                engine.evict(tid)
                last_seen.pop(tid, None)
                if action_model is not None:
                    a_buffers.pop(tid, None)
                    a_penalty.pop(tid, None)
                    a_last_pred.pop(tid, None)

        # --- action head: subtract a DISENGAGEMENT penalty from the gaze score ---
        status_draw = result['status']
        if action_model is not None:
            torch = a_cfg['torch']
            for tid in people:
                buf = a_buffers[tid]
                if (len(buf) >= a_cfg['min_frames'] and
                        proc_idx - a_last_pred.get(tid, -10**9) >= a_cfg['every']):
                    seq = a_cfg['resample'](buf, a_cfg['seq'])
                    with torch.no_grad():
                        x = torch.from_numpy(seq).unsqueeze(0).to(a_device)
                        probs = torch.softmax(action_model(x), dim=1)
                        p_dis = float(probs[0, a_cfg['dis_idx']].item())
                    # only CONFIDENT disengagement bites; scale linearly past the threshold
                    pen = max(0.0, (p_dis - args.dis_tau) / max(1e-6, 1.0 - args.dis_tau))
                    pen *= args.action_penalty_max
                    prev = a_penalty.get(tid, 0.0)
                    alpha = 0.4 if pen >= prev else 0.25   # ease penalty in, release it slower
                    a_penalty[tid] = alpha * pen + (1.0 - alpha) * prev
                    a_last_pred[tid] = proc_idx
            status_draw = {}
            for tid, st in result['status'].items():
                if st is None or st['score'] is None:
                    status_draw[tid] = st
                    continue
                pen = a_penalty.get(tid, 0.0)
                reason = st['reason']
                if pen > 0.05:
                    reason = (reason + '+act') if reason and reason != 'at-focal' else 'act'
                status_draw[tid] = {'score': max(0.0, min(1.0, st['score'] - pen)),
                                    'reason': reason, 'contributed': st.get('contributed')}

        # --- overlay ---
        result_draw = dict(result)
        result_draw['status'] = status_draw
        n_scored = sum(1 for s in status_draw.values() if s and s['score'] is not None)
        performer_boxes = [people[t]['bbox'] for t in engine.performers if t in people]
        draw_gaze_rays(frame, people, result['focal'], performer_boxes, engine.performers)
        for tid, p in people.items():
            draw_person(frame, tid, p['bbox'], p['geom'],
                        tid in engine.performers,
                        tid in result['gazed_performers'],
                        status_draw.get(tid))
        draw_global(frame, result_draw, n_scored, src_w)

        # crowd history for the end-of-run summary
        if result['started']:
            cs = [s['score'] for s in status_draw.values() if s and s['score'] is not None]
            if cs:
                crowd_history.append(float(np.mean(cs)))

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
                  f"| coh {result['debug']['coherence']:.2f} "
                  f"face {result['debug']['max_facing']} stand {result['debug']['max_stand']} "
                  f"move {result['debug']['max_move']} "
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

    if crowd_history:
        ch = np.array(crowd_history, dtype=np.float32)
        tag = " - action disengagement penalty" if args.action_model else ""
        print(f"\n=== ENGAGEMENT SUMMARY (gaze base{tag}) ===")
        if args.action_model:
            print(f"action model     : {args.action_model} (dis-tau {args.dis_tau}, "
                  f"max penalty {args.action_penalty_max})")
        print(f"scored frames    : {len(ch)}")
        print(f"mean crowd eng   : {ch.mean()*100:.1f}%")
        print(f"median crowd eng : {float(np.median(ch))*100:.1f}%")
        print(f"min / max        : {ch.min()*100:.1f}% / {ch.max()*100:.1f}%")
        print(f"frames >= 50%    : {float(np.mean(ch >= 0.5)*100):.1f}%")
    else:
        print("\n=== ENGAGEMENT SUMMARY ===\n(no LIVE frames - never reached a started/focal state)")


if __name__ == '__main__':
    main()
