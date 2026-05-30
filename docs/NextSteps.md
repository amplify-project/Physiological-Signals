# Next Steps: Concert Engagement System

This document captures planned future work for the AudiencePose/Concert Engagement system.

---

## Post Jazzahead Demo Roadmap

This section captures issues observed during live demoing, with emphasis on efficiency, robust behavior under load, and practical concert conditions.

### 1. EmotiBit visibility before face registration
- [ ] EmotiBit plots should be visible even before being registered to a facial ID.
- [ ] Decouple physio visibility from face-enrollment state so operators can validate sensor health immediately.

**Proposed solutions**
- [ ] Standard practice: render all discovered EmotiBit streams as unassigned rows immediately, independent of face ID state.
- [ ] Standard practice: add assignment states (`unassigned`, `assigned`, `stale`) and keep plotting active in all states.
- [ ] State-of-the-art: add confidence-based auto-association between tracked people and devices using temporal cues, with manual override.

### 2. Multi-person FPS collapse and runtime efficiency
- [ ] During demoing, many people were simultaneously detected. Some FPS reduction per additional person is expected, but frame rate dropped to around 3 FPS.
- [ ] Maintain a minimum FPS target of 12 at all times.
- [ ] Evaluate adaptive frame processing to maintain 12 FPS (for example, controlled frame dropping, selective per-frame work, or phased inference).

**Proposed solutions**
- [ ] Standard practice: implement a hard real-time budget loop with degradation tiers to enforce minimum 12 FPS.
- [ ] Standard practice: run detector every `N` frames, reuse tracker state between detections, and run pose every `M` frames per target.
- [ ] Standard practice: dynamically downscale input and cap active targets by priority under load.
- [ ] State-of-the-art: asynchronous multi-rate pipeline (capture, detect, pose, classify, render, publish) with bounded queues and frame dropping under backpressure.
- [ ] State-of-the-art: export detector/inference path to optimized runtime (ONNX/TensorRT where available) for lower latency.

#### 2.b. ID inflation and tracker stability
- [ ] Additional IDs were repeatedly assigned to the same people after detect/lost/redetect cycles.
- [ ] IDs reached the 1000s for about 20 people.
- [ ] Review tracker identity persistence and reuse strategy to reduce duplicate IDs.
- [ ] Evaluate memory impact of large ID churn and buffer retention.
- [ ] Depending on achieved frame-rate solutions, consider a mode that prioritizes visual tracking of the 6 people wearing EmotiBits.

**Proposed solutions**
- [ ] Standard practice: switch to or tune robust MOT settings (track age, minimum hits, reactivation window) to reduce identity churn.
- [ ] Standard practice: separate detector-internal IDs from stable application-level person IDs.
- [ ] Standard practice: enforce lifecycle cleanup and caps for inactive tracks and stale buffers.
- [ ] State-of-the-art: add appearance re-identification embeddings for long occlusion recovery and identity stitching.

#### 2.c. Context duration drift
- [ ] In demo images, FPS was around 3 but context grew to around 20 seconds.
- [ ] Context window should stay at 10 seconds.
- [ ] Investigate and fix context-duration drift so temporal context remains pinned to target duration.

**Proposed solutions**
- [ ] Standard practice: convert context control to time-based buffering instead of fixed frame-floor behavior.
- [ ] Standard practice: keep a 10-second target window and resample buffered features to the model input length when FPS is low.
- [ ] Standard practice: set minimum sequence constraints from inference viability (first estimate threshold), not from static frame counts.
- [ ] State-of-the-art: include time-delta encoding for irregular frame spacing in the temporal model.

#### 2.d. Underutilized hardware at low FPS
- [ ] Demo images show around 3 FPS while CPU and GPU were not fully burdened.
- [ ] Profile pipeline stages to identify serialization bottlenecks and non-hardware-limited stalls.
- [ ] Evaluate parallel and asynchronous execution paths where safe and measurable.

**Proposed solutions**
- [ ] Standard practice: add stage-level wall-time instrumentation (capture, detect, pose, classify, render, publish, logging).
- [ ] Standard practice: remove blocking synchronization points and pre-allocate tensors/buffers to reduce per-frame overhead.
- [ ] Standard practice: parallelize independent CPU-heavy tasks (pose per target/view) with bounded worker pools.
- [ ] State-of-the-art: add timeline tracing for queue wait vs compute time and schedule work by deadlines.

### 3. Pipeline redundancy and 2D-first optimization gate
- [ ] There is pipeline redundancy with double plotting and related duplicate work.
- [ ] Streamline plotting/data paths before adding complexity.
- [ ] Establish an efficiency gate for 2D perspective mode before proceeding to 360 feeds.
- [ ] 360 processing currently scales roughly 4x and can drop FPS to less than 1, which is unacceptable for live use.

**Proposed solutions**
- [ ] Standard practice: enforce a single source of truth for physio + vision state and a single render path per signal.
- [ ] Standard practice: disable noncritical visual components in performance mode with feature flags.
- [ ] Standard practice: define and enforce a 2D performance acceptance gate (minimum FPS, latency, ID stability) before 360 enablement.
- [ ] State-of-the-art: event-driven architecture with cost-aware widgets that can auto-throttle under load.

### 4. Engagement model behavior and fairness across audience styles
- [ ] Retrain the engagement model toward action-based engagement signals.
- [ ] Reduce unfair dependence on distance from camera and direct gaze.
- [ ] Ensure seated but highly engaged audiences are not penalized.
- [ ] Better distinguish arousal from engagement.
- [ ] Explore rolling local-max normalization per audience context (for example, low-motion audiences vs highly dynamic audiences) without bias.
- [ ] Improve handling so applause/cheering away from camera center still contributes appropriately.
- [ ] Reduce direct-camera-gaze bias since audiences primarily look at performers, not the camera.

**Proposed solutions**
- [ ] Standard practice: split targets into separate heads (engagement vs arousal) with distinct evaluation metrics.
- [ ] Standard practice: rebalance training by distance bands, viewing angles, seated/standing contexts, and off-center subjects.
- [ ] Standard practice: add temporal/context features that reward collective audience response beyond direct gaze.
- [ ] Standard practice: apply per-session calibration and post-hoc score calibration for venue-specific behavior.
- [ ] State-of-the-art: multi-task and domain-adaptive training with fairness slicing across audience styles and venue conditions.
- [ ] State-of-the-art: rolling local-baseline normalization for audience-specific dynamic range while preserving cross-session comparability.

### 5. Facial identification operating range
- [ ] Facial identification currently works reliably only at less than about 10 feet.
- [ ] Improve recognition performance for stage-mounted camera distances in live venues.
- [ ] Target practical operation for performance-area conditions (approximately 8 m^2).

**Proposed solutions**
- [ ] Standard practice: make face ID optional and non-blocking so core engagement remains stable at long range.
- [ ] Standard practice: use higher-resolution crops and match only on stable frontal frames.
- [ ] Standard practice: maintain person identity between sparse face matches using tracker continuity.
- [ ] State-of-the-art: fuse face and body re-identification embeddings for longer-range identity persistence.

### 6. Data saving overhead not yet demo-validated
- [ ] The demo did not include data saving overhead.
- [ ] Run end-to-end performance validation with save options enabled.
- [ ] Quantify FPS and latency impact for `--save`, `--save-engagement`, and `--save-keypoints` in realistic multi-person sessions.

**Proposed solutions**
- [ ] Standard practice: benchmark no-save vs each save mode under the same scripted workload and report FPS/latency deltas.
- [ ] Standard practice: move writes to asynchronous background workers with bounded queues and chunked flush.
- [ ] Standard practice: apply backpressure policy that drops optional logs before blocking inference.
- [ ] State-of-the-art: adaptive telemetry policy that lowers logging verbosity automatically when frame budget is threatened.

### Priority Summary
- [ ] Primary focus: improved efficiency and stable minimum frame rate under demo crowd load.
- [ ] Secondary focus: behavioral correctness, model refinement, and range improvements for production-like concerts.

---

## 360° Camera Support

### Status: Implemented — Awaiting Hardware Validation

The 360° virtual camera extraction pipeline has been implemented in `live_multiperson_binary_v2.py`. It extracts 4 perspective views from equirectangular frames, processes each through the existing YOLO → MediaPipe → Transformer pipeline, and displays a 2×2 mosaic in a single GUI window.

**Still requires validation:**
- [ ] Test with actual 360° camera hardware (Insta360, GoPro MAX, Ricoh Theta, etc.)
- [ ] Verify auto-detection of equirectangular format from live 360° camera feed
- [ ] Validate real-time performance at 4× compute cost with real equirectangular frames
- [ ] Confirm engagement score aggregation across views produces meaningful results
- [ ] Test `--format 360` override flag with 360° video files

### The Challenge
The current pipeline relies on:
- **YOLO26**: Trained on COCO dataset (perspective images)
- **MediaPipe Holistic**: Trained on perspective imagery

Neither model handles **equirectangular projection** (the native format of 360° cameras), where:
- Straight lines become curved
- People near the poles are heavily distorted
- Scale varies dramatically across the image

Running these models directly on equirectangular frames will produce poor results.

### Recommended Solution: Virtual Camera Extraction

Rather than retraining models on equirectangular data (expensive, time-consuming), we recommend **extracting perspective views** from the 360° feed:

```
360° Equirectangular Frame
         │
         ▼
┌─────────────────────────────────────┐
│   Virtual Camera Extraction          │
│   (py360convert / OpenCV)            │
└─────────────────────────────────────┘
         │
         ▼
┌─────┬─────┬─────┬─────┐
│Front│Right│ Back│Left │  (2-4 perspective views)
└─────┴─────┴─────┴─────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Existing Pipeline (per view)       │
│   YOLO26 → MediaPipe → Transformer   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Score Aggregation                  │
│   (weighted average / max / voting)  │
└─────────────────────────────────────┘
         │
         ▼
    Crowd Engagement Score
```

### Implementation Details

#### Libraries
- **py360convert**: `pip install py360convert` - purpose-built for 360° projections
- **OpenCV**: Can also handle equirectangular-to-perspective conversion

#### Virtual Camera Parameters
Each extracted view needs:
- **FOV**: 90° horizontal (typical perspective camera)
- **Yaw**: Rotation around vertical axis (0°, 90°, 180°, 270° for 4 views)
- **Pitch**: Typically 0° (level with horizon)
- **Output resolution**: 640×480 or 1280×720

#### Example Code Skeleton
```python
import py360convert
import numpy as np

def extract_views(equirect_frame, num_views=4):
    """Extract perspective views from 360° equirectangular frame."""
    views = []
    for i in range(num_views):
        yaw = i * (360 / num_views)  # 0°, 90°, 180°, 270°
        perspective = py360convert.e2p(
            equirect_frame,
            fov_deg=(90, 90),  # horizontal, vertical FOV
            u_deg=yaw,
            v_deg=0,  # level with horizon
            out_hw=(480, 640)
        )
        views.append(perspective)
    return views
```

#### Score Aggregation Options
1. **Weighted Average**: Weight by number of people detected in each view
2. **Maximum**: Take highest engagement (optimistic)
3. **Median**: Robust to outlier views
4. **Person-Weighted**: Sum(engagement × confidence) / Sum(confidence)

### Effort Estimate
- **Virtual camera extraction module**: 2-3 days
- **Pipeline integration**: 2-3 days
- **Score aggregation logic**: 1 day
- **Testing and tuning**: 2-3 days
- **Total**: ~1-2 weeks

### Hardware Considerations
- Processing 4 views = 4× computational cost
- May need to reduce frame rate or view count on CPU-only systems
- GPU strongly recommended for real-time 360° processing

### Alternative Approaches (Not Recommended)
1. **Retrain models on equirectangular data**: 
   - Requires large annotated 360° dataset (doesn't exist)
   - Months of work, uncertain results
   
2. **Fisheye-aware models**:
   - Some research exists but not production-ready
   - Would require custom model development

---

## Data Logging (`--save`) Validation

### Status: Implemented — Partial Validation

The `--save` flag and `EngagementLogger` are implemented and tested on Windows with a single-person 2D webcam feed. Each session saves to a timestamped subfolder under `data/sessions/`:

- **`engagement_data.jsonl`** — per-person, per-frame records (track_id, engagement_score, bbox, buffer_fill, crowd_average, people_count, fps)
- **`keypoints.npz`** — raw MediaPipe Holistic keypoints as compressed NumPy arrays:
  - `frames` (N,) int32 — frame number per entry
  - `track_ids` (N,) int32 — YOLO track ID per entry
  - `keypoints` (N, 543, 3) float16 — 33 pose + 468 face + 21 left hand + 21 right hand
  - `bboxes` (N, 4) int32 — padded crop bounding box per entry (x1, y1, x2, y2) in pixel coords
  - `frame_size` (2,) int32 — source frame dimensions (width, height)

A validation/visualisation script is available at `scripts/analysis/view_keypoints.py`.
Animated playback is available at `scripts/analysis/playback_keypoints.py` with `--all-tracks` and `--smooth` options.

The following still require validation:

- [ ] **Multi-person (2D):** Validate with multiple people simultaneously tracked — verify per-person track ID handling and crowd-level aggregation in JSONL
- [ ] **Multi-person (360°):** Validate with multiple people across 4 virtual views — confirm cross-view person counts and score aggregation are logged correctly
- [ ] **macOS / Linux:** Verify session subfolder creation, JSONL writes, graceful shutdown on non-Windows platforms
- [ ] **Long-duration stress test:** Run for 30+ minutes with a large audience to confirm buffered I/O (30-frame flush interval) has no impact on inference FPS
- [ ] **Crash recovery:** Confirm that JSONL data written before an unexpected crash is valid and recoverable

---

## Cross-Platform Validation

The system is designed to run on Windows, macOS, and Linux with automatic device detection (CUDA / MPS / CPU). Development and testing has been primarily on Windows with CUDA. The following still require validation on macOS and Linux:

- [x] **macOS (Apple Silicon / MPS):** Inference pipeline (YOLO → MediaPipe → Transformer) runs correctly on MPS device
- [x] **macOS (Apple Silicon / MPS):** GUI overlay renders correctly via `cv2.imshow`
- [ ] **macOS (Intel):** CPU fallback works correctly
- [ ] **Linux (CUDA):** Full pipeline including smart camera detection and GPU inference
- [ ] **Linux (CPU):** CPU fallback performance is acceptable for real-time use
- [ ] **Cross-platform:** Data logging (`--save`) writes JSONL and session summary correctly on all platforms
- [ ] **Cross-platform:** Redis pub/sub connects and publishes on all platforms
- [ ] **Cross-platform:** Signal handlers (SIGINT/SIGTERM) trigger graceful shutdown and summary write
- [ ] **Cross-platform:** 360° mode mosaic rendering on macOS and Linux

---

## Other Future Work

### Progressive Confidence Scoring — ✅ Implemented
- Estimates produced from as few as 30 frames (~1s) with zero-padded partial buffers
- Per-person `conf:X%` label and cyan fill bar shown during buffer ramp, fading at 100%
- Crowd average is confidence-weighted (partial-buffer people weighted by buffer fill ratio)
- Console displays calibration progress with first-estimate and full-confidence timing
- Branch: `feature/progressive-confidence-scoring`

### Camera Index Caching — ✅ Implemented
- Caches the last successfully used camera index to `.last_camera` for instant startup
- First run: full sequential scan of indices 0-9 (unchanged behaviour)
- Subsequent runs: probes only the cached index — skips the slow scan entirely
- Automatic fallback to full scan if cached camera is unavailable (unplugged, index changed)
- Re-caches the newly selected camera after fallback scan
- Cross-platform compatible (no threading, no platform-specific APIs)

### Console Cleanup — ✅ Implemented
- MediaPipe/TFLite C++ startup warnings suppressed via OS-level stderr fd redirect
- Stderr restored after first inference frame to preserve real error reporting
- Redis publish events logged to console (`📡 Redis pub → engagement_score: X.XXXX`)

### Bbox Passthrough & Extended NPZ Format — ✅ Implemented
- `extract_features()` now returns `(keypoints, padded_bbox)` tuple — padded crop box in pixel coords
- NPZ format extended with `bboxes` (N,4 int32) and `frame_size` (2, int32)
- Enables post-hoc coordinate transform from crop-relative to frame-space positions

### Multi-Person Keypoint Playback — ✅ Implemented
- `--all-tracks` mode shows all tracked people simultaneously with 6 distinct colour-coded skeletons
- Crop-relative MediaPipe keypoints transformed to frame-space coordinates using bounding boxes
- JSONL bbox fallback for old NPZ files that lack embedded bboxes (reads companion `engagement_data.jsonl`, applies 10% padding, estimates frame resolution)
- Face tiles ordered by horizontal bbox position so left-to-right layout matches scene
- Simplified single-panel layout (removed separate face zoom and hands panels)

### Playback Temporal Smoothing — ✅ Implemented
- `--smooth` flag enables median-based outlier rejection + 5-frame temporal moving average
- Frame-space centroids used for outlier detection (avoids false positives from crop-window drift)
- Both keypoints and bboxes are smoothed so the crop→frame transform stays stable
- Replaces outlier frames with neighbourhood median rather than deleting them (all tracks stay visible)

### Enriched Redis Payload
- Upgrade Redis pub/sub from bare float to JSON with per-person data
- Include track IDs, individual scores, people count, and timestamps
- Coordinate with AR glasses team before changing — they currently expect a bare float on `engagement_score`

### Facial Expression / Emotion Estimation
- The 468 face mesh landmarks are already saved as raw keypoints in the NPZ archive (indices 33–500)
- Post-hoc analysis can derive facial signals (smile, mouth openness, eye openness, head pose, dynamism) from the saved keypoints without any live inference changes
- Future: train a lightweight emotion classifier on geometric features extracted from the archived keypoints (e.g. valence/arousal or discrete emotions)
- Note: pseudo-AU approaches (arbitrary normalisation thresholds) were evaluated and rejected — raw keypoint archival preserves maximum flexibility for future analysis

### Reduced Temporal Window (5-Second Model)
- Current model uses a 10-second sliding window (`sequence_length=300` @ 30fps), matching the Kinetics-700 and DAiSEE source clip durations
- This creates an inherent ~10-second lag between real behaviour change and score response
- The Temporal Transformer architecture and positional encoding already support shorter sequences (proven by progressive scoring at 30+ frames)
- Retraining with `sequence_length=150` (5s @ 30fps) would halve inference lag with minimal code changes
- Kinetics 10-second clips can be split into two 5-second samples (doubles training data) or randomly cropped for augmentation
- Trade-off: faster response vs. less temporal context — may reduce accuracy for sustained engagement patterns
- Recommended approach: train a 5s variant and A/B compare against the 10s model on held-out data before committing

### Model Accuracy Revisit
- Current binary model (82.75%) tends toward high engagement only when subject is stationary and staring at screen
- Revisit training data distribution and class balance after all current features are stable
- Consider fine-tuning on more diverse engagement scenarios (standing, moving, group settings)
- Test with multi-person live scenarios to evaluate real-world accuracy

### GUI Controls & Menu Bar
- Replace current keyboard-only controls with a proper GUI menu bar (OpenCV `createTrackbar` or migrate to Dear ImGui / Qt overlay)
- **Data recording toggle** — opt-in record button in the UI to start/stop `--save` mid-session without restarting the application
- **Facial ID enrolment** — menu for registering up to 5 participants with reference images (see Facial Identification section below)
- **Display toggles** — show/hide bounding boxes, engagement bars, skeleton overlays, FPS counter
- Future: settings panel for confidence threshold, Redis channel name, output directory

### Facial Identification & Physiological Monitoring
Identify up to 5 pre-enrolled audience members in the live video feed, enabling technicians to visually locate specific participants during a performance.

**Enrolment:** ✅ Implemented
- ~~GUI-based enrolment of up to 5 participants before/during a session~~
- Keyboard-based enrolment (R key) during live session — picks largest unidentified person in frame
- ✅ Encode face crops into 512-dim embeddings via facenet-pytorch (MTCNN + InceptionResnetV1)
- ✅ Persistent enrollment storage (`.face_enrollments/enrollments.pt`) — survives app restarts
- ✅ Auto-naming (`Participant 1`, `Participant 2`, etc.)
- Future: capture front, left-angled, and right-angled reference images per person (3-angle registration)
- Future: pre-saved reference image files for enrolment without live camera

**Visual Identification:** ✅ Implemented
- ✅ Identified participants receive magenta bounding boxes to distinguish them from the standard red→green engagement gradient boxes
- ✅ Focus mode (F key) cycles through: all → none → individual participants — allows operator to highlight specific individuals without visual clutter
- ✅ Participant label + engagement score displayed alongside bounding box when highlighted
- ✅ Throttled MTCNN matching (every 10 frames) with cached results — maintains ~14-19 FPS
- ✅ MTCNN crash guard for small/blurry crops (try-except RuntimeError)
- ✅ EmotiBit serial always shown as bbox label for enrolled participants (with or without focus)
- Future: distinct per-participant colours (currently all magenta)

**360° Facial Registration — TODO**

Face identification is currently **disabled in 360° mode** (`not is_360` guard throughout). However this is not technically necessary — the 4 perspective views are already extracted via `py360convert.e2p()` and each view is a standard perspective image that MTCNN handles natively.

**Proposed implementation:**
- Run MTCNN face identification on each person's view crop (stored in `person['view']` + bounding box) rather than the raw equirectangular frame
- R key in 360° mode: show picklist of detected EmotiBit serials (same as 2D), then associate with the largest unidentified person across all 4 views
- F key focus cycling: works identically — `focus_target` matches against `person['identified_as']` and the mosaic tile for that view gets the magenta box
- Face enrollment: use the cropped perspective view region (same `register_from_crop()` call, no changes to `FaceIdentifier`)
- No retraining required — MTCNN was trained on perspective images, perspective views are already extracted

**Implementation effort:** ~1–2 days. The main changes are:
1. Remove `not is_360` from the face ID loop — iterate `people_data` and look up the view crop via `views[person['view']]` + `person['bbox']`
2. Allow R and F keys in 360° mode
3. Draw magenta box inside the mosaic tile for the correct view (not on the equirectangular)

- [ ] Implement 360° face registration using per-view perspective crops
- [ ] Test with actual 360° hardware

**Integration with EmotiBit Physiological Sensors:** ✅ Implemented (April 2026; SD/RF v2 update May 2026)
- ✅ `physio/multiemotibit_UDP_SD_RFv2.py` — direct UDP EmotiBit discovery, no LSL or Oscilloscope dependency
- ✅ EDA/HR-derived standard-deviation metrics streamed to Redis at 1 Hz per device on `device:{serial}:physio_metrics`
- ✅ Random Forest continuous valence/arousal predictions from EDA + HR features using the v2 RF models
- ✅ Sidebar panel (280px, hstacked): per-participant `EDA SD` and `HR SD` readouts from the SD metrics stream
- ✅ Up to 6 EmotiBit devices simultaneously — dynamic compact row heights
- ✅ R key opens picklist of detected-but-unassigned serials; serial becomes persistent participant label
- ✅ Mouse click on sidebar row sets focus_target; magenta bounding box appears on focused participant only
- ✅ scikit-learn pinned to 1.1.3 for binary compatibility with pkl models (saved under 1.1.2)
- Future: 3-angle face registration for improved re-identification accuracy across sessions

**Technical Approach:** ✅ Implemented
- ✅ FaceNet (facenet-pytorch: MTCNN + InceptionResnetV1/vggface2) for encoding and matching
- ✅ Face crops from YOLO person detection pipeline
- ✅ Matching throttled to every 10 frames, cached results carried forward
- ✅ Identified faces linked to YOLO track_id for persistent tracking across frames
- ✅ Identification events logged in JSONL data when `--save` is enabled (`identified_as`, `face_similarity` fields)
- ✅ Cosine similarity threshold: 0.65
- ✅ Disable with `--no-face-id` flag

### Multi-Camera Fusion
- Combine scores from multiple camera angles
- De-duplicate individuals seen by multiple cameras
- Create venue-wide engagement heatmap

### Historical Analytics
- Store engagement timelines in database
- Correlate with performance segments
- Generate post-show analytics reports

### Performer Feedback
- Real-time engagement display for performers
- Haptic feedback devices
- AR overlay for stage monitors

---

## Iteration 3 — Completed Feature Blurbs (For GitHub)

> **Note:** These blurbs are ready to copy-paste into the GitHub Iteration 3 project board once it is created.

### File I/O: Engagement Data Logging (`--save`)

**Description:** Added `--save` flag to the real-time inference script that enables per-frame engagement data logging for post-experience academic analysis. Each session creates a timestamped subfolder under `data/sessions/` containing an append-only JSONL file with per-person, per-frame records.

**JSONL fields per record:** `timestamp`, `frame`, `track_id`, `engagement_score`, `bbox`, `buffer_fill`, `crowd_average`, `people_count`, `fps`. 360° mode additionally includes `view` and `yaw`.

**Key implementation details:**
- Append-only JSONL format — crash-safe, no data loss on unexpected termination
- Buffered I/O with 30-frame flush interval — minimal impact on inference FPS
- Per-session subfolders with UTC timestamps + unique session ID in folder name
- `--save-dir` flag for custom output directory
- Graceful shutdown via SIGINT/Ctrl+C flushes and closes all files

**Files modified:** `scripts/inference/live_multiperson_binary_v2.py` (EngagementLogger class), `.gitignore`, `README.md`

---

### File I/O: Raw Keypoint Archival (NPZ)

**Description:** When `--save` is enabled, the system archives all raw MediaPipe Holistic keypoints to a compressed NumPy `.npz` file alongside the JSONL engagement data. This preserves the rawest possible skeletal data for post-hoc analysis without imposing any interpretation or thresholding at capture time.

**NPZ arrays:**
- `frames` (N,) int32 — frame number per entry
- `track_ids` (N,) int32 — YOLO track ID per entry
- `keypoints` (N, 543, 3) float16 — 33 pose + 468 face + 21 left hand + 21 right hand
- `bboxes` (N, 4) int32 — padded crop bbox (x1, y1, x2, y2) in pixel coords
- `frame_size` (2,) int32 — source frame dimensions (width, height)

**Key implementation details:**
- float16 precision halves storage (~3.7 GB vs 7.4 GB for 50 people × 30 minutes)
- Frame and track_id indices enable efficient slicing by person or time
- Compressed NPZ format provides ~50% further reduction on disk
- Validation script at `scripts/analysis/view_keypoints.py` renders pose skeleton, face mesh, and hand landmarks from saved data
- Enables future post-hoc derivation of facial signals, synchrony metrics, and skeleton replay without any live inference changes

**Files modified:** `scripts/inference/live_multiperson_binary_v2.py` (EngagementLogger._save_keypoints, extract_features, process_frame), `scripts/analysis/view_keypoints.py` (new)

### Camera Selection: Interactive Multi-Camera Picker

**Description:** When the system detects more than one real camera, it now shows a tiled live-preview window instead of silently auto-selecting the first one. Users can see all camera feeds simultaneously and pick the desired source before inference starts.

**Key implementation details:**
- Two-phase detection: Phase 1 checks cached (last-used) camera with a full 3-frame test; Phase 2 quick-scans remaining indices for additional cameras only (keeping the fast-startup path when only one camera is present)
- 3-frame real-content validation for every candidate: brightness check, uniformity check, and frame-diff check. All three must pass — this correctly rejects static virtual cameras such as SMPTE colour-bar sources that pass brightness/uniformity alone
- Interactive picker opens a `cv2` tiled grid window (up to 3 columns, scaled 320×240 tiles) showing live feeds from all detected cameras
- Label bar per tile shows: `[N]  camX  WxH  360°  last used  [default]`; default tile highlighted in green
- Key bindings: `1`/`2`/`3`... — select by number; `Enter`/`Space` — accept default; `Q`/`Esc` — quit application
- Default priority: last-used camera first, then 360°, then first 2D
- Window closes immediately on selection (`destroyAllWindows()` + 30× `waitKey(1)` flush for Windows message-queue compatibility)
- Selected camera index written to `.last_camera` cache for next startup
- `--camera <idx>` flag bypasses auto-detection entirely when a specific index is known

**Files modified:** `scripts/inference/live_multiperson_binary_v2.py` (`select_camera_interactively`, camera scan block, `import math`), `README.md`
**Branch:** `feature/camera-selection` → merged to `master` (March 2026)

---

*Last updated: March 2026*
