# Concert Engagement + Physiological Signals

Real-time audience engagement estimation for live performances using pose-based machine learning and EmotiBit physiological signal processing.

---

## Current Working Repository

This repository is now the shared source of truth for the concert engagement system.

Work should proceed from Sowmya's `amplify-project/Physiological-Signals` remote rather than the earlier standalone Audience Pose / local `concert_engagement` repository. The previously independent parts of the pipeline have been ported into this project so development can continue from one combined codebase:

- Computer-vision engagement inference, multi-person tracking, face registration, 360-video support, logging, analysis, and training utilities are now in the project-level `scripts/`, `src/`, `models/`, and `docs/` folders.
- Sowmya's newer EmotiBit physiological pipeline is now the project-level Physio implementation in `physio/multiemotibit_UDP_SD_RFv2.py`.
- The multiperson GUI consumes the new physiological standard-deviation stream (`eda_sd` and `hr_sd`) from `device:{serial}:physio_metrics`.
- `Handover_JuneSession/` remains as the up-to-date handover package and reference copy for the demo-ready workflow.

In short: clone and work from this repository for future changes to either the vision or physiological parts of the pipeline.

---

## 🚀 Quick Start

### Automated Setup (Recommended)

**Windows (PowerShell):**
```powershell
git clone https://github.com/amplify-project/Physiological-Signals.git
cd Physiological-Signals
.\setup.ps1
```

**macOS / Linux:**
```bash
git clone https://github.com/amplify-project/Physiological-Signals.git
cd Physiological-Signals
chmod +x setup.sh
./setup.sh
```

The setup scripts will:
- ✅ Detect your GPU (CUDA/MPS) and install the correct PyTorch version
- ✅ Create and activate a virtual environment
- ✅ Install all dependencies

### Manual Setup

```bash
# 1. Clone repository
git clone https://github.com/amplify-project/Physiological-Signals.git
cd Physiological-Signals

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install PyTorch (IMPORTANT: match your GPU!)
# CUDA (Windows/Linux with NVIDIA GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# MPS (macOS Apple Silicon):
pip install torch torchvision torchaudio
# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install -r requirements.txt

```

### Running the System

```bash
# Activate environment (if not already active)
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# Start Redis
.\redis\redis-server.exe     # Windows
# redis-server               # macOS/Linux (install via brew/apt)

# Run real-time engagement system (Binary V2 - 82.75% accuracy)
python scripts/inference/live_multiperson_binary_v2.py
# If multiple cameras are detected, a tiled live-preview window opens:
#   press 1/2/3... to pick, Enter to accept the default (last used / 360° preferred), Q to quit

# Force a specific camera index — skips auto-detection entirely
python scripts/inference/live_multiperson_binary_v2.py --camera 2
python scripts/inference/live_multiperson_binary_v2.py --video path/to/video.mp4

# Force 360° mode for equirectangular video
python scripts/inference/live_multiperson_binary_v2.py --video 360video.mp4 --format 360

# Enable all data logging (engagement JSONL + keypoints NPZ)
python scripts/inference/live_multiperson_binary_v2.py --save

# Save engagement data only (lightweight, ~370 MB / 30 min)
python scripts/inference/live_multiperson_binary_v2.py --save-engagement

# Save raw keypoints only (543×3 float16 NPZ)
python scripts/inference/live_multiperson_binary_v2.py --save-keypoints

# Save to a custom directory
python scripts/inference/live_multiperson_binary_v2.py --save --save-dir /path/to/output

# Disable facial identification for lightweight operation
python scripts/inference/live_multiperson_binary_v2.py --no-face-id

# Validate saved keypoints (visual skeleton plot)
python scripts/analysis/view_keypoints.py data/sessions/<session>/keypoints.npz
python scripts/analysis/view_keypoints.py data/sessions/<session>/keypoints.npz --frame 50 --track 1

# Animated keypoint playback (slider, play/pause, keyboard controls)
python scripts/analysis/playback_keypoints.py data/sessions/<session>/keypoints.npz

# Multi-person playback (all tracks shown simultaneously, colour-coded)
python scripts/analysis/playback_keypoints.py <path> --all-tracks

# Smoothed playback (median-based outlier rejection + temporal averaging)
python scripts/analysis/playback_keypoints.py <path> --all-tracks --smooth

# (Optional) Subscribe to engagement scores
python scripts/inference/console_subscriber.py
```

**Output**: Crowd engagement score (0.0-1.0) published to Redis channel `engagement_score` at 1 Hz. Visual overlay shows red→green engagement bar at top of window. Progressive confidence scoring produces initial estimates within ~1 second of launch, with per-person confidence indicators during buffer ramp-up. Data logging is opt-in via `--save` (both), `--save-engagement` (JSONL only), or `--save-keypoints` (NPZ only). Sessions are saved to `data/sessions/`.

---

## 📋 Requirements

| Requirement | Windows | Linux | macOS |
|-------------|---------|-------|-------|
| **Python** | 3.8 - 3.11 | 3.8 - 3.11 | 3.8 - 3.11 |
| **GPU** | NVIDIA CUDA 11.8+ | NVIDIA CUDA 11.8+ | Apple Silicon (MPS) |
| **Fallback** | CPU | CPU | CPU (Intel Macs) |
| **Redis** | 5.0+ | 5.0+ | 5.0+ |
| **RAM** | 8GB min, 16GB rec. | 8GB min, 16GB rec. | 8GB min, 16GB rec. |
| **Webcam** | 720p+ (auto-detected) | 720p+ (auto-detected) | 720p+ (auto-detected) |
| **360° Camera** | Optional | Optional | Optional |
| **Disk (with data logging)** | ~3 GB per 30 min / 50 people (`--save`) | ~3 GB per 30 min / 50 people (`--save`) | ~3 GB per 30 min / 50 people (`--save`) |

**Platform Notes**:
- **Windows**: Use `setup.ps1` for automated setup. Redis available via WSL or [Memurai](https://www.memurai.com/)
- **Linux**: Use `setup.sh` for automated setup. Install Redis via package manager (`apt install redis-server`)
- **macOS**: Use `setup.sh` for automated setup. Install Redis via Homebrew (`brew install redis`)
- **GPU Detection**: Automatic — CUDA on Windows/Linux, MPS on Apple Silicon, CPU fallback everywhere

---

## 🎯 Model Performance

**Binary Engagement Classifier** — 82.75% accuracy

- **Engaged**: Dancing, clapping, cheering, singing, playing instruments, attentive gazing
- **Disengaged**: Phone use, sleeping, negative body language, checking time

**Output**: Engagement score (0.0–1.0) published to Redis at 1 Hz

---

## ✨ Features

### 📷 Smart Camera Detection
- **Phase 1**: Checks cached (last-used) camera index with a full 3-frame real-content test — instant startup when the same camera is plugged in
- **Phase 2**: Quick-scans remaining indices (0–9) with a 3-frame test to detect any additional cameras
- When only one real camera is found, selects it automatically
- **When multiple real cameras exist**, opens a tiled live-preview window showing all feeds simultaneously — press `1`/`2`/`3`... to choose (1-indexed by list position, not OS device index), `Enter` to accept the default (last-used first, then 360°), `Q` to quit
- **`--select-camera` / `-s`** forces the picker to appear on every run, even when only one camera is detected, and bypasses the last-used cache so a freshly plugged-in USB webcam is always discovered. Useful for non-technical end users handing the laptop between people. The bundled `Handover_JuneSession/3_START_ENGAGEMENT.bat` passes this flag by default.
- Default camera highlighted in green in the picker; camera used most recently labelled "last used"
- Filters out virtual cameras (OBS, NDI, SMPTE colour bars) using brightness, uniformity, and frame-diff checks — static or blank feeds are rejected
- Caches the selected camera index for instant startup next time
- Prioritises 360° cameras as default when no last-used preference is set
- Cross-platform: DirectShow backend on Windows, default backend on macOS/Linux
- Clean console output — MediaPipe/TFLite C++ warnings suppressed via OS-level stderr redirect

### 🖥️ Resizable GUI Window
- Main display window is created as `WINDOW_NORMAL` and resized to fit `frame width + physio sidebar`, capped at 1600 px so it never opens larger than a typical laptop screen.
- Without this OpenCV defaults to `WINDOW_AUTOSIZE`, which on smaller displays pushes the title bar, RHS physio sidebar, and bottom FPS overlay off-screen and breaks the `q`-to-quit shortcut.
- `opencv-contrib-python` is pinned to `4.10.0.84` in `requirements.txt` to avoid HighGUI regressions seen on some Windows machines with 4.11.x.

### 📝 Logging
Every pipeline entry-point writes a rotating log file so post-mortem debugging is possible after a session, without impacting frame rates.

- **Where:**
  ```
  data/logs/engagement/engagement_<timestamp>_pid<n>.log   # multiperson camera app
  data/logs/emotibit/emotibit_<timestamp>_pid<n>.log       # EmotiBit UDP + ML pipeline
  data/logs/subscriber/subscriber_<timestamp>_pid<n>.log   # Redis subscriber CLI
  ```
  10 MB per file × 5 rotations = ~50 MB ceiling per script. New file per run.

- **What is logged:**
  - **Startup context** — platform, Python version, PID, full CLI args, camera/Redis/device choices, model paths.
  - **Errors and warnings** — Redis disconnects, camera open failures, missing models, disk-full events, parse failures on Redis messages, MediaPipe / YOLO init issues.
  - **Uncaught exceptions** — full stack trace via `sys.excepthook`, so a crash on Sowmya's laptop can be diagnosed from the log alone.
  - **Lifecycle events** — session start/end, model load, FPS calibration result, save-dir path, EmotiBit device discovery, face enrolment events.
  - All messages carry a millisecond timestamp and the originating thread name.

- **What is NOT logged** (by design):
  - Per-frame engagement scores and keypoints — those still go to the existing JSONL/NPZ files under `data/sessions/`. Logging them would be high-frequency noise; the session data IS the per-frame record.
  - Console `print(...)` output is unchanged; the console log handler only echoes WARNING+ so the user-facing on-screen feedback stays the same.

- **How it stays fast (async):** producer threads (camera loop, EmotiBit UDP receiver, Redis listener) only push records onto an in-memory `queue.Queue` via a `QueueHandler` (microsecond cost, no disk I/O). A single background `QueueListener` thread drains the queue and handles all formatting + file writes. The 30 FPS render loop is not blocked even at high log volume.

- **For developers:** any module can opt in with one line and messages flow into the same log file:
  ```python
  import logging
  log = logging.getLogger(__name__)
  log.info("loaded model in %.2fs", elapsed)
  ```
  The shared setup lives in [src/applog.py](src/applog.py).

### 🌐 360° Camera Support
- Auto-detects equirectangular video format (2:1 aspect ratio)
- Extracts 4 perspective views at 90° FOV
- Aggregates engagement scores across all views
- Works with Insta360, GoPro MAX, Ricoh Theta, etc.

### 🎯 Real-time Analysis
- YOLO26 person detection (NMS-free, 43% faster on CPU)
- MediaPipe Holistic pose estimation (543 keypoints)
- Temporal Transformer engagement classification
- Adaptive buffer sizing based on measured FPS
- Progressive confidence scoring — first estimates from ~1s, full confidence at ~10s
- Per-person confidence indicator and fill bar during buffer ramp-up

---

## 📋 TODO — Insta360 ONE RS 360° Validation

### Status: In Progress (2026-04-17)

**Goal**: Validate the 360° equirectangular pipeline with real Insta360 ONE RS footage.

- [x] **Get Insta360 footage onto this PC**
  - USB storage mode works (device appears as "Ambarell A9 DSC EVK Platf" on D:\)
  - Camera USB VID `4255` only visible in control/app mode; storage mode uses Ambarella chipset descriptor
  - Raw `.insv` files copied to `data/insta360_raw/`
  - USB detection helper script: `check_insta360.ps1`
- [x] **Export as equirectangular MP4** (5888×2880, 2:1 aspect ratio) via Insta360 Studio
  - Test file: `VID_20260212_050934_00_001.mp4` (5.9s, 30 FPS, 178 frames)
- [x] **Run 360° pipeline validation**:
  ```
  python scripts/inference/live_multiperson_binary_v2.py --video VID_20260212_050934_00_001.mp4 --format 360
  ```
- [ ] **Fix issues found during validation** (see below)

### Issues Found (2026-04-17)
1. **Only 2 perspective views extracted** (front/back) — expected 4 views at 90° intervals
2. **Frame rate ~3 FPS** — unacceptable; likely processing all views per frame at 5888×2944 is too heavy
3. **Engagement score stuck at 0%** — no engagement detected despite people visible in the footage

### Next Steps
- [ ] Investigate why only 2 views are extracted instead of 4
- [ ] Profile and optimise 360° view extraction (downscale equirectangular before perspective projection?)
- [ ] Debug 0% engagement — check if people are being detected in the perspective crops, check keypoint extraction
- [ ] Re-test with longer 360° footage (current clip is only 5.9s — may not fill the temporal buffer)

### Notes
- Insta360 Studio installed on this PC (v5.9.4)
- Camera USB VID: `4255`, PID: `0001`
- Camera raw files are `.insv` (dual fisheye) — must be stitched/exported to equirectangular MP4 first
- The ONE RS has **no built-in webcam mode** — USB is storage only
- LCD sleep on the camera is normal — camera stays on
- In USB storage mode, camera appears as "Ambarell A9 DSC EVK Platf USB Device" (Ambarella SoC), not with VID_4255

### 🎬 Keypoint Playback & Validation
- Animated frame-by-frame playback of saved keypoints with interactive controls (play/pause, slider, speed)
- Multi-person mode (`--all-tracks`): all tracked people shown simultaneously with colour-coded skeletons
- Frame-space coordinate transform: crop-relative MediaPipe keypoints mapped to real scene positions via bounding boxes
- JSONL bbox fallback: old NPZ files without embedded bboxes can load them from companion `engagement_data.jsonl`
- Temporal smoothing (`--smooth`): median-based outlier rejection + 5-frame moving average over keypoints and bboxes

### 🧑 Facial Identification
- Real-time face identification using facenet-pytorch (MTCNN + InceptionResnetV1)
- Register participants via keyboard (R key) — captures face embedding from YOLO crop
- 512-dimensional face embeddings with cosine similarity matching (threshold 0.65)
- Throttled matching (every 10 frames) with cached results for minimal FPS impact
- Focus mode cycling (F key): highlight all / none / individual participants with magenta bounding box
- Persistent enrollment storage (`.face_enrollments/enrollments.pt`) — survives app restarts
- Identification logged in JSONL data (`identified_as`, `face_similarity` fields) when `--save` is enabled
- Disable with `--no-face-id` flag for lightweight operation

### EmotiBit Physiological Integration

The project-level Physio pipeline uses Sowmya's SD/RF v2 EmotiBit implementation: `physio/multiemotibit_UDP_SD_RFv2.py`. Older project-level Physio entry points and non-v2 RF model files have been removed so the project has one active physiological pipeline.

**Running the full stack:**
```powershell
# 1. Start Redis (if not already running)
Start-Process -FilePath "redis\redis-server.exe" -ArgumentList "redis\redis.windows.conf" -WindowStyle Hidden

# 2. Start EmotiBit publisher (foreground - keep visible to monitor SD metrics/valence/arousal)
.venv\Scripts\python.exe physio/multiemotibit_UDP_SD_RFv2.py

# 3. Start main GUI (separate terminal)
.venv\Scripts\python.exe scripts/inference/live_multiperson_binary_v2.py
```

**Key features:**
- `physio/multiemotibit_UDP_SD_RFv2.py` - UDP-based multi-device EmotiBit discovery and SD/RF streaming (no LSL/Oscilloscope required)
- Discovers EmotiBit devices automatically via UDP broadcast on port 3131
- Filters EDA (1 Hz lowpass) and HR (0.5 Hz lowpass) in real time
- Random Forest predictions for continuous valence and arousal from EDA + HR features (models: `physio/rf_valence_full_v2.pkl`, `physio/rf_arousal_full_v2.pkl`)
- Computes standard-deviation metrics over live EmotiBit windows: `edl_sd`, `temperature_roc_sd`, `scr_frequency_sd`, `hr_sd`, and `ibi_sd`
- Publishes SD metrics to Redis on `device:{serial}:physio_metrics`, including explicit `eda_sd` and `hr_sd` fields for the multiperson GUI
- Publishes affect outputs to Redis on `device:{serial}:valence_cont` and `device:{serial}:arousal_cont`
- Logs all raw signals, SD metrics, predictions, and processing time to CSV in `emotibit_recordings/`

**GUI sidebar (280px panel hstacked on the right):**
- One row per enrolled participant — radio button, EmotiBit serial label, and live Physio values
- Accepts `hr_sd` and `eda_sd` from `device:{serial}:physio_metrics`; falls back to legacy HR/EDA plots when those channels are available
- SD metrics are shown in the sidebar when the new standard-deviation Physio stream is active
- Scales dynamically for up to 6 simultaneous EmotiBit devices
- Click a row to focus that participant (magenta bounding box on their video feed)
- Press **R** to open a picklist of detected-but-unassigned EmotiBit serials; select with arrow keys or number keys, confirm with Enter
- Enrolled participants always shown with their EmotiBit serial as bounding box label regardless of radio button state

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

## 🗺️ Roadmap

See [docs/NextSteps.md](docs/NextSteps.md) for planned features and technical details.
