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

### 📦 Checkpoint Format (PyTorch 2.6+)

`models/action_transformer_12gpus_binary_v2_cleaned/best_model.pth` is stored
as a **slim `state_dict`** (just the 55 model tensors, ~14 MB) — not the full
training-time bundle. This is deliberate:

- PyTorch 2.6 changed `torch.load`'s default to `weights_only=True`, which uses a
  restricted unpickler that rejects optimiser state, argparse `Namespace`
  objects, numpy scalars, etc. A full training checkpoint trips this with
  `WeightsUnpickler error: Unsupported operand …`.
- A bare `state_dict` (a `dict[str, Tensor]`) loads cleanly under the strict
  default, with no `weights_only=False` opt-out and no security trade-off.

**When re-training**, run the re-export utility before committing the new
checkpoint so end-users on torch 2.6+ are not blocked:

```bash
python scripts/utils/reexport_checkpoint.py \
  path/to/training_checkpoint.pth \
  -o models/action_transformer_12gpus_binary_v2_cleaned/best_model.pth
```

The script verifies the output round-trips under `weights_only=True` before
overwriting. **Do not** revert the loader to `weights_only=False` to "fix" a
load error — re-export the checkpoint instead.

The pinned stack is `torch==2.6.0+cu124` (see `Handover_JuneSession/constraints.txt`).

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

#### Per-wearer session z-score (shipped on `Post_Canteen_Bug_Fixes`)

The original `physio_metrics` bundle published raw `numpy.std()` of the 5 s buffer as magnitude-only SDs. The 4 June canteen recordings exposed two limitations: (1) impossible EmotiBit beat-detector values (IBI 80 ms / 6,160 ms, HR > 220 BPM) propagated untouched into the SDs, producing a ~270 ms `ibi_sd` floor; (2) magnitudes don't tell the operator whether a person is rising above or settling below their own normal — the question the stage-side view actually needs to answer.

The replacement is live on `Post_Canteen_Bug_Fixes` and matches the methodology already published in the IMX '26 adult paper (rolling-median + z-score) and the IMEX infant paper (whole-session z-score, motivated by the absence of a resting baseline when sensors are rotated across wearers):

- **Per-device running mean/SD via Welford's algorithm** — one-pass, numerically stable; converges to the true session mean within ~60–90 s and barely moves thereafter, so sustained elevations stay visibly elevated (unlike a short EMA, which habituates).
- **Calibration gate** — first ~60 s after a wearer assignment marked as `calibrating: true` in the published payload; the GUI shows `calibrating Ns` and suppresses spline / readout values until done. Z-scores published thereafter as $z_t = (x_t - \mu_n)/\sigma_n$ for HR, EDA, IBI, temperature ROC and SCR frequency.
- **Off-wrist event (≥ 2 s, short threshold)** — separate from the 15 s baseline-reset threshold. While off-wrist the publisher drops `*_z` keys from the payload entirely and sets `off_wrist: true` so the GUI dims the panel and renders an `OFF-WRIST` watermark within ~2 s of skin-contact loss. Two detection paths:
  - *Low-magnitude rule* — EDA mean < 0.10 µS AND PPGGreen mean < 1500 counts over the last second of raw samples (well-behaved units).
  - *Frozen-channel rule* — EDA std < 0.005 µS AND PPGGreen std < 5 counts over the same window. Catches units whose ADS1114 EDA front-end rails at a high pinned value when removed and whose PPG latches on a fabric reflection (confirmed on `MD-V5-0000448` 9 June, EDA pinned at 2.595 µS identical to 5 dp for 20+ consecutive samples). Reads from the raw `all_signal_values` buffer with no smoothing in front of it.
- **Wearer-swap detection (auto)** — multi-condition trigger: sustained simultaneous off-wrist on EDA + PPG for ≥ 15 s, followed by return-to-plausible, optionally confirmed by a step change in skin temperature baseline. Biased toward missing resets rather than firing false ones.
- **Skin-temp sensor preference** — if Thermopile (MLX90632, medical-grade skin temp; MD-V5 hardware) is streaming it's preferred over the legacy `Temperature1` (MAX30101 die temp, reads ~4–6 °C below skin). Per-sensor plausibility gates: 24–42 °C for Thermopile, 22–36 °C for die temp. Original sensor choice is logged once at startup per device. *(Original code path used `Temperature0`/`Temperature1` with a 30–38 °C gate, so every die-temp sample was being rejected on MD-V5 hardware and STROC never plotted.)*
- **EDA channel fallback** — z-score representative is computed from `EDL` (derived tonic) if available, otherwise falls back to `EDA` (ADS1114 raw, the actual primary path on MD-V5 wearers in current bundle). *(Original code only consulted `EDL`, so on wearers that emit `EA` but not `EL` the EDA spline was silently empty for the whole session.)*
- **Loose-strap handling** — partial channel dropout (EDA falls but PPG/accelerometer continue) does **not** trigger a reset; instead the affected channel is flagged `low-quality` in the publish bundle so the GUI can render it greyed-out rather than as a falsely calm baseline.

**Required upstream filtering (precondition for any baseline approach):**
- Plausibility gates: HR ∈ [40, 200] BPM, IBI ∈ [300, 1500] ms, temperature per-sensor gates as above.
- Short median pre-filter on per-beat IBI to suppress the EmotiBit sample-and-hold artefacts.
- Hampel filter on EDA for spike rejection.
- Rejected samples are excluded from the Welford update so artefacts never enter the running stats.

Backwards-compatibility: the original `eda_sd` / `hr_sd` / `ibi_sd` / `temperature_roc_sd` / `scr_frequency_sd` magnitude fields remain in the `physio_metrics` payload (the operator still wants to read "this person is at HR 110 right now"); the signed `*_z` fields and the `off_wrist` / `calibrating` booleans are added alongside them. The Valence/Arousal RF path is unchanged. To be validated by Sowmya + Eoghan against the canteen recordings before merging to `main`.

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

### Progress log — June 2026 (branch `Post_Bremen_Bug_Fixes`)

Completed work against this roadmap, in commit order:

- **Item 1 — EmotiBit visibility before face registration** ✅
  - Sidebar now plots every connected EmotiBit immediately on discovery, independent of facial enrollment state. Three-state messaging (`unassigned` / `assigned` / `stale`) drives the label; plots stay live in all states. Y-axis autoscale rewritten to zoom-to-fit so tiny variations fill the panel.
  - Commits: `50247b1`, `832d553`, `5de7e84`, `1da0024`.

- **Item 3 — Pipeline redundancy** ✅ (plotting half)
  - Sowmya's separate matplotlib live-plot loop in `multiemotibit_UDP_SD_RFv2.py` was redundant with the GUI sidebar plots — removed. Publisher now runs headless. `matplotlib` pin dropped from both `requirements.txt` copies.
  - Commit: `daa44da`.
  - Note: the **2D-first efficiency gate** for 360° is a sequencing rule, not a code change. 360° tuning is deferred until the 2D path holds ≥ `TARGET_FPS_FLOOR` (12) on demo crowds.

- **Item 2.b — ID inflation / tracker stability** ✅
  - YOLO botsort assigns a fresh monotonically-increasing id every time a person is re-detected after a lost frame. In dense audiences the id space climbed into the 1000s while only ~20 people were present. Each stale id retained a feature buffer (~417 KB), leaking RAM and bloating the per-frame crowd-engagement aggregator.
  - Implemented the timeout-based eviction the original author had left commented out: new `_evict_stale_tracks()` runs at the top of `process_frame` (2D) and `process_view` (360°). Constants: `STALE_TRACK_TIMEOUT_FRAMES = 60` (~2 s @ 30 FPS — comfortably above brief occlusions so botsort's own 30-frame `track_buffer` recovery still works), `MAX_TRACKED_IDS = 64` (hard cap, oldest-unseen evicted first, active ids never touched).
  - Behaviour: id labels still increment monotonically (botsort owns that counter), but `len(person_buffers)` stays bounded and per-frame Python cost no longer grows with stale-id history. Display remapping is an open option for cosmetic id reuse.
  - Commit: `6be189b`.

- **Item 2.c — Context duration drift** ✅
  - The action transformer was trained on 10-second, 300-frame DAiSEE/Kinetics-700 clips at 30 FPS. The per-track feature buffer was a fixed-FRAME `deque` (default 64, clamped [60, 300]), so as live FPS varied the wall-clock window silently drifted: at 3 FPS a 60-frame buffer spans 20 s of real time — wildly off-distribution. The old FPS-adaptive `update_sequence_length` loop could not represent 10 s outside [6, 30] FPS, and the overlay `Context: X.Xs` derived from `sequence_length / live_fps` exhibited the drift seen in the June demo screenshots.
  - Replaced with a time-windowed buffer: entries are `(monotonic_ts, features_flat)` tuples, pruned to `TARGET_DURATION_SECONDS = 10.0` on every append. `_resample_buffer()` interpolates the buffer onto exactly `MODEL_INPUT_FRAMES = 300` evenly-spaced timesteps via vectorised piecewise-linear `np.searchsorted + lerp` immediately before each model call. Inference gate switched from `MIN_INFERENCE_FRAMES` to `MIN_INFERENCE_SECONDS = 1.0`. Overlay reads `system.current_context_seconds` (longest active buffer span, capped at 10 s) so it sits steady at 10.0 s regardless of FPS. Obsolete adaptation loop removed from main loop; `update_sequence_length` retained but unused by the live path.
  - Commit: `9f0fbee`.

- **Item 2.d — Underutilized hardware at low FPS / crowd-load FPS floor** ✅
  - MediaPipe holistic costs ~30–50 ms per person on CPU and runs once per detected person per frame, so 30 people × 40 ms = 1.2 s/frame = 0.8 FPS — the collapse seen in Sowmya's screen grabs.
  - Added an adaptive per-frame MediaPipe budget driven by `TARGET_FPS_FLOOR = 12.0` and an EWMA of measured per-person extract cost. `_select_persons_for_extraction()` round-robins through detected ids with that budget; selected persons get the full MediaPipe + model pass, unselected persons reuse their cached score with the fresh YOLO bbox (still drawn live, still aggregated). The roadmap 2.c time-windowed buffer absorbs the sparse-in-time extraction transparently — the model is unaware of the throttle.
  - Constants: `TARGET_FPS_FLOOR = 12.0`, `MP_BUDGET_FRACTION = 0.70`, `MP_TIME_EWMA_ALPHA = 0.1`, `MIN_PERSONS_PER_FRAME = 1`. Mirrored into the 360° path as a per-view budget. Overlay grows a `MP: selected/total` suffix only when throttling is active.
  - Commit: `6fd1216`.

**Outstanding under Item 2.d:** if the round-robin throttle alone doesn't hit 12 FPS on the largest expected audiences, the next lever is a thread pool of per-worker MediaPipe holistic instances (true parallel extraction). Not implemented yet because MediaPipe holistic objects are not thread-safe — needs a `threading.local` of holistic instances and a `ThreadPoolExecutor` around the per-person loop.

### Progress log — June 2026 (branch `Post_Canteen_Bug_Fixes`)

Quality-of-life fixes from the 4 June canteen run review (data saving + GUI clarity):

- **Console-output throttle (engagement side)** — Redis-publish line now mirrors the EmotiBit publisher's pattern: prints every second for the first 3 publishes, then once per 60 s. Keeps terminal I/O off the hot path during long sessions without losing the ability to confirm the pipeline is alive at startup.

- **MP throttle visibility** — when MediaPipe is round-robin throttled the HUD now shows e.g. `MP: 1/3 rr2`; the `rr` offset increments every frame so the operator can see the rotation cycling. Each bbox of a person who got fresh MediaPipe **this frame** also gets a small cyan dot top-right, so the round-robin is visually traceable across the crowd.

- **Sidebar HRSD + STROC layout** — the temperature ROC SD readout (`Ṡ SD …`) was being drawn on top of the EDA spline plot and was therefore invisible. Both readouts now sit side-by-side on the same row header line **above** the plot rectangle, with shorter labels (`HRSD`, `STROC`) so they fit at `sidebar_w=280`. EDA / EDA SD spline behaviour unchanged. *(Superseded by the unified z-score panel rework — see below.)*

- **Display-ID remap on the overlay** — the on-screen bbox label was using the raw YOLO `bytetrack` id which inflates rapidly in crowds (2 058 distinct ids over a 47 min canteen run with ~6 people on screen). Boxes now read `P1`, `P2`, … from a recyclable pool of small integer slots; raw `track_id` is still what's written to `engagement_data.jsonl` and the keypoint NPZ chunks for offline analysis. Slots are returned to the pool when their underlying track is dropped by `_evict_stale_tracks()` (>60 frames absent or `MAX_TRACKED_IDS=64` cap), so the next new person picks up the smallest free `P*` slot. **Caveat:** if churn is severe enough that >64 distinct tracks are seen within any ~2 s window, eviction will recycle slots while their tracks are still live, which would visibly jump the numbers in front of the operator. The fix in that regime is to bump `MAX_TRACKED_IDS` (the eviction cap) or `STALE_TRACK_TIMEOUT_FRAMES` (the absence threshold) — both constants near line 320 of `live_multiperson_binary_v2.py`.

- **Data-saving validation (canteen run, 47 min, ~6 people on screen, MediaPipe throttled)** — NPZ chunks compress to ~4 MB total across 27 files; the engagement JSONL is 29 MB. NPZ schema is correct (`frames`, `track_ids`, `keypoints (N,543,3) float16`, `bboxes`, `frame_size`). The "MB not GB" surprise is dominated by 2.d throttling (only ~1 of every ~6 detected people gets MP → NPZ) plus ~39 % of buffered keypoint rows being all-zero (Holistic returned no landmarks for that crop). No corruption.

- **Physio rework — unified z-score sidebar + off-wrist watermark (9 June)** — wires the per-wearer Welford z-score path described earlier into the GUI and fixes several silent-bug regressions exposed during the 9 June bench session:
  - **Single panel per wearer, three traces**: HR (green), EDA (yellow / cyan), TEMP (magenta) all share one fixed ±3 SD axis. Header carries three colour-keyed numeric readouts (`HR +z.zz  EDA +z.zz  TEMP +z.zz`) that switch to bold with a faint coloured pill behind them when `|z| ≥ 2`. Each spline segment thickens past `|z| = 2` and the latest sample is marked with a dot (white halo in the alert band). Replaces the old EDA-spline / HRSD-text / STROC-text layout.
  - **Rolling 10-second window** on the plot — publisher emits one `physio_metrics` per second so the panel shows the most recent 10 samples spread across the panel width. Full history stays in the subscriber deque for debugging / replay; only the plot is windowed.
  - **`OFF-WRIST` watermark** within ~2 s of contact loss. The publisher drops `*_z` keys from the payload entirely while off-wrist (faking a zero baseline would lie about skin contact when the device is on a table), so the splines gap out naturally rather than freezing on the last good value. Re-attach and the splines resume immediately; only a sustained ≥ 15 s off-wrist event triggers a full baseline reset.
  - **Frozen-channel off-wrist detector** — some EmotiBit units (`MD-V5-0000448` on 9 June) don't collapse to floor when removed; the ADS1114 EDA front-end rails at a high pinned value (~2.6 µS identical to 5 dp for 20+ consecutive samples) and PPGGreen latches on a fabric reflection (~2 600 counts). Real skin contact always produces measurable jitter, so the detector now ORs the original magnitude rule with a zero-std fallback (EDA std < 0.005 µS AND PPG std < 5 counts over the last second of raw samples). Reads from `DeviceAggregator.all_signal_values` (raw UDP stream, no smoothing in front of it) so the frozen-std signal is genuine.
  - **Skin-temp sensor preference** — Thermopile (MLX90632 medical-grade) is now preferred over `Temperature1` (MAX30101 die temp) where available, with per-sensor plausibility gates (24–42 °C vs 22–36 °C). Fixes the silent-STROC bug on MD-V5 hardware where every die-temp sample was being rejected by the legacy 30–38 °C gate.
  - **EDA fallback (EDL → EDA)** — z-score representative now falls back to `EDA` when `EDL` is empty. Fixes the silent-EDA-spline bug on wearers that emit the `EA` tag but not `EL`.
  - **Main window resizable** — the engagement `cv2.namedWindow` flag was switched from default `WINDOW_AUTOSIZE` (which locks to source frame size) to `WINDOW_NORMAL` so Windows can maximise / snap / fullscreen the GUI.

**Sync status (June 2026):** both `Handover_JuneSession/live_multiperson_binary_v2.py` and `scripts/inference/live_multiperson_binary_v2.py` carry the Post-Bremen and Post-Canteen progress-log items. The only intentional divergence is the `applog` import bootstrap (Handover copy resolves it from its own folder; project-tree copy resolves it from `src/`).

---

### 1. EmotiBit visibility before face registration ✅
- [x] EmotiBit plots should be visible even before being registered to a facial ID.
- [x] Decouple physio visibility from face-enrollment state so operators can validate sensor health immediately.

**Proposed solutions**
- [x] Standard practice: render all discovered EmotiBit streams as unassigned rows immediately, independent of face ID state.
- [x] Standard practice: add assignment states (`unassigned`, `assigned`, `stale`) and keep plotting active in all states.
- [ ] State-of-the-art: add confidence-based auto-association between tracked people and devices using temporal cues, with manual override.

### 2. Multi-person FPS collapse and runtime efficiency (partially ✅)
- [x] During demoing, many people were simultaneously detected. Some FPS reduction per additional person is expected, but frame rate dropped to around 3 FPS.
- [ ] Maintain a minimum FPS target of 12 at all times. *(2D path holds 9–12 FPS post 2.b/2.c/2.d on canteen-class crowds; not yet validated at 30+ people. See Item 2.e for the GPU-keypoint follow-up if the throttle alone is insufficient.)*
- [x] Evaluate adaptive frame processing to maintain 12 FPS (for example, controlled frame dropping, selective per-frame work, or phased inference).

**Proposed solutions**
- [ ] Standard practice: implement a hard real-time budget loop with degradation tiers to enforce minimum 12 FPS.
- [ ] Standard practice: run detector every `N` frames, reuse tracker state between detections, and run pose every `M` frames per target.
- [ ] Standard practice: dynamically downscale input and cap active targets by priority under load.
- [ ] State-of-the-art: asynchronous multi-rate pipeline (capture, detect, pose, classify, render, publish) with bounded queues and frame dropping under backpressure.
- [ ] State-of-the-art: export detector/inference path to optimized runtime (ONNX/TensorRT where available) for lower latency.

#### 2.b. ID inflation and tracker stability ✅
- [x] Additional IDs were repeatedly assigned to the same people after detect/lost/redetect cycles.
- [x] IDs reached the 1000s for about 20 people.
- [x] Review tracker identity persistence and reuse strategy to reduce duplicate IDs.
- [x] Evaluate memory impact of large ID churn and buffer retention.
- [ ] Depending on achieved frame-rate solutions, consider a mode that prioritizes visual tracking of the 6 people wearing EmotiBits.

**Proposed solutions**
- [ ] Standard practice: switch to or tune robust MOT settings (track age, minimum hits, reactivation window) to reduce identity churn.
- [x] Standard practice: separate detector-internal IDs from stable application-level person IDs. *(Display-ID remap `P1`/`P2`/… on overlay; raw `track_id` retained in saved data.)*
- [x] Standard practice: enforce lifecycle cleanup and caps for inactive tracks and stale buffers. *(`_evict_stale_tracks()`, `STALE_TRACK_TIMEOUT_FRAMES=60`, `MAX_TRACKED_IDS=64`.)*
- [ ] State-of-the-art: add appearance re-identification embeddings for long occlusion recovery and identity stitching.

#### 2.c. Context duration drift ✅
- [x] In demo images, FPS was around 3 but context grew to around 20 seconds.
- [x] Context window should stay at 10 seconds.
- [x] Investigate and fix context-duration drift so temporal context remains pinned to target duration.

**Proposed solutions**
- [x] Standard practice: convert context control to time-based buffering instead of fixed frame-floor behavior.
- [x] Standard practice: keep a 10-second target window and resample buffered features to the model input length when FPS is low.
- [x] Standard practice: set minimum sequence constraints from inference viability (first estimate threshold), not from static frame counts. *(Now `MIN_INFERENCE_SECONDS=1.0`.)*
- [ ] State-of-the-art: include time-delta encoding for irregular frame spacing in the temporal model.

#### 2.d. Underutilized hardware at low FPS ✅
- [x] Demo images show around 3 FPS while CPU and GPU were not fully burdened.
- [x] Profile pipeline stages to identify serialization bottlenecks and non-hardware-limited stalls. *(MediaPipe Holistic on CPU TFLite identified as the dominant per-person cost.)*
- [x] Evaluate parallel and asynchronous execution paths where safe and measurable.

**Proposed solutions**
- [x] Standard practice: add stage-level wall-time instrumentation (capture, detect, pose, classify, render, publish, logging). *(EWMA per-person extract cost feeds the budget loop.)*
- [ ] Standard practice: remove blocking synchronization points and pre-allocate tensors/buffers to reduce per-frame overhead.
- [ ] Standard practice: parallelize independent CPU-heavy tasks (pose per target/view) with bounded worker pools. *(Round-robin throttle in place; true thread pool deferred — MediaPipe holistic is not thread-safe.)*
- [ ] State-of-the-art: add timeline tracing for queue wait vs compute time and schedule work by deadlines.

#### 2.e. GPU-native keypoint extractor (move off MediaPipe)
- [ ] Root cause behind the GPU idling in 2.d: MediaPipe Holistic's Python binding is CPU-only TFLite. The GPU graph exists in MediaPipe C++ but is not exposed through Python, so on RTX-class hardware the card sits near 0% while one CPU core saturates. Per-person cost (~30–50 ms) is the hard ceiling that the 2.d round-robin throttle only papers over.
- [ ] Measure first: confirm post-patch FPS on representative crowd sizes (10, 20, 30 people, 2D and 360°) with 2.b + 2.c + 2.d in place. Only commit to a swap if the throttle alone cannot hold the 12 FPS floor.
- [ ] If a swap is warranted, the action transformer was trained on the MediaPipe 543-keypoint schema (33 pose + 468 face mesh + 21 + 21 hands, each (x, y, z) → 1629 floats). Any replacement must either provide an adapter onto that schema or trigger a retrain.

**Proposed solutions**
- [ ] Standard practice: **DWPose / RTMPose via ONNXRuntime-CUDA or TensorRT.** 133 whole-body keypoints (body + hands + face contour), batched across all detected persons in a single GPU call — directly addresses the 30-person collapse. Loses dense face mesh; needs a keypoint adapter to the 543-point schema, or a retrain on the reduced schema. Some scaffolding already exists under `src/dwpose_engagement/`.
- [ ] Standard practice: **MMPose (PyTorch).** Same model family as DWPose, easier to finetune, heavier dependencies. Useful if we decide to retrain rather than adapt.
- [ ] Standard practice: **MediaPipe Tasks GPU C++ with a pybind shim.** Preserves the exact 543-keypoint topology so no action-transformer retrain, but binding work is non-trivial and the graph is still single-process.
- [ ] State-of-the-art: **Sapiens (Meta, 2024).** Best-in-class accuracy, larger VRAM footprint, batched. Overkill for live demo today, viable target if hardware scales.
- [ ] Evaluation plan: small benchmark script — batched DWPose vs current MediaPipe on a recorded demo clip — comparing FPS at N = {1, 10, 20, 30} persons and per-keypoint agreement on the points that exist in both schemas. Decide adapter-vs-retrain from the agreement numbers and a held-out DAiSEE pass.

### 3. Pipeline redundancy and 2D-first optimization gate (partially ✅)
- [x] There is pipeline redundancy with double plotting and related duplicate work. *(Sowmya's matplotlib live-plot loop removed; publisher runs headless.)*
- [x] Streamline plotting/data paths before adding complexity.
- [ ] Establish an efficiency gate for 2D perspective mode before proceeding to 360 feeds. *(Gate is the 12 FPS floor under Item 2; not yet formalised as a regression check.)*
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
- [ ] Documented detector/recogniser limits (facenet-pytorch, MTCNN `min_face_size=40`, InceptionResnetV1 trained on 160×160 crops): detection floor ≈ 40 px face, recognition usable from ~80 px, reliable from ~120 px. On a 1080p / ~60° FOV webcam this maps to ≈ 7.3 m / 3.7 m / 2.4 m respectively; halve for the Insta360 5.7K equirectangular feed.
- [ ] Enrollment distance bounds runtime range: a far enrollment (≤100 px face) stores an upsampled/low-information embedding that no runtime crop can rescue. Effective runtime range ≈ min(enrollment quality, runtime crop quality).

**Proposed solutions**
- [ ] Standard practice: make face ID optional and non-blocking so core engagement remains stable at long range.
- [ ] Standard practice: use higher-resolution crops and match only on stable frontal frames.
- [ ] Standard practice: maintain person identity between sparse face matches using tracker continuity.
- [ ] Standard practice: **multi-angle / multi-shot enrollment.** Capture 3–5 crops per person at enrollment (frontal + slight left/right yaw, ideally at ≤ 1.5 m on 2D / ≤ 1 m on 360°), store as a small per-person gallery, and match runtime crops against the nearest-neighbour in the gallery (or against a quality-weighted average embedding). Cheap to implement on top of the existing `facenet-pytorch` path, no model retrain, and is the standard fix for the "enrollment ceiling" failure mode.
- [ ] Standard practice: enforce an enrollment quality gate (minimum face-pixel size, frontal pose, sharpness) and prompt the user to re-enrol if not met, so the runtime envelope is predictable.
- [ ] State-of-the-art: fuse face and body re-identification embeddings for longer-range identity persistence.

### 6. Data saving overhead not yet demo-validated (partially ✅)
- [x] The demo did not include data saving overhead. *(47-min canteen run with `--save-engagement` + `--save-keypoints` did not destabilise FPS; NPZ schema validated.)*
- [x] Run end-to-end performance validation with save options enabled.
- [ ] Quantify FPS and latency impact for `--save`, `--save-engagement`, and `--save-keypoints` in realistic multi-person sessions. *(Comparative no-save vs save benchmark still pending.)*

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

See the **Post Jazzahead Demo Roadmap** above for the live tracker of efficiency, ID stability, and model-fairness work.
