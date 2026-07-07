# Concert Engagement + Physiological Signals

Real-time audience engagement estimation for live performances using pose-based machine learning and EmotiBit physiological signal processing.

---

## Current Working Repository

This repository is now the shared source of truth for the concert engagement system.

Work should proceed from Sowmya's `amplify-project/Physiological-Signals` remote rather than the earlier standalone Audience Pose / local `concert_engagement` repository. The previously independent parts of the pipeline have been ported into this project so development can continue from one combined codebase:

- Computer-vision engagement inference, multi-person tracking, face registration, 360-video support, logging, analysis, and training utilities are now in the project-level `scripts/`, `src/`, `models/`, and `docs/` folders.
- Sowmya's newer EmotiBit physiological pipeline is now the project-level Physio implementation in `physio/multiemotibit_UDP_SD_RFv2.py`.
- The multiperson GUI consumes the new physiological standard-deviation stream (`eda_sd` and `hr_sd`) from `device:{serial}:physio_metrics`.
- `Handover_JulySession/` remains as the up-to-date handover package and reference copy for the demo-ready workflow.

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

> **Registered-only engagement:** the crowd score now aggregates only people identified by facial recognition as **registered** attendees (the EmotiBit-wearing parents), not the whole crowd. Bystanders are still YOLO-tracked (so they can be face-matched and enrolled) but skip MediaPipe + the transformer entirely and are **not** drawn — only registered parents get bounding boxes, restoring the 12 fps floor. A registered parent's box turns **magenta** only when their EmotiBit row is selected in the sidebar, for at-a-glance owner identification. The physio sidebar now reflows to **two columns** beyond four wearers (up to 8 combos). With face ID active (2D), registered-only stays on even before anyone enrols — **zero registered means zero crowd score** (no bystander is ever averaged in). Only `--no-face-id` or 360° mode (no per-face matching) falls back to whole-crowd aggregation.

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

The pinned stack is `torch==2.6.0+cu124` (see `Handover_JulySession/constraints.txt`).

---

## ✨ Features

### 📷 Smart Camera Detection
- **Phase 1**: Checks cached (last-used) camera index with a full 3-frame real-content test — instant startup when the same camera is plugged in
- **Phase 2**: Quick-scans remaining indices (0–9) with a 3-frame test to detect any additional cameras
- When only one real camera is found, selects it automatically
- **When multiple real cameras exist**, opens a tiled live-preview window showing all feeds simultaneously — press `1`/`2`/`3`... to choose (1-indexed by list position, not OS device index), `Enter` to accept the default (last-used first, then 360°), `Q` to quit
- **`--select-camera` / `-s`** forces the picker to appear on every run, even when only one camera is detected, and bypasses the last-used cache so a freshly plugged-in USB webcam is always discovered. Useful for non-technical end users handing the laptop between people. The bundled `Handover_JulySession/3_START_ENGAGEMENT.bat` passes this flag by default.
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

## 🧠 Physiological Signal Processing (EmotiBit)

**Key features:**
- `physio/multiemotibit_UDP_SD_RFv2.py` - UDP-based multi-device EmotiBit discovery and SD/RF streaming (no LSL/Oscilloscope required)
- Discovers EmotiBit devices automatically via UDP broadcast on port 3131
- Filters EDA (1 Hz lowpass) and HR (0.5 Hz lowpass) in real time
- Random Forest predictions for continuous valence and arousal from EDA + HR features (models: `physio/rf_valence_full_v2.pkl`, `physio/rf_arousal_full_v2.pkl`)
- Computes standard-deviation metrics over live EmotiBit windows: `edl_sd`, `temperature_roc_sd`, `scr_frequency_sd`, `hr_sd`, and `ibi_sd`
- Publishes SD metrics to Redis on `device:{serial}:physio_metrics`, including explicit `eda_sd` and `hr_sd` fields for the multiperson GUI
- Publishes affect outputs to Redis on `device:{serial}:valence_cont` and `device:{serial}:arousal_cont`
- Logs all raw signals, SD metrics, predictions, and processing time to CSV in `emotibit_recordings/`

### Per-wearer session z-scores

Alongside the magnitude SDs, the pipeline publishes signed **per-wearer session z-scores** so the stage-side view shows whether a wearer is rising above or settling below *their own* normal — for HR, EDA, IBI, temperature ROC and SCR frequency:

- **Running mean/SD via Welford's algorithm** — one-pass and numerically stable; converges within ~60–90 s.
- **Fast calibration, expanding baseline** — a ~2 s warm-up (`CALIBRATION_SECONDS = 2`), then z-scores publish as $z_t = (x_t - \mu_n)/\sigma_n$; the baseline keeps refining and **freezes at 60 s** (`BASELINE_CAP_SECONDS`) so short wears stay accurate.
- **Off-wrist handling (≥ 2 s)** — the publisher drops the `*_z` keys and sets `off_wrist: true`; the GUI dims the panel and shows an `OFF-WRIST` watermark. Detected by a low-magnitude rule (EDA < 0.10 µS and PPGGreen < 1500 counts) or a frozen-channel rule (near-zero EDA/PPG variance).
- **Wearer-swap auto-reset** — a sustained off-wrist gap resets the baseline on re-fit, so a new wearer is never scored against the previous one.
- **Skin-temp sensor preference** — Thermopile (medical-grade) over die-temperature when available, with per-sensor plausibility gates (24–42 °C / 22–36 °C).
- **Artefact rejection** — plausibility gates (HR 40–200 BPM, IBI 300–1500 ms), a median IBI pre-filter and a Hampel EDA spike filter; rejected samples never enter the baseline.

The original magnitude `*_sd` fields stay in the `physio_metrics` payload for backwards compatibility; the signed `*_z` fields and the `off_wrist` / `calibrating` booleans are additive.

### 📟 Physio GUI Sidebar
- One row per enrolled participant — radio button, EmotiBit serial label, and live Physio values
- Accepts `hr_sd` and `eda_sd` from `device:{serial}:physio_metrics`; falls back to legacy HR/EDA plots when those channels are available
- SD metrics are shown in the sidebar when the new standard-deviation Physio stream is active
- Scales dynamically for up to 8 simultaneous EmotiBit devices
- Click a row to focus that participant (magenta bounding box on their video feed)
- Press **R** to open a picklist of detected-but-unassigned EmotiBit serials; select with arrow keys or number keys, confirm with Enter
- Enrolled participants always shown with their EmotiBit serial as bounding box label regardless of radio button state

---

## ⚖️ EU AI Act & Data Protection

> **Not legal advice** — informational only; confirm with your DPO / legal counsel and the official [EU AI Act Compliance Checker](https://ai-act-service-desk.ec.europa.eu/en/eu-ai-act-compliance-checker).

This system is **in scope** of Regulation (EU) 2024/1689 (the AI Act) and the GDPR: it performs **emotion recognition** (engagement/affect from face + pose), **biometric identification** (face matching), and **physiological inference** (EmotiBit EDA/HR → valence/arousal). Face embeddings and physiological readings are **special-category personal data**.

Key obligations to observe:

- **Prohibited-use boundary (Art. 5(1)(f))** — emotion recognition is banned in **workplace and education** settings (in force since Feb 2025). Intended use here is live events / audience engagement only.
- **Transparency (Art. 50)** — inform the people exposed to the emotion-recognition / biometric processing (venue signage + an information notice).
- **GDPR (Art. 9 / 35)** — an explicit lawful basis for special-category data and a **DPIA**; enforce retention limits on face embeddings, physio CSVs and session recordings.
- **Risk classification (Annex III)** — confirm whether the specific deployment is high-risk (→ risk management, logging, human oversight, accuracy documentation).

A full obligations **checklist** is maintained in the deployment bundle: [Handover_JulySession/README_HANDOVER.md](Handover_JulySession/README_HANDOVER.md).

Privacy-supporting measures already in the build: registered-people-only scoring (bystanders are not scored, drawn or aggregated), recyclable on-screen IDs (`P1`, `P2`, …), off-wrist data dropped rather than faked, and a local-first (`localhost`) data flow.

---

## 🗺️ Roadmap

Outstanding development work — performance/scaling, engagement-model fairness, and
facial-identification range — is tracked in **[ROADMAP.md](ROADMAP.md)**.
