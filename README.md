# Concert Engagement + Physiological Signals

Real-time audience engagement estimation for live performances using pose-based machine learning and EmotiBit physiological signal processing.

> **Release V3.2 — Fused Dual-Camera Scoring & Networked Redis (2026-11):**
> *(landed in the `Handover_SeptSession/` reference build)*
> - **One fused engagement score per person across both cameras** — in dual-camera mode a registered person seen on both feeds previously got a value that flipped between the two independent pipelines. Sightings are now merged by EmotiBit serial into a **single reliability-weighted, temporally-smoothed score** (fuller buffer + more confident identity weigh more), shown identically on both overlays and published once on `device:{serial}:engagement`.
> - **Redis as a true network hub, with zero-config discovery** — `1_START_REDIS.bat` on the hub now also launches Bryan's mDNS advertiser (`_amplify-redis._tcp.local.`), and the EmotiBit publisher + engagement GUI default to `--redis-host auto`, discovering the hub over the network (env `REDIS_HOST` and an explicit `--redis-host <ip>` still override; `--no-discover` opts out). So physio + engagement clients on one laptop reach a Redis broker on another PC with **no IP typing**. The bundled `redis.windows.conf` ships `protected-mode no` (trusted-LAN only) so remote clients are accepted.
> - **Bandwidth-aware dual-camera resolution (4K-capable, not 1080p-capped)** — two USB cameras on one bus are opened together, their **concurrent** frame rate measured, and the highest sustainable mode kept (4K → 1440p → 1080p → 720p). If the shared bus can't sustain it, the app steps down and advises putting each camera on its **own USB controller** (powered hubs don't help).
> - **Scalable overlay** — the physio sidebar lays out in 1–3 columns for 11+ wearers and overlay fonts scale with capture resolution so labels stay legible at 4K.

> **Release V3.1 — Individual Engagement Channels (2026-08):**
> *(landed in the `Handover_JulySession/` reference build)*
> - **Per-participant engagement over Redis** — in addition to the crowd average (`engagement_score`), each registered participant's engagement is now published on its own `device:{serial}:engagement` channel, keyed by the **same EmotiBit serial** the physio publisher uses, so an AR client can pattern-subscribe `device:*:engagement` and merge each person's engagement with their HR/EDA/valence/arousal. Payload carries a `confirmed` flag (positive face ID = green vs appearance-inferred guess = red) plus a numeric `confidence`.
> - **Identity re-binding (appearance re-ID)** — when a registered participant's tracker id churns and their face is too small to re-match, their EmotiBit identity is re-bound from a torso colour signature (+ recent last-known position), so their engagement keeps flowing through track loss instead of going dark. Guesses are flagged distinctly from confirmed face matches and self-correct when a real face reappears.
> - **Identity provenance in the session JSONL** — every row now carries `emotibit_id`, `id_source` (`face` / `coast` / `inferred`) and `id_confidence`, so post-concert analysis can group per participant and filter out low-confidence guesses.
> - **Auto camera resolution** — capture negotiates the camera's highest workable resolution via a ceiling probe + step-down FPS ladder, replacing a fixed 1080p request that silently fell back to 480p on 720p webcams.
> - **Console mirrored to the run log** — stdout/stderr (boot banner, per-publish lines) are tee'd into the engagement log, so a headless/handover session is fully reconstructable from the log alone.

> **Release V3.0 — July Family Lab (2026-07):**
> - **Real-time audio classification** (music / singing / pause) with CSV + WAV recording and automatic microphone selection — new `audio/` component.
> - **Full-rate accelerometer/IMU capture** — raw motion samples are no longer decimated to 1 Hz; every accel/gyro/mag sample is logged to `raw_motion_<serial>_<ts>.csv` and latest raw sensor values stream to Redis (`device:{serial}:raw_sensors`), with Redis Streams mirrors for audio–physio time alignment.
> - **Registered-adults-only engagement** — crowd score and gaze focal-point voting are restricted to face-ID-registered (EmotiBit-wearing) adults; bystanders and infants are never scored or drawn.
> - Reliability fixes: EmotiBit port-conflict detection, disconnect-on-window-close, firewall automation, sidebar shows only devices connected this session.

---

## Current Working Repository

This repository is now the shared source of truth for the concert engagement system.

Work should proceed from Sowmya's `amplify-project/Physiological-Signals` remote rather than the earlier standalone Audience Pose / local `concert_engagement` repository. The previously independent parts of the pipeline have been ported into this project so development can continue from one combined codebase:

- Computer-vision engagement inference, multi-person tracking, face registration, 360-video support, logging, analysis, and training utilities are now in the project-level `scripts/`, `src/`, `models/`, and `docs/` folders.
- Sowmya's newer EmotiBit physiological pipeline is now the project-level Physio implementation in `physio/multiemotibit_UDP_SD_RFv2.py`.
- Will's real-time audio classification (music + singing detection) is now the project-level implementation in `audio/` (script, config, YamNet ONNX + singing-head models).
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

**Output**: Crowd engagement score (0.0-1.0) published to Redis channel `engagement_score` at 1 Hz. Per-participant engagement is additionally published on `device:{serial}:engagement` (one channel per registered EmotiBit). Visual overlay shows red→green engagement bar at top of window. Progressive confidence scoring produces initial estimates within ~1 second of launch, with per-person confidence indicators during buffer ramp-up. Data logging is opt-in via `--save` (both), `--save-engagement` (JSONL only), or `--save-keypoints` (NPZ only). Sessions are saved to `data/sessions/`.

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

**Binary Engagement Classifier** — 82.75% accuracy (**Standard / Concert** model)

The original **Standard / Concert** model (`…binary_v2_cleaned`, 82.75%) was
trained to reward expressive, higher-arousal behaviour:

- **Engaged**: clapping, dancing, cheering, singing, playing instruments, attentive gazing
- **Disengaged**: phone use, sleeping, negative body language, checking time

> ⚠️ **The newer attention-first family-concert model is different.**
> `action_transformer_family_concert_v1` targets a docile, **seated** family
> audience watching performers **at a distance**, so it deliberately **drops the
> performer-style / high-arousal actions** — **dancing, cheering, playing
> instruments, recording** — because a seated audience doesn't do them (and
> performers are excluded from scoring anyway). There, **quiet attentive
> stillness / sustained gaze at the stage is the strongest engagement cue**, and
> the action head is used mainly as a *disengagement* detector (phone use,
> looking away, fidgeting / restlessness, yawning, sleeping). Full kept/dropped
> inventory in [docs/engagement_action_inventory.md](docs/engagement_action_inventory.md).

**Output**: Engagement score (0.0–1.0) published to Redis at 1 Hz

### 🧾 Trained models

Both models share the same `TemporalTransformer` architecture (input 543×3 =
1629, `d_model`=256, 8 heads, 4 layers, mean-pool, 2-class head, ~3.4 M params)
and the same Kinetics-700 feature set (86 fine classes → binary). They differ
only in label semantics and training bias.

| | **Standard / Concert** | **Young Families (v0)** |
|---|---|---|
| Checkpoint | `models/action_transformer_12gpus_binary_v2_cleaned/best_model.pth` | `models/action_transformer_young_families_v0/best_model.pth` |
| Best val accuracy | **82.75%** | **74.15%** |
| Best epoch | 38 / 50 | 13 / 50 |
| Sequence length | 300 | 64 |
| Loss | CrossEntropy (unweighted) | Class-weighted CE (engagement=1.5, disengagement=1.0) |
| Label change | — | `staring` → **engagement** (only dataset delta) |
| Engagement class index | 1 | **0** |
| Hardware | 12× Tesla T4 (DDP) | 3× Tesla T4 (GPUs 0/9/10, DDP) |
| Wall-clock train time | ~hours (12-GPU run) | **~48 min** (50 epochs, ~47–68 s/epoch) |
| Trainer | `train_action_transformer_ddp_v2.py` | generated by `scripts/training/patch_young_families_trainer.py` |

**How accuracy is measured.** Validation accuracy = fraction of held-out
validation clips (Kinetics-700 val split, converted to the binary
engagement/disengagement target) whose predicted class matches the label. The
trainer evaluates the model on this split after every epoch and keeps the
checkpoint with the highest value (`best_model.pth`); the numbers above are that
best epoch. It is **not** yet measured on real concert footage — see
`docs/young_families_engagement_model.md` and `ROADMAP.md` for the planned
rear-/front-camera video evaluation.

> **Why Young Families scores lower (74% vs 83%).** It deliberately re-points
> ambiguous poses toward "engaged" (class-weighted loss + `staring` flip), which
> trades raw val-split accuracy for the behaviour the young-families context
> needs (presume-engaged, penalise only clear disengagement). The two numbers are
> therefore not directly comparable — they optimise different objectives. Peak
> was reached early (epoch 13) then plateaued, so early stopping would give the
> same model faster.

> ⚠️ **Engagement class index differs between models.** For the standard model
> engagement is index **1**; for Young Families it is index **0**
> (`idx_to_action = {0: 'engagement', 1: 'disengagement'}`). Inference code must
> read the correct index — see `scripts/inference/video_engagement_offline.py`.

### 🧠 What the "Young Families" model considers as adult reactions

The young-families profile does **not** read facial micro-expressions. Adult
reactions are inferred from **body pose, head / gaze orientation and coarse
actions** (from the MediaPipe skeleton + YOLO person/object detector), and are
combined with each parent's **wearable physiology** (EmotiBit). Its stance is
*presume engaged, and only subtract when a clear anti-engagement cue appears*:

$$\text{score}=\sigma\!\Big(w_0 + w_m\,p_\text{engaged} - \sum_k \lambda_k\,a_k\Big)$$

where $p_\text{engaged}$ is the transformer's engagement probability, $w_0$ the
default-engaged prior, $a_k$ the anti-cue activations and $\lambda_k$ their
penalties.

**Considered pro-engagement cues**

| Cue | Where handled |
|-----|---------------|
| Clapping / applause | trained action head |
| Dancing / mimicking the performers | trained action head |
| Singing along | action head + mouth/jaw landmarks (or audio) |
| Sustained attention / still head toward the stage | learned `staring`→engaged + head-pose-variance feature |
| Tapping hands on lap (moving to the rhythm) | rhythmic-motion feature |
| Seated / settled | posture feature |

**Considered anti-engagement cues**

| Cue | Where handled |
|-----|---------------|
| Turned away from the performance | torso / head-orientation feature |
| Distracted by their child (attending to / holding them) | body-orientation + arm-pose feature |
| Agitation / a lot of movement | motion-energy feature (high movement ⇒ *dis*engaged here) |
| Phone use | trained action head (`phone_distraction`) |
| Eating / drinking | trained action head (`eating_drinking`) |
| Touching face | hand-to-face-proximity feature |
| Holding an object (bottle / phone) | YOLO object-in-hand feature |
| Sudden synchronised gaze shift across several parents | multi-person temporal feature |

Roughly half the cues come from the **trained action head** (clapping, dancing,
singing, eating/drinking, phone); the rest are **geometric / temporal rule-layer
features** computed from the pose stream per registered adult, per rolling
window. A **standing** adult is treated as a performer and excluded from
audience scoring.

**Data taken into account for the score.** Per registered adult: the
transformer's engagement probability, the rule-layer anti-cue features above,
and — separately, from the EmotiBit wristband — heart rate, electrodermal
activity (skin conductance) and skin temperature, mapped to valence / arousal.
Face recognition is used **only** to match each person to their own EmotiBit, not
to read emotion. Full taxonomy and rationale in
[docs/young_families_engagement_model.md](docs/young_families_engagement_model.md).

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

### 🎯 Per-Participant Engagement Streaming & Identity Re-binding
- **One Redis channel per EmotiBit** — each registered participant's engagement streams on `device:{serial}:engagement` at ~1 Hz, alongside the legacy crowd-average `engagement_score` float. The `{serial}` is the same EmotiBit id (e.g. `MD-V5-0000334`) the physio publisher streams on `device:{serial}:physio_metrics`, so engagement and physiology for one person share a key. Subscribe with `PSUBSCRIBE device:*:engagement`.
- **JSON payload** — `{device, engagement, confirmed, confidence, source, timestamp}`. `confirmed` is `true` for a live/coasting **face** match (green in AR) and `false` for an appearance-**inferred** guess (red in AR); `confidence` is the numeric match strength and `source` is `face` / `coast` / `inferred`.
- **Identity re-binding** — when a registered participant's tracker id churns and their face is too small to re-match, their EmotiBit identity is re-bound from a torso colour-histogram signature (+ recent last-known position), so scoring survives track loss. A real face match always outranks a guess, and a wrong guess self-corrects the moment a face reappears.
- **Provenance in the session JSONL** — every per-person row carries `emotibit_id`, `id_source` (`face` / `coast` / `inferred`) and `id_confidence`, so per-participant post-concert analysis can group by wearer and filter low-confidence guesses. Inferred identities are marked `~NNNN?` in the on-screen overlay.
- See [Handover_JulySession/README_HANDOVER.md](Handover_JulySession/README_HANDOVER.md) for the full AR-dev subscribe contract and a Unity C# example.

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

### 👀 Gaze-First Engagement Scoring (Gaze_Rules)
The per-person engagement score is a **fusion** of a rules-based gaze engine
([scripts/inference/gaze_rules.py](scripts/inference/gaze_rules.py)) and the
action-transformer model. The gaze engine works entirely from the MediaPipe
keypoints the model already consumes — no extra per-frame compute.

- **Gaze rays** — a 2D gaze direction per person from face/shoulder landmarks,
  quality-gated on landmark visibility and head size.
- **Crowd focal point** — least-squares intersection of the audience's rays.
  When a quorum (≥ 50%) agrees within a tolerance cone the focal point locks
  and the show is considered **started**; until then the pipeline runs in
  `PRE-SHOW` mode on the pure action-model score.
- **Per-person gaze score** — cone score around the focal direction with an
  asymmetric EMA: dips are slow (the audience is presumed engaged; divergence
  is usually transient), recovery is fast, and sustained off-focal gaze decays
  faster. Unreadable gaze (face straight at/away from the camera) drifts
  slowly toward neutral instead of freezing. Movement is never penalised.
- **Synchronized shift detection** — when most of the crowd swings its gaze at
  once (a door opens, a flash goes off) the collective dip is suppressed and a
  `SHIFT!` flag is raised instead of penalising everyone.
- **Performer detection** — sticky promotion via mobility (roaming the space),
  prolonged standing, or standing while the seated crowd faces them.
  Performers are drawn in orange, never scored into the crowd average, keep a
  permanent colour-histogram appearance signature for re-identification after
  track loss, and their tracks are ghost-coasted through short occlusions.
- **Fusion** — gaze is the base score; the action model rescues confident pro
  cues and caps confident anti cues:
  `fused = 0.7·gaze + 0.3·action`, floored at 0.65 when `action ≥ 0.75`,
  capped at 0.40 when `action ≤ 0.25`. Gaze scores are live within a couple of
  frames, so newcomers are scored without the ~4 s model warm-up.
- **Registered adults only** — with face-ID active, only enrolled
  (EmotiBit-wearing) adults are gaze-scored, and only their rays vote for the
  crowd focal point; bystanders and infants are never scored. Performer
  detection still works for unregistered people (bbox mobility). With
  `--no-face-id` or in 360° mode, everyone is scored.
- **Stable IDs in crowds** — tuned BoT-SORT config
  ([scripts/inference/botsort_gaze.yaml](scripts/inference/botsort_gaze.yaml)):
  90-frame lost-track buffer, tighter match threshold, YOLO at `imgsz` 1280
  offline for small far-away performers.
- **Offline tuning harness** —
  [scripts/inference/video_gaze_rules_offline.py](scripts/inference/video_gaze_rules_offline.py)
  renders the full rules overlay (rays, focal point, performer boxes,
  per-person scores) on recorded footage; thresholds were tuned on the 4th
  Family Lab video (dancer / sax / accordion / full-band segments).

**Known caveats:**

- **Fewer than 3 gaze rays (e.g. 2 registered adults)** — a ray-based focal
  point needs ≥ 3 rays (two 2D rays always intersect *somewhere*, so a 2-ray
  focal is degenerate noise). Small audiences are instead scored against the
  **performer's bounding box** as the gaze target: once a performer is
  promoted the show counts as started (`Gaze: LIVE perf:1`) and "off-focal"
  means looking away from the performer. Shift detection scales down too —
  with 2 people, *both* must swing their gaze together to register a `SHIFT!`.
- **Non-moving seated musician** — performer promotion relies on mobility,
  prolonged standing, or standing-while-faced. A musician who stays *seated
  and stationary* (e.g. at a piano) is never promoted, so with < 3 rays there
  is no gaze target at all: scoring falls back to the pure action-model score
  until the performer moves or stands. This is a deliberate conservative
  fallback — no focal point is invented from insufficient geometry.

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
- **Full-rate raw motion logging** — the 1 Hz processing window only ever kept the latest sample per second, silently dropping ~24 of every 25 motion samples. A dedicated `RawMotionLogger` (queue + per-device writer thread, never blocking the shared UDP receive thread) now records **every** accelerometer/gyroscope/magnetometer sample to `raw_motion_<serial>_<ts>.csv`
- **Redis Streams for time-sync** — `physio_metrics`, `valence_cont` and `arousal_cont` are mirrored into capped Redis Streams, plus a `device:{serial}:raw_sensors` stream (accel/gyro/mag, PPG, temperature, battery, SpO2), enabling timestamp-based alignment with the audio recordings
- **SCRAmplitude sentinel gating** — EmotiBit's ~9999 placeholder values are converted to NaN via a plausibility gate (0–25 µS) with a `quality_scr_amplitude` flag, so placeholders never enter the data as numeric readings

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

## 🎵 Real-Time Audio Detection (Music + Singing)

Standalone audio classifier that logs concert audio state alongside the EmotiBit data (see [audio/README.md](audio/README.md) for full details):

- `audio/audio_monitor_csv.py` — YamNet (ONNX) music detection + MLP singing head on YamNet embeddings; states `PAUSE | MUSIC | SINGING`
- Outputs a detection CSV (~0.48 s hop) and a continuously auto-saved WAV to `emotibit_recordings/`, timestamped for alignment with the physio CSVs
- Thresholds and model paths in `audio/audio_config.py`; models in `audio/models/` (`yamnet_model.onnx`, `sing_detection_head.pt`)
- **Automatic microphone selection** (`--auto-device`) — probes each physical input device for ~1 s and picks the one with the strongest RMS signal (the audio analogue of the camera non-black-frame auto-select); the launcher uses this by default, pass a device ID to override
- Runs entirely on **CPU** (ONNX Runtime CPU provider + tiny MLP) — no GPU required
- Launcher for the demo bundle: `Handover_JulySession/4_START_AUDIO_REALTIME.bat`; from the repo root run `python audio/audio_monitor_csv.py --list-devices` then `python audio/audio_monitor_csv.py --quiet`
- Test helper: `physio/simulate_emotibits.py` publishes synthetic EmotiBit Redis streams for GUI testing without hardware

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
