# Concert Engagement AR System — Full Handover

> **V2 — June 2026**
> Contains the complete real-time engagement pipeline: computer vision inference, physiological signal processing (EmotiBit), Redis pub/sub streaming, and the Unity C# subscriber for AR glasses.

This is a **self-contained package**. No other repositories needed.

---

## 📦 Contents

```
handover_ar_v1/
├── SETUP.bat                        ← Run this ONCE on first install
├── 1_START_REDIS.bat                ← Step 1: start Redis server
├── 2_START_EMOTIBIT.bat             ← Step 2: start EmotiBit publisher
├── 3_START_ENGAGEMENT.bat           ← Step 3: start camera + AI inference
├── requirements.txt                 ← All Python deps, fully pinned
│
├── live_multiperson_binary_v2.py    ← Main engagement inference + GUI
├── face_identifier.py               ← Face registration module
├── multiemotibit_UDP_SD_RFv2.py     ← EmotiBit SD metrics → Valence/Arousal → Redis (multi-device)
├── redis_subscriber_all_devices.py  ← CLI diagnostic: prints all Redis channels
├── test_subscriber.py               ← CLI diagnostic: prints engagement scores
├── UnityRedisSubscriber.cs          ← Drop-in Unity C# Redis subscriber
│
├── model/
│   └── best_model.pth               ← Temporal Transformer (82.75% accuracy)
├── yolo11n.pt                       ← YOLO11-nano person detector weights (primary)
├── yolo26n.pt                       ← YOLO26-nano person detector weights (fallback)
├── rf_valence_full_v2.pkl           ← Random Forest — continuous Valence [0-2]
├── rf_arousal_full_v2.pkl           ← Random Forest — continuous Arousal [0-2]
│
└── redis/
    ├── redis-server.exe             ← Bundled Redis server (Windows)
    ├── redis-cli.exe
    └── redis.windows.conf
```

---

## 🖥️ System Requirements

- **Windows 10/11** (bundled Redis server is Windows-only; macOS/Linux need Redis installed separately)
- **Python 3.10, 3.11, or 3.12** — download from https://www.python.org/downloads/
  - During install: tick **"Add Python to PATH"** and **"tcl/tk and IDLE"**
- **NVIDIA GPU** recommended (CUDA 12.4) — runs on CPU but will be slow
- **Redis** — bundled for Windows in the `redis/` folder
- An **EmotiBit** wearable device (MD-V5 or newer) on the same Wi-Fi network

---

## 🚀 Quick Start (Windows)

### First time only — run setup

Double-click **`SETUP.bat`**

This will:
1. Create a `.venv` virtual environment
2. Install PyTorch (CUDA 12.4 for NVIDIA GPU)
3. Install all remaining dependencies at pinned, verified versions
4. Download the Face ID weights into `model/20180402-114759-vggface2.pt`

> If you don't have an NVIDIA GPU, open `SETUP.bat` in Notepad first and swap the PyTorch install line (instructions are inside the file).

---

### Every session — run in order

Open **three separate terminal/command prompt windows** and run one script in each:

| Window | Script | Purpose |
|--------|--------|---------|
| 1 | `1_START_REDIS.bat` | Redis message broker (keep open the whole session) |
| 2 | `2_START_EMOTIBIT.bat` | EmotiBit multi-device EDA/HR processing → publishes to Redis |
| 3 | `3_START_ENGAGEMENT.bat` | Camera + AI engagement inference → publishes to Redis |

**Order matters** — start Redis first, then EmotiBit, then Engagement.

> **VS Code users:** The batch files are designed to be double-clicked from Explorer (each opens its own `cmd.exe` window). Inside VS Code's integrated terminal they all run in the same tab and block each other. Either double-click from Explorer, or open three separate VS Code terminal tabs and run the venv + script directly:
> ```
> .venv\Scripts\Activate.ps1; python multiemotibit_UDP_SD_RFv2.py # tab 2
> .venv\Scripts\Activate.ps1; python live_multiperson_binary_v2.py # tab 3
> ```

---

## 🎮 Using the Engagement GUI

The main engagement window (`3_START_ENGAGEMENT.bat`) opens an OpenCV camera view with a sidebar:

- **Bounding boxes** are labelled `P1`, `P2`, … — small recyclable display IDs that stay stable for the operator. The raw tracker ID is still recorded in the saved data for offline analysis.
- **Engagement score** is shown as a coloured overlay on each person's bounding box (green = engaged, red = disengaged).
- A small **cyan dot** in the top-right of a bounding box means that person received fresh MediaPipe extraction this frame (the rest reuse their last score). When the system is throttling under crowd load the HUD shows e.g. `MP: 1/3 rr2`.
- **Sidebar** shows one row per enrolled person with their EmotiBit serial, plus three readouts:
  - **EDA spline plot** — live electrodermal activity over the rolling window.
  - **HRSD** — 5-second standard deviation of heart rate.
  - **STROC** — 5-second standard deviation of skin-temperature rate-of-change.
- **Radio buttons** on the sidebar select which detected person to treat as the "focus target".
- Press **`R`** to open a picklist of detected-but-unassigned EmotiBit serials; arrow keys / number keys to select, **Enter** to confirm, **Esc** to cancel.
- Press **`q`** or **Esc** to quit gracefully.

### CLI flags for `live_multiperson_binary_v2.py`

The `3_START_ENGAGEMENT.bat` file invokes the script with sensible defaults; pass extra flags by editing the .bat or running the script directly inside the venv:

| Flag | Purpose |
|------|---------|
| `--camera N` | Pick camera index (0, 1, 2, …) |
| `--select-camera` / `-s` | Interactive camera picker at startup |
| `--video PATH` | Run on a recorded video file instead of a live camera |
| `--save` | Save **everything**: engagement JSONL + keypoint NPZ chunks + raw video |
| `--save-engagement` | Save only `engagement_data.jsonl` (~370 MB / 30 min) |
| `--save-keypoints` | Save only the compressed MediaPipe keypoint NPZ chunks |
| `--save-dir PATH` | Override the default `data/sessions/<timestamp>/` location |
| `--no-face-id` | Skip the face-ID enrol step (faster startup, no per-person identity) |
| `--redis-host HOST` / `--redis-port N` | Point at a non-default Redis broker |
| `--device cuda` / `cpu` / `mps` | Force a specific compute device (default `auto`) |

### EmotiBit device discovery (Window 2)

When `2_START_EMOTIBIT.bat` starts (runs `multiemotibit_UDP_SD_RFv2.py`):
1. Power on your EmotiBit device(s).
2. Wait for them to appear in the console (up to 20 seconds).
3. **Press ENTER** once all expected devices are listed to begin streaming.
4. The publisher then runs **headless** — there is no separate matplotlib window. All physio plots are rendered inside the engagement GUI sidebar.

---

## 📡 Redis Channels — What Gets Published

All AR data flows over Redis pub/sub on `localhost:6379`.

### Engagement (Vision system)

| Channel | Payload | Description |
|---------|---------|-------------|
| `engagement_score` | `{frame, persons:[{id, score, bbox, identified, …}], crowd_avg, fps, ts}` | Single per-frame snapshot for all detected persons. The Unity / `test_subscriber.py` consumers extract their target by id. |

### Physiological (EmotiBit)

Replace `{device}` with the EmotiBit's MAC-derived serial (e.g. `MD-V5-0000448`).

| Channel | Payload | Description |
|---------|---------|-------------|
| `device:{device}:physio_metrics` | `{device, timestamp, eda_sd, edl_sd, hr_sd, ibi_sd, temperature_roc_sd, scr_frequency_sd}` | All 5 s standard-deviation metrics published once per second. The GUI sidebar consumes this. |
| `device:{device}:valence_cont` | `{device, valence, timestamp}` | Continuous Valence score [0–2] from the Random Forest. |
| `device:{device}:arousal_cont` | `{device, arousal, timestamp}` | Continuous Arousal score [0–2] from the Random Forest. |

All payloads are JSON strings.

### Planned change — per-wearer session z-score (under validation on `Post_Canteen_Bug_Fixes`)

The 4 June canteen recordings revealed two issues with the current `physio_metrics` bundle: (1) impossible EmotiBit beat-detector values (IBI 80 ms / 6,160 ms, HR > 220 BPM) flow untouched into the SDs, producing a ~270 ms `ibi_sd` floor; (2) magnitude-only SDs don't tell the operator whether a wearer is rising above or settling below their own normal. The planned replacement mirrors the methodology in the IMX '26 adult paper (rolling-median + z-score) and the IMEX infant paper (whole-session z-score, motivated by the absence of a resting baseline when sensors rotate across wearers):

- **Per-device running mean/SD (Welford)** — converges to the true session mean within ~60–90 s and barely moves thereafter, so sustained elevations stay visibly elevated (a short EMA would habituate them away).
- **Calibration gate** — first ~60 s marked "calibrating" in the GUI; thereafter publish signed $z_t = (x_t - \mu_n)/\sigma_n$ for HR, EDA, IBI, temperature ROC, SCR frequency alongside the existing magnitude fields.
- **Wearer-swap detection (auto)** — fires only on sustained simultaneous silence on all channels (EDA + PPG + temperature + accelerometer) for ≥ 30 s followed by return-to-plausible plus a step change in skin temperature baseline. Biased to miss rather than false-trigger.
- **Manual "new wearer" button** in each GUI sidebar tile — resets that device's Welford state and re-arms the calibration gate. Operator override for the auto-detector.
- **Loose-strap handling** — partial channel dropout flags the affected channel as `low-quality` in the publish bundle; does not trigger a reset.

**Required upstream filtering** (precondition; rejected samples are excluded from Welford updates):
- HR ∈ [40, 200] BPM, IBI ∈ [300, 1500] ms, temperature ∈ [30, 38] °C plausibility gates.
- Median pre-filter on per-beat IBI to suppress EmotiBit sample-and-hold artefacts.
- Hampel filter on EDA for spike rejection.

Existing `eda_sd` / `hr_sd` / `ibi_sd` / `temperature_roc_sd` / `scr_frequency_sd` fields stay in the payload (so Unity and the GUI sidebar continue to work unchanged); `*_z` fields and a `quality` flag are added alongside. The Valence/Arousal RF path is unchanged. To be validated by Sowmya + Eoghan on `Post_Canteen_Bug_Fixes` before merging to `main`.

---

## 🔌 Unity Integration

Drop **`UnityRedisSubscriber.cs`** into your Unity project. It subscribes to the Redis channels above and exposes the scores to your AR scene. Requires the [StackExchange.Redis](https://github.com/StackExchange/StackExchange.Redis) NuGet package in your Unity project.

---

## ✅ Verifying Data Flow

With Redis and at least one publisher running, open a terminal and run:

```bash
# Activate venv first
.venv\Scripts\activate

# Prints all EmotiBit channels as they stream:
python redis_subscriber_all_devices.py

# Prints engagement scores from the CV system:
python test_subscriber.py
```

Both should print JSON messages every second when the systems are running.

---

## 🛑 Stopping

- Press **`q`** or **Esc** in the engagement window, or **Ctrl+C** in any terminal.
- In Window 2 (EmotiBit publisher), press **Ctrl+C** to flush and save all CSV recordings.
- Close the Redis terminal last.

---

## 📁 Session Recordings (EmotiBit)

Every EmotiBit session automatically saves to an `emotibit_recordings/` folder (created next to `multiemotibit_UDP_SD_RFv2.py`). Each session produces:
- A `.csv` with **all** EmotiBit signals (EDA, HR, PPG, accelerometer, gyroscope, temperature, SpO2, …)
- A `.json` metadata file with device info and session timestamps

## 📁 Engagement And Keypoint Recordings

`3_START_ENGAGEMENT.bat` runs the engagement system in save mode. Each engagement session automatically creates a timestamped folder under:

```
data\sessions\
```

Example:

```
data\sessions\2026-05-26_10-15-30_a1b2c3d4e5f6\
```

Each session folder contains:
- `engagement_data.jsonl` - per-frame/per-person engagement records, bounding boxes, crowd average, FPS, and face ID fields when available
- `keypoints_001.npz`, `keypoints_002.npz`, ... - compressed MediaPipe keypoint chunks saved periodically during the session

If data saving has to stop because the disk or memory is full, the live engagement system continues running and shows a warning banner in the video window.

---

## 🔧 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named torch` | Run `SETUP.bat` — venv not set up yet |
| `redis.exceptions.ConnectionError` | Start `1_START_REDIS.bat` before the other scripts |
| No EmotiBit devices found | Check device is powered on and on the **same Wi-Fi** network; run the .bat as Administrator (it will auto-add firewall rules) |
| Engagement window black / no camera | Check camera index — pass `--camera 1` (or 2) as an argument to `3_START_ENGAGEMENT.bat` |
| `FaceIdentifier init failed` | Re-run `SETUP.bat` so it installs the Face ID weights, or place `model/20180402-114759-vggface2.pt` in the project manually |
| `Model not found` | Make sure `model/best_model.pth`, `yolo26n.pt`, `rf_valence_full_v2.pkl`, `rf_arousal_full_v2.pkl` are all present in the folder |
| `WeightsUnpickler error: Unsupported operand …` when loading `best_model.pth` | The repo's `best_model.pth` was overwritten with a full training checkpoint instead of a slim `state_dict`. Run `git lfs pull` to refresh, or re-export it with `python scripts/utils/reexport_checkpoint.py <bundle.pth> -o models/.../best_model.pth`. **Do not** patch the code to pass `weights_only=False` — see the Model Checkpoint section in the project root `README.md`. |
| `best_model.pth` resolves to a path outside the repo (e.g. `C:\Users\<name>\AMPLIFY\models\...`) | You're on a pre-`5590200` commit. `git pull` on `GUI_Bug_Fix` to get the path-detection fix. |
| `git pull` overwrote local edits | Always `git stash -u` (or commit) **before** `git checkout` / `git pull`. Recovery: `git reflog` for committed work, VS Code Timeline (`View → Open View → Timeline`) for unsaved-to-git edits. |
| PyTorch CUDA not detected | Check NVIDIA drivers are up to date; run `python -c "import torch; print(torch.cuda.is_available())"` |

---

## 🏗️ Architecture Overview

```
EmotiBit device(s)                 Machine running this system
  EDA, HR, PPG, …  ──UDP──▶  multiemotibit_UDP_SD_RFv2.py
                                   │  filter + RF predict
                                   ▼
Camera feed         ──────▶  live_multiperson_binary_v2.py
  YOLO11 detect                    │  Temporal Transformer
  MediaPipe pose                   │  engagement score
  Face ID                          │
                                   ▼
                              Redis (localhost:6379)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
            AR Glasses (Unity)            Any other subscriber
            UnityRedisSubscriber.cs       test_subscriber.py
```

---

## 🐍 Python Version

**Tested with Python 3.10, 3.11, 3.12** on Windows 11.
