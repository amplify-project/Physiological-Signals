# Concert Engagement AR System — Full Handover

> **V1 — April 2026**
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

- **Radio buttons** on the sidebar select which detected person to treat as the "focus target" whose engagement score is highlighted and published
- **Engagement score** is displayed as a coloured overlay on each person's bounding box (green = engaged, red = disengaged)
- Press **`q`** or **Escape** to quit gracefully

### EmotiBit device discovery (Window 2)

When `2_START_EMOTIBIT.bat` starts (runs `multiemotibit_UDP_SD_RFv2.py`):
1. Power on your EmotiBit device(s)
2. Wait for them to appear in the console (up to 20 seconds)
3. Press **Enter** to accept, or wait 5 s after the last device — it auto-accepts
4. A live plot window opens showing real-time filtered **EDA** and **Heart Rate** per device

---

## 📡 Redis Channels — What Gets Published

All AR data flows over Redis pub/sub on `localhost:6379`.

### Engagement (Vision system)

| Channel | Payload | Description |
|---------|---------|-------------|
| `engagement:scores` | `{person_id, score, timestamp}` | Per-person binary engagement score |
| `engagement:focus` | `{person_id, score, timestamp}` | Score for the selected focus target only |

### Physiological (EmotiBit)

Replace `{device}` with the EmotiBit's MAC-derived ID (e.g. `MD-V5-0000448`).

| Channel | Payload | Description |
|---------|---------|-------------|
| `device:{device}:hr_filtered` | `{device, HR_filtered, timestamp}` | Low-pass filtered Heart Rate (BPM) |
| `device:{device}:eda_filtered` | `{device, EDA_filtered, timestamp}` | Low-pass filtered EDA (µS) |
| `device:{device}:valence_cont` | `{device, valence, timestamp}` | Continuous Valence score [0–2] |
| `device:{device}:arousal_cont` | `{device, arousal, timestamp}` | Continuous Arousal score [0–2] |

All payloads are JSON strings.

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

- Press **`q`** in the engagement window, or **Ctrl+C** in any terminal
- Close the EmotiBit plot window (press `q` inside it) — this flushes and saves all CSV recordings
- Close the Redis terminal last

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
| Live plot window doesn't open | Python was installed without Tcl/Tk — reinstall Python and tick "tcl/tk and IDLE" |
| `Model not found` | Make sure `model/best_model.pth`, `yolo26n.pt`, `rf_valence_full_v2.pkl`, `rf_arousal_full_v2.pkl` are all present in the folder |
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

---

## ⚠️ Known Issues

See **`KNOWN_ISSUES.md`** for a full list of bugs found and fixed during handover testing (April 2026).
