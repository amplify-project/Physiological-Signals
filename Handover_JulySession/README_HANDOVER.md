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
├── gaze_rules.py                    ← Gaze-first rules engine (focal point, performer detection)
├── botsort_gaze.yaml                ← Tuned BoT-SORT tracker config (stable IDs in crowds)
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

### Every session — just double-click the batch files, in order

**No terminal or typing needed** — from Windows Explorer, double-click the three numbered `.bat` files **in order**. Each opens its own window; leave all three open for the whole session, and wait for each to be ready before starting the next:

| # | Double-click | What happens |
|---|--------------|--------------|
| 1 | `1_START_REDIS.bat` | Starts the Redis message broker — keep this window open the whole session |
| 2 | `2_START_EMOTIBIT.bat` | Discovers your EmotiBit(s) and publishes EDA/HR to Redis (press **ENTER** once your devices are listed) |
| 3 | `3_START_ENGAGEMENT.bat` | Starts the camera + AI engagement inference and opens the GUI |

**Order matters** — Redis (1) must be running before EmotiBit (2), and both before Engagement (3).

> **VS Code users:** The batch files are designed to be double-clicked from Explorer (each opens its own `cmd.exe` window). Inside VS Code's integrated terminal they all run in the same tab and block each other. Either double-click from Explorer, or open three separate VS Code terminal tabs and run the venv + script directly:
> ```
> .venv\Scripts\Activate.ps1; python multiemotibit_UDP_SD_RFv2.py # tab 2
> .venv\Scripts\Activate.ps1; python live_multiperson_binary_v2.py # tab 3
> ```

---

## 🎮 Using the Engagement GUI

The main engagement window (`3_START_ENGAGEMENT.bat`) opens an OpenCV camera view with a sidebar:

- **Bounding boxes** are labelled `P1`, `P2`, … — small recyclable display IDs that stay stable for the operator. The raw tracker ID is still recorded in the saved data for offline analysis.
- **Engagement score** is shown as a coloured overlay on each person's bounding box (green = engaged, red = disengaged). The score is a **fusion of two signals**: the gaze rules (base) and the action-transformer model (rescue/override) — see the *Gaze-first engagement scoring* section below.
- **Performers** are drawn with an **orange** box labelled `PERFORMER` and are excluded from the crowd score. The box thickens (with a `<<` marker) when the crowd's shared gaze point sits on that performer.
- **Gaze rays** — thin lines from each audience member's head toward where they are looking, projected to the crowd's shared focal point. Performers never get a ray, and rays are clipped at performer boxes.
- The **HUD** shows the gaze engine state: `Gaze: PRE-SHOW` (no shared focal point yet — pure action-model scoring) or `Gaze: LIVE perf:N` (focal point locked, N performers detected). `SHIFT!` flags a synchronized crowd gaze shift (e.g. a door opening).
- A small **cyan dot** in the top-right of a bounding box means that person received fresh MediaPipe extraction this frame (the rest reuse their last score). When the system is throttling under crowd load the HUD shows e.g. `MP: 1/3 rr2`.
- **Sidebar** shows one row per enrolled person with their EmotiBit serial. Each row has:
  - **Header readouts** — three small colour-coded values right-aligned next to the serial: `HR ±z.zz`, `EDA ±z.zz`, `TEMP ±z.zz`. Values are per-wearer session z-scores (deviation from this wearer's own running mean, in their own SD units). When `|z| ≥ 2` the readout switches to bold with a faint coloured pill behind it.
  - **Unified physio panel** — single rolling spline plot on a fixed ±3 SD axis with reference lines at 0 and ±2 SD. Three traces share the panel: HR (green), EDA (yellow / cyan), TEMP (magenta). Each segment darkens and thickens past `|z| = 2` and the latest sample is marked with a small dot (plus a white halo when in the alert band). Window is the most recent **10 seconds** — older points scroll off the left.
  - **Calibration** — for the first ~2 s after a wearer is assigned the panel shows `calibrating Ns` while the Welford baseline reaches its minimum sample floor; no splines are drawn yet. Plotting then begins almost immediately and the baseline keeps **expanding** (converging over ~60 s, then frozen at a 60 s cap) so early z-scores refine as more samples arrive.
  - **OFF-WRIST** — when sensor contact is lost the panel dims and a faint red `OFF-WRIST` watermark appears within ~2 s. Splines gap out naturally rather than freezing on the last good value. Re-attach the device and the splines resume immediately; a sustained ≥ 2 s off-wrist gap forces a fresh baseline on re-fit (so a new wearer is never plotted against the previous wearer's mean / SD).
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

## � Gaze-first engagement scoring (Gaze_Rules)

The engagement score is no longer the raw action-transformer output. A rules-based **gaze engine** (`gaze_rules.py`) provides the base score, fused with the model:

**How the gaze engine works** (all from the same MediaPipe keypoints the model already consumes — no extra compute):

1. **Gaze rays** — a 2D gaze direction per person from face/shoulder landmarks, quality-gated (visibility, head size).
2. **Crowd focal point** — least-squares intersection of the rays. When a quorum of the audience (≥ 50%) agrees within a tolerance cone, the focal point locks and the show is considered **started**. Until then the pipeline stays in `PRE-SHOW` and uses the pure action-model score.
3. **Per-person score** — a cone score around the focal direction, smoothed with an asymmetric EMA: dips are slow (audience presumed engaged, divergence usually transient), recovery is fast. Sustained off-focal gaze decays faster. Unreadable gaze (face straight at/away from camera) drifts slowly toward neutral rather than freezing.
4. **Synchronized shift detection** — if most of the crowd swings its gaze at once (a door opens, a phone flash), the score dip is suppressed and `SHIFT!` is flagged instead of penalizing everyone.
5. **Performer detection** — sticky promotion of people who move around the space, stand for long periods, or stand while the seated crowd faces them. Performers keep an appearance signature so they are re-recognized after track loss (ghost coasting + colour histogram matching) and are **never scored** into the crowd average.

**Fusion with the action model** (constants at the top of `live_multiperson_binary_v2.py`):

```
fused = 0.7 * gaze + 0.3 * action
if action ≥ 0.75 and fused < 0.65 → fused = 0.65   # confident pro cue rescues
if action ≤ 0.25              → fused = min(fused, 0.40)  # confident anti cue caps
```

- Gaze scores are available within a couple of frames (no 4 s model warm-up), so people entering the frame are scored almost immediately.
- **Registered adults only** — with face-ID active, only enrolled (EmotiBit-wearing) adults are gaze-scored and only their rays vote for the focal point; bystanders and infants are never scored. Performers are still detected among unregistered people via movement.
- Before the show starts (no shared focal point) scoring falls back to the pure action model.
- Tracking uses the bundled **`botsort_gaze.yaml`** (long lost-track buffer, tighter match threshold) so IDs survive occlusion in crowds.

All thresholds are documented in `gaze_rules.py` and were tuned on the 4th Family Lab video (dancer / sax / accordion / full-band segments).

---

## �📡 Redis Channels — What Gets Published

All AR data flows over Redis pub/sub on `localhost:6379`.

### Engagement (Vision system)

| Channel | Payload | Description |
|---------|---------|-------------|
| `engagement_score` | `{frame, persons:[{id, score, bbox, identified, …}], crowd_avg, fps, ts}` | Single per-frame snapshot. In the default **registered-people-only** mode `persons` and `crowd_avg` cover only enrolled (EmotiBit-wearing) participants — bystanders are tracked cheaply by YOLO for enrolment but are not scored, drawn or aggregated. The Unity / `test_subscriber.py` consumers extract their target by id. |

### Physiological (EmotiBit)

Replace `{device}` with the EmotiBit's MAC-derived serial (e.g. `MD-V5-0000448`).

| Channel | Payload | Description |
|---------|---------|-------------|
| `device:{device}:physio_metrics` | `{device, timestamp, eda_sd, edl_sd, hr_sd, ibi_sd, temperature_roc_sd, scr_frequency_sd, hr_z, eda_z, ibi_z, temperature_roc_z, scr_frequency_z, calibrating, calibration_remaining_s, baseline_n, session_age_s, off_wrist, quality, ...events}` | Published once per second. Magnitude `*_sd` fields are always present; signed per-wearer session `*_z` fields are present once calibration is complete AND the wearer is on-wrist (the publisher drops them otherwise, so the GUI can grey the readouts instead of carrying a stale value). `off_wrist` is a boolean the GUI uses to dim the panel and draw an `OFF-WRIST` watermark within ~2 s of contact loss. |
| `device:{device}:valence_cont` | `{device, valence, timestamp}` | Continuous Valence score [0–2] from the Random Forest. |
| `device:{device}:arousal_cont` | `{device, arousal, timestamp}` | Continuous Arousal score [0–2] from the Random Forest. |

All payloads are JSON strings.

### Per-wearer session z-score (shipped on `Post_Canteen_Bug_Fixes`)

The 4 June canteen recordings revealed two issues with the original magnitude-only `*_sd` bundle: (1) impossible EmotiBit beat-detector values (IBI 80 ms / 6,160 ms, HR > 220 BPM) flowed untouched into the SDs, producing a ~270 ms `ibi_sd` floor; (2) magnitude SDs don't tell the operator whether a wearer is rising above or settling below their own normal. The replacement (live on this branch, mirroring the IMX '26 adult paper rolling-median + z-score and the IMEX infant paper whole-session z-score):

- **Per-device running mean/SD (Welford)** — numerically stable; converges to the true session mean within ~60–90 s and barely moves thereafter, so sustained elevations stay visibly elevated (a short EMA would habituate them away).
- **Calibration gate** — a short ~2 s warm-up (`CALIBRATION_SECONDS`, matching the minimum Welford sample floor) is marked `calibrating` in the payload; the GUI shows `calibrating Ns` and omits the per-trace readouts during it. Thereafter signed $z_t = (x_t - \mu_n)/\sigma_n$ for HR, EDA, IBI, temperature ROC and SCR frequency is published alongside the existing magnitude fields. The baseline is an **expanding window**: z-scores publish as soon as the floor is met and keep refining as Welford accumulates, capped at `BASELINE_CAP_SECONDS = 60` after which the mean / SD freeze so sustained elevations stay visible.
- **Wearer-swap detection (auto)** — a sustained simultaneous off-wrist ≥ `SWAP_PINNED_SECONDS = 2 s` forces a fresh baseline the moment the sensor is re-fitted, so a new wearer is never scored against the previous wearer's mean / SD. Biased to miss rather than false-trigger.
- **Off-wrist event (≥ 2 s)** — separate, much shorter threshold. While off-wrist the publisher drops `*_z` keys from the payload entirely (faking a baseline-relative value would mislead the operator into reading "calm" when the device is on a table) and sets `off_wrist: true` so the GUI can render an `OFF-WRIST` watermark. Two detection paths cover both common failure modes:
  - **Low-magnitude rule** — EDA mean < 0.10 µS AND PPGGreen mean < 1500 counts over the last second of raw samples (well-behaved units).
  - **Frozen-channel rule** — EDA std < 0.005 µS AND PPGGreen std < 5 counts over the same window (catches units whose ADS1114 EDA front-end rails at a high pinned value when removed and whose PPG latches on a fabric reflection — confirmed on `MD-V5-0000448` 9 June, where EDA was pinned at 2.595 µS identical to 5 dp for 20+ consecutive samples).
- **Skin-temp sensor preference** — if the wearable streams Thermopile (MLX90632, medical-grade skin temp; MD-V5 hardware) it's preferred over the legacy `Temperature1` (MAX30101 die temp, reads ~4–6 °C below skin). Plausibility gates are now per-sensor: 24–42 °C for Thermopile, 22–36 °C for die temp. Original sensor choice is logged once at startup per device.
- **Loose-strap handling** — partial channel dropout flags the affected channel as `low-quality` in the publish bundle; does **not** trigger a reset.

**Required upstream filtering** (precondition; rejected samples are excluded from Welford updates):
- HR ∈ [40, 200] BPM, IBI ∈ [300, 1500] ms, temperature per-sensor gates as above.
- Median pre-filter on per-beat IBI to suppress EmotiBit sample-and-hold artefacts.
- Hampel filter on EDA for spike rejection.

Backwards-compat: the original `*_sd` magnitude fields stay in the payload so the Unity subscriber and any legacy consumer continue to work unchanged. The signed `*_z` fields and the `off_wrist` / `calibrating` booleans are additive. The Valence/Arousal RF path is unchanged.

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
| `No real cameras detected (… virtual cameras skipped)` even though a real webcam is attached | The camera scanner now warms the sensor up before its live-vs-virtual test, so this should be resolved. If a slow-starting camera is still mis-flagged, bypass the filter by passing `--camera 0` (or its index) directly. |
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

## ⚖️ EU AI Act & Data-Protection Compliance

> **Not legal advice.** This section is an engineering-informed summary to guide
> compliance work, not a legal determination. Verify with your DPO / legal
> counsel and the official
> [EU AI Act Compliance Checker](https://ai-act-service-desk.ec.europa.eu/en/eu-ai-act-compliance-checker).

This system is **in scope** of Regulation (EU) 2024/1689 (the **AI Act**) and the
GDPR, because it combines three heavily regulated capabilities:

- **Emotion recognition** — engagement/affect inferred from face + pose video (`live_multiperson_binary_v2.py`).
- **Biometric identification** — face matching / enrolment (`face_identifier.py`, VGGFace2 embeddings).
- **Physiological inference** — EmotiBit EDA/HR → valence/arousal (`multiemotibit_UDP_SD_RFv2.py`).

Face embeddings and physiological readings are **special-category personal data**.

### Key legal touchpoints

| Ref | Obligation | Relevance here |
|-----|-----------|----------------|
| **Art. 5(1)(f)** | Emotion recognition is **prohibited in workplace & education** (narrow medical/safety exceptions), in force since **2 Feb 2025** | ✅ OK for live-events/audience use; ⛔ **must not** be redeployed in classrooms, staff or training settings |
| **Art. 50** | Deployers of emotion-recognition / biometric-categorisation systems must **inform the people exposed** | Requires clear audience notice/signage + information notice |
| **Annex III** | Biometric & some emotion-recognition uses may be **high-risk** → risk mgmt, data governance, logging, human oversight, accuracy docs, registration | Confirm final risk class via the checker |
| **GDPR Art. 9 / 35** | Special-category data needs an explicit lawful basis + a **DPIA** | Face embeddings + physio signals |
| **Timeline** | Prohibitions: Feb 2025 · Transparency (Art. 50): Aug 2026 · High-risk: Aug 2026–2027 | Plan accordingly |

### Compliance checklist

Tick items as they are completed and evidenced. Most are **team/legal actions**,
not code changes.

**Scope & classification**
- [ ] Run the official [EU AI Act Compliance Checker](https://ai-act-service-desk.ec.europa.eu/en/eu-ai-act-compliance-checker) and archive the result
- [ ] Record our **role(s)**: provider (building) and/or deployer (operating)
- [ ] Document the **intended purpose** and an explicit statement that it is **not** for workplace/education (Art. 5(1)(f))
- [ ] Confirm final **risk classification** (prohibited / high-risk / limited-risk + transparency)

**Transparency (Art. 50)**
- [ ] Audience **signage / notice** at the venue before capture
- [ ] Written **information notice** (what is captured, why, retention, contact, rights)
- [ ] On-device / on-screen indication that emotion & biometric processing is active

**Data protection (GDPR)**
- [ ] Complete a **DPIA** covering biometric + physiological processing
- [ ] Define **lawful basis** (explicit consent for special-category data where required)
- [ ] Set and enforce **retention limits** for face embeddings, physio CSVs and session recordings
- [ ] **Data minimisation** review; document who can access stored data and how it's secured
- [ ] Data Processing Agreements with any third parties / cloud

**If high-risk (Annex III)**
- [ ] Risk-management system and technical documentation (Annex IV)
- [ ] Data governance / dataset documentation for the trained models
- [ ] Event **logging & traceability** of operation
- [ ] Defined **human oversight** measures
- [ ] **Accuracy, robustness & bias** evaluation documented
- [ ] Registration in the EU database (where applicable)

### Privacy-supporting measures already in the design

These are engineering facts about the current build — helpful evidence, not proof of full compliance:

- **Registered-people-only mode** — only enrolled (consented) participants are scored, drawn and logged; bystanders are tracked by YOLO solely to enable enrolment and are **not** scored, displayed or aggregated.
- **Recyclable display IDs** (`P1`, `P2`, …) are shown on screen instead of raw identifiers.
- **Off-wrist / stale-data handling** — physiological z-scores are dropped (not faked) when sensor contact is lost, avoiding misleading inferences.
- **Local-first data flow** — Redis and all processing run on `localhost`; nothing is sent off-device by default.

---

## 🐍 Python Version

**Tested with Python 3.10, 3.11, 3.12** on Windows 11.
