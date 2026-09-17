# Concert Engagement AR System — Full Handover

> **V3 — September 2026 (`Handover_PostSeptSession`)**
> Contains the complete real-time engagement pipeline: computer vision inference, physiological signal processing (EmotiBit), the abstract particle-mesh visualisation (mirroring the AR glasses), audio monitoring, Redis pub/sub streaming, and the Unity C# subscriber for AR glasses.

This is a **lightweight bundle** built for email/drive transfer: no virtual environment, no installed packages, and **no model weights**. Run **`SETUP.bat`** once — it creates the venv, installs every pinned dependency, and downloads all publicly available weights. The **custom AMPLIFY model weights are not included** and must be requested from the project team (SETUP.bat prints exactly which files are missing and where to put them — see *Model weights* below).

---

## 📦 Contents

```
Handover_PostSeptSession/
├── SETUP.bat                        ← Run this ONCE on first install
├── 1_START_REDIS.bat                ← Step 1 (hub PC): start Redis + mDNS advertiser (one window each)
├── 1B_START_REDIS_ADVERTISER.bat    ← Optional: run the advertiser on its own (1_START_REDIS already launches it)
├── 2_START_EMOTIBIT.bat             ← Step 2: start EmotiBit publisher (auto-discovers the hub)
├── 3_START_ENGAGEMENT.bat           ← Step 3: start camera + AI inference (auto-discovers the hub)
├── 4_START_AUDIO_REALTIME.bat       ← Optional: real-time audio music/singing monitor
├── 5_START_LIGHT_INSTRUMENTS.bat    ← Optional: light-instruments Redis subscriber (+_REMOTE variant)
├── requirements.txt                 ← All Python deps, fully pinned
├── constraints.txt                  ← Locks torch cu124 + protobuf during install
│
├── live_multiperson_binary_v2.py    ← Main engagement inference + GUI (incl. abstract particle panel)
├── abstract_mapping_guide.md        ← How to read + operate the abstract particle-mesh panel
├── gaze_rules.py                    ← Gaze-first rules engine (focal point, performer detection)
├── botsort_gaze.yaml                ← Tuned BoT-SORT tracker config (stable IDs in crowds)
├── face_identifier.py               ← Face registration module
├── applog.py                        ← Shared crash/console logging helper
├── redis_service_advertiser.py      ← mDNS/DNS-SD advertiser so clients + glasses auto-find Redis (no IP typing)
├── redis_discovery.py               ← Shared client-side mDNS browse helper (used by the publisher + GUI)
├── multiemotibit_UDP_SD_RFv2.py     ← EmotiBit SD metrics → Valence/Arousal → Redis (multi-device)
├── audio_monitor_csv.py             ← Real-time audio classifier (YamNet + singing head)
├── audio_config.py                  ← Audio monitor settings
├── redis_subscriber_light_instruments*.py / simulate_light_instruments.py
├── UnityRedisSubscriber.cs          ← Drop-in Unity C# Redis subscriber
│
├── models/
│   └── yamnet_class_map.csv         ← YamNet class labels (weights NOT included — see below)
│
└── redis/
    ├── redis-server.exe             ← Bundled Redis server (Windows)
    ├── redis-cli.exe
    └── redis.windows.conf
```

### Model weights — what SETUP.bat gets vs. what you must request

| File | Used by | How you get it |
|---|---|---|
| `model/20180402-114759-vggface2.pt`, `model/face_detection_yunet_2023mar.onnx`, `model/face_recognition_sface_2021dec.onnx` | Face ID | **Downloaded by SETUP.bat** (public) |
| `yolo11n.pt` | Person detector | **Downloaded by SETUP.bat** (public; also auto-downloads on first run) |
| `model/best_model_family_concert.pth` | Engagement model (3_START_ENGAGEMENT) | **Request from project team** |
| `rf_valence_full_v2.pkl`, `rf_arousal_full_v2.pkl` | EmotiBit publisher (2_START_EMOTIBIT) | **Request from project team** |
| `models/yamnet_model.onnx`, `models/sing_detection_head.pt` | Audio monitor (4_START_AUDIO_REALTIME, optional) | **Request from project team** |

Copy the requested files to the paths shown, then re-run `SETUP.bat` — its final check confirms everything is in place.

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
4. Download the face models into `model/`: YuNet + SFace ONNX (long-range face ID) and the legacy FaceNet weights (`20180402-114759-vggface2.pt`, fallback only)
5. Download the YOLO person-detector weights (`yolo11n.pt`)
6. Check for the **custom AMPLIFY model weights** and list any that are missing (these are not downloadable — request them from the project team, copy them to the printed paths, and re-run SETUP.bat to verify)

> If you don't have an NVIDIA GPU, open `SETUP.bat` in Notepad first and swap the PyTorch install line (instructions are inside the file).

---

### Every session — just double-click the batch files, in order

**No terminal or typing needed** — from Windows Explorer, double-click the three numbered `.bat` files **in order**. Each opens its own window; leave all three open for the whole session, and wait for each to be ready before starting the next:

| # | Double-click | What happens |
|---|--------------|--------------|
| 1 | `1_START_REDIS.bat` | Starts the Redis message broker — keep this window open the whole session |
| 1b | `1B_START_REDIS_ADVERTISER.bat` | *(optional)* Advertises Redis on the Wi-Fi so the AR glasses auto-connect without typing an IP. Keep the window open; it waits for Redis and starts advertising once it answers |
| 2 | `2_START_EMOTIBIT.bat` | Discovers your EmotiBit(s) and publishes EDA/HR to Redis (press **ENTER** once your devices are listed) |
| 3 | `3_START_ENGAGEMENT.bat` | Starts the camera + AI engagement inference and opens the GUI |

**Order matters** — Redis (1) must be running before EmotiBit (2), and both before Engagement (3). The advertiser (1b) can be started any time after Redis — it waits for Redis on its own.

> **VS Code users:** The batch files are designed to be double-clicked from Explorer (each opens its own `cmd.exe` window). Inside VS Code's integrated terminal they all run in the same tab and block each other. Either double-click from Explorer, or open three separate VS Code terminal tabs and run the venv + script directly:
> ```
> .venv\Scripts\Activate.ps1; python multiemotibit_UDP_SD_RFv2.py # tab 2
> .venv\Scripts\Activate.ps1; python live_multiperson_binary_v2.py # tab 3
> ```

---

## � Running Redis on a separate PC (Redis as the network hub)

Redis is the comms hub and does **not** have to run on the same machine as the EmotiBit publisher and the engagement GUI. A common setup is a dedicated **broker PC** running Redis, with the physio + vision clients on another laptop on the **same LAN**. Thanks to **mDNS auto-discovery**, the clients find the hub automatically — no IP typing.

**On the broker PC** (the one that runs Redis):

1. Double-click **`1_START_REDIS.bat`**. It starts **two** things: Redis (main window) *and* the **mDNS advertiser** (its own "Redis Advertiser" window). The advertiser waits for Redis to answer PING, then announces this PC's LAN IP:6379 on the network (service `_amplify-redis._tcp.local.`) and re-announces if the address changes. The bundled `redis.windows.conf` binds all interfaces (`bind 0.0.0.0`) with `protected-mode no` so LAN clients are accepted. *(Safe only on a trusted, isolated venue/lab network — never expose port 6379 to the public internet. To lock it down, set `protected-mode yes` + a `requirepass` and pass the password from the clients.)*
2. Allow inbound **TCP 6379** through Windows Firewall on the broker PC (first connection usually prompts; otherwise add a rule).

**On this laptop** (EmotiBit publisher + engagement GUI) — just run the launchers, **no IP needed**:

```powershell
.\2_START_EMOTIBIT.bat        # discovers the hub over mDNS
.\3_START_ENGAGEMENT.bat      # discovers the hub over mDNS
```

Each client defaults to `--redis-host auto`: it browses mDNS for ~5 s and prints e.g. `Discovered Redis broker at 192.168.1.50:6379`. **Resolution order:** explicit `--redis-host` → env `REDIS_HOST` → mDNS discovery → `localhost`. The engagement GUI's built-in EmotiBit **subscriber** follows the same resolved host, so the physio sidebar and the per-participant/crowd publishes all track the broker.

**If mDNS is blocked** (some guest/enterprise Wi-Fi filter multicast or isolate clients) or you prefer a fixed target, pass the IP explicitly — this skips discovery:

```powershell
.\2_START_EMOTIBIT.bat --redis-host 192.168.1.50
.\3_START_ENGAGEMENT.bat --redis-host 192.168.1.50
```

Use `--no-discover` to force the localhost fallback without the ~5 s browse. Both `.bat` files forward extra arguments straight to their Python script. On startup the publisher prints the broker it connected to (`> Connected to Redis at HOST:PORT`), or `!! Could not reach Redis at …` with a hint if it can't.

> **Venue tip:** for maximum reliability give the hub PC a **DHCP reservation** (fixed IP) on the router and/or run on a dedicated router/AP — public/guest Wi-Fi with *AP client isolation* blocks all peer traffic (including Redis itself), which no discovery method can work around.

> `1B_START_REDIS_ADVERTISER.bat` still exists to run the advertiser **on its own** (e.g. if you start Redis a different way); `1_START_REDIS.bat` already launches it, so you normally don't need 1B. The advertiser also takes `--advertise-address <ip>` to force a specific interface.

---

## �🎮 Using the Engagement GUI

The main engagement window (`3_START_ENGAGEMENT.bat`) opens an OpenCV camera view with a sidebar:

- **Bounding boxes** are drawn for **registered (enrolled) people only**. The radio-selected focus person gets a **magenta** box; every other registered person is colour-coded by engagement score (red → orange → green). Bystanders are tracked internally (so they can be enrolled) but never drawn or scored. Labels show the name, score, and `f:NNpx` — the native face width in pixels from the last face-ID pass (the distance-test readout).
- **Engagement score** is a **fusion of two signals**: the gaze rules (base) and the action-transformer model (rescue/override) — see the *Gaze-first engagement scoring* section below.
- **Performers** are detected and used internally by the gaze engine (focal point, ray clipping) but get **no overlay box** by design.
- **Gaze rays** — thin lines from each audience member's head toward where they are looking, projected to the crowd's shared focal point. Performers never get a ray, and rays are clipped at performer boxes.
- The **HUD** shows the gaze engine state: `Gaze: PRE-SHOW` (no shared focal point yet — pure action-model scoring) or `Gaze: LIVE perf:N` (focal point locked, N performers detected). `SHIFT!` flags a synchronized crowd gaze shift (e.g. a door opening).
- A small **cyan dot** in the top-right of a bounding box means that person received fresh MediaPipe extraction this frame (the rest reuse their last score). When the system is throttling under crowd load the HUD shows e.g. `MP: 1/3 rr2`.
- **Sidebar** shows one row per enrolled person with their EmotiBit serial. The panel lays out in **1–3 columns** so up to ~12 wearers (e.g. 11+ infant/parent pairs) stay legible at once. Each row has:
  - **Header readouts** — three small colour-coded values right-aligned next to the serial: `HR ±z.zz`, `EDA ±z.zz`, `TEMP ±z.zz`. Values are per-wearer session z-scores (deviation from this wearer's own running mean, in their own SD units). When `|z| ≥ 2` the readout switches to bold with a faint coloured pill behind it.
  - **Unified physio panel** — single rolling spline plot on a fixed ±3 SD axis with reference lines at 0 and ±2 SD. Three traces share the panel: HR (green), EDA (yellow / cyan), TEMP (magenta). Each segment darkens and thickens past `|z| = 2` and the latest sample is marked with a small dot (plus a white halo when in the alert band). Window is the most recent **10 seconds** — older points scroll off the left.
  - **Calibration** — for the first ~2 s after a wearer is assigned the panel shows `calibrating Ns` while the Welford baseline reaches its minimum sample floor; no splines are drawn yet. Plotting then begins almost immediately and the baseline keeps **expanding** (converging over ~60 s, then frozen at a 60 s cap) so early z-scores refine as more samples arrive.
  - **OFF-WRIST** — when sensor contact is lost the panel dims and a faint red `OFF-WRIST` watermark appears within ~2 s. Splines gap out naturally rather than freezing on the last good value. Re-attach the device and the splines resume immediately; a sustained ≥ 2 s off-wrist gap forces a fresh baseline on re-fit (so a new wearer is never plotted against the previous wearer's mean / SD).
- **Radio buttons** on the sidebar select which detected person to treat as the "focus target".
- Press **`R`** to enroll a face: pick an EmotiBit serial from the picklist (arrow/number keys, **Enter** to confirm). If no EmotiBits are streaming, generic `Participant-1…5` names are offered instead. Then a guided **3-pose capture** starts: face **FRONT**, press **SPACE**; turn part-way **LEFT** (~45°, both eyes still visible), **SPACE**; part-way **RIGHT**, **SPACE**. **Esc** cancels at any point. Enroll at **~1–1.5 m** from the camera.
- Press **`C`** to clear all enrollments, **`F`** to cycle the focus target.
- Press **`V`** (dual-camera only) to switch which live feed the next **`R`** / **`SPACE`** enrollment captures from.
- Press **`q`** or **Esc** to quit gracefully.

### Abstract particle-mesh panel (AR glasses mirror)

Each wearer's sidebar row includes an **abstract particle-mesh visual** — a faithful port of what the RayNeo X3 Pro glasses render: a 5×5 grid of yellow-green particles whose shape, colour, sparks, pulse, and blue corner halos are driven by the same tonic EDA / temperature RoC / SCR frequency / heart-rate z-scores and engagement stream the glasses use. **Full interpretation and controls reference: [`abstract_mapping_guide.md`](abstract_mapping_guide.md).** Quick summary:

- **Resize** — drag the grip dots on a panel's **bottom border** (down = bigger). Drag the sidebar's **left border** to resize the whole sidebar.
- **View** — left-drag inside a panel to orbit, mouse-wheel or **`↑`/`↓`/`+`/`-`** to zoom (hovered panel, or all), right-click to reset the view.
- **Mock signals (demo/verification)** — lowercase pushes a channel to **+2.5 SD**, `Shift`+key to **−2.5 SD**, decaying back over ~2.5 s: **`T`** tonic EDA, **`W`** temperature wave, **`S`** SCR frequency, **`H`** heart rate, **`G`** engagement. **`E`** injects one instant SCR ring event (`Shift+E` negative), **`X`** clears all mocks. Mocks affect only the abstract visual — never the ±SD splines, Redis, or logged data.

### CLI flags for `live_multiperson_binary_v2.py`

The `3_START_ENGAGEMENT.bat` file invokes the script with sensible defaults; pass extra flags by editing the .bat or running the script directly inside the venv:

| Flag | Purpose |
|------|---------|
| `--camera N` | Pick camera index (0, 1, 2, …) |
| `--camera2 N` | Enable **DUAL-CAMERA mode** with a second camera index (2D only). See below. |
| `--select-camera` / `-s` | Interactive camera picker at startup (also offers a second camera for dual mode) |
| `--video PATH` | Run on a recorded video file instead of a live camera |
| `--save` | Save **everything**: engagement JSONL + keypoint NPZ chunks + raw video |
| `--save-engagement` | Save only `engagement_data.jsonl` (~370 MB / 30 min) |
| `--save-keypoints` | Save only the compressed MediaPipe keypoint NPZ chunks |
| `--save-dir PATH` | Override the default `data/sessions/<timestamp>/` location |
| `--no-face-id` | Skip the face-ID enrol step (faster startup, no per-person identity) |
| `--redis-host HOST` / `--redis-port N` | Point at a non-default Redis broker |
| `--device cuda` / `cpu` / `mps` | Force a specific compute device (default `auto`) |

### Dual-camera mode

A single machine can drive **two cameras** to cover a wide or U-shaped audience where one lens can't see everyone. Enable it either by passing `--camera2 N`, or by using the interactive picker (`-s`): after choosing the primary camera it offers a second 2D camera (press its number, or **Enter/S** to stay single-camera).

How it works:

- Each camera runs its **own** full pipeline (tracker, MediaPipe, action model, face-ID) in a **worker thread**, so per-camera track ids never collide and the two feeds process concurrently to recover frame rate.
- People are **merged by EmotiBit serial**, and a registered person seen on **both** feeds gets **one fused engagement score**, not two. Each camera scores that person independently (separate trackers/buffers), so the two raw scores differ; the merge combines every scored sighting of a serial into a single reliability-weighted value (a fuller temporal buffer and a more confident identity count for more) and smooths it frame-to-frame. That **one** score is shown identically on **both** overlays and published once — no more a value that flips between the two feeds. Un-enrolled bystanders are not de-duplicated but also don't contribute to the crowd score.
- The GUI shows both feeds **side-by-side** with the shared physio sidebar; a **single merged crowd average** is published to `engagement_score` and per-participant channels are published once per serial.
- **Resolution is negotiated against the shared USB bus, not capped at 1080p.** Two USB cameras on one controller share bandwidth, so the app opens both, measures their **concurrent** frame rate, and keeps the highest mode (4K → 1440p → 1080p → 720p) the pair can sustain — 4K stays on a fast USB3 bus. If the bus can't sustain a higher mode it steps down and prints a warning recommending you give each camera its **own USB controller** (see performance note below). Watch the startup line `🎥🎥 Dual capture: WxH @ ~Nfps/cam`.
- **2D only** — dual mode is not available for 360° cameras. Position the two cameras to cover each half of the audience with **minimal overlap**.
- Enroll each face on whichever camera sees it: press **`V`** to switch the active enrollment feed, then **`R`** / **`SPACE`** as usual. An enrollment done on one camera is recognised on the other (the face repo is shared).

> **Performance & USB bandwidth:** two pipelines on one PC roughly share one GPU + the CPU MediaPipe budget, so expect a lower per-camera frame rate than single-camera. For **capture resolution**, the limiter is usually the **USB bus**: two cameras on the same controller share it, and two 4K MJPG streams will starve each other. To run both at high resolution, give each camera its **own USB controller** — plug them into ports on **different physical buses** (e.g. one front-header + one rear port, a **USB PCIe/ExpressCard add-in card**, or one on a **USB-C / Thunderbolt** port). A *powered hub does not help* — it still shares one upstream bus. For post-hoc analysis resolution matters more than frame rate; a frame-skip option can be added later if needed.

### EmotiBit device discovery (Window 2)

When `2_START_EMOTIBIT.bat` starts (runs `multiemotibit_UDP_SD_RFv2.py`):
1. Power on your EmotiBit device(s).
2. Wait for them to appear in the console (up to 20 seconds).
3. **Press ENTER** once all expected devices are listed to begin streaming.
4. The publisher then runs **headless** — there is no separate matplotlib window. All physio plots are rendered inside the engagement GUI sidebar.

---

## 📡 Redis auto-discovery for the AR glasses (Window 1b)

The latest glasses app **scans the Wi-Fi for the Redis broker automatically** — performers no longer type an IP address; they just launch the app after Redis is running. This is powered by a small sidecar, `redis_service_advertiser.py`, launched from **`1B_START_REDIS_ADVERTISER.bat`**.

How it works:

- It advertises the broker over **mDNS/DNS-SD** as the service type `_amplify-redis._tcp.local.` on the machine's primary LAN address, which the glasses browse for.
- It only advertises **while Redis actually answers `PING`**, and withdraws the record the moment Redis stops — so the glasses never latch onto a dead broker. It also re-registers if the PC's IP changes mid-session.
- It is **independent** of Redis and of the data publishers: start it any time after `1_START_REDIS.bat`; it waits for Redis on its own and needs no ordering relative to EmotiBit/Engagement. Press **Ctrl+C** in its window to stop advertising.

Useful flags (edit the `.bat` or run the script inside the venv):

| Flag | Purpose |
|------|---------|
| `--redis-host HOST` / `--redis-port N` | Point the health check at a non-default Redis |
| `--advertise-address A.B.C.D` | Announce a specific IPv4 instead of auto-detecting the primary interface |
| `--service-name NAME` | Override the advertised DNS-SD instance label (default `Amplify Redis`) |
| `--verbose` | Debug logging |

> Requires the `zeroconf` package (added to `requirements.txt`; installed by `SETUP.bat`). Kept as a **separate window** for now so its logs are easy to watch while debugging; it can later be folded into `1_START_REDIS.bat` to save a click.

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

## 🎯 Long-range face identification — the 7 m audience-distance feature (Audience_Distance)

The project requires **face detection at 6–7 m**. The physics: a face is ~15 cm wide, so on a 1080p webcam with ~80° FOV a face at 7 m is only **~24 px** wide — far below what the old face stack could use. The pipeline was rebuilt around that constraint.

### Models used

| Stage | Model | Why |
|-------|-------|-----|
| Face **detection** | **YuNet** (`model/face_detection_yunet_2023mar.onnx`, OpenCV DNN) | WIDER-FACE-trained tiny-face detector; finds faces down to ~10 px and returns 5 landmarks for alignment. Replaces MTCNN (which loses faces below ~30 px). |
| Face **recognition** | **SFace** (`model/face_recognition_sface_2021dec.onnx`, OpenCV DNN) | Embeds the landmark-aligned 112×112 crop into a 128-d vector; cosine similarity against the enrolled gallery. Replaces FaceNet/InceptionResnetV1. |
| Fallback | MTCNN + FaceNet (facenet-pytorch) | Used automatically only if the ONNX models are missing (`SETUP.bat` step 6b downloads them). |

Both new models run as plain OpenCV DNN — no new Python dependencies, ~5 ms per face on CPU. The backend lives in `face_identifier.py` (`ONNXFaceIdentifier`, selected by `create_face_identifier()`).

### Changes made

1. **1080p30 capture negotiation** — Logitech cams default to 640×480, and Windows backends lock their format once streaming (DSHOW additionally caps uncompressed 1080p at ~1 fps). On startup the camera is reopened via **MSMF with MJPG requested before the first read**, scanning MSMF indices (DSHOW and MSMF number devices differently) and accepting only a verified ≥1920-wide live stream at ≥10 fps. Console prints `Capture boosted … 1920x1080@~30fps` or an explicit FAILED warning. Resolution is the distance budget: a 7 m face is ~24 px at 1080p but ~10 px at 480p.
2. **Head-crop matching with free zoom** — face ID runs on the upscaled top-third of each tracked person's box rather than the full frame (keeps neighbours out, multiplies effective resolution). At close range (<1 m the bbox fills the frame and the face sits mid-box) it automatically retries on the full person box.
3. **Multi-angle enrollment** — the 3-pose (front/left/right) SPACE-guided capture stores 6 gallery embeddings per person: each pose plus a **simulated low-resolution copy** (face shrunk to 24 px and back), so distant blurry probes match a same-domain gallery entry. Re-enroll whenever capture resolution changes — a stale gallery halves similarity.
4. **Distance-graded matching** — `identify_crop()` reports the native face width (`face_px`). Faces ≥ 40 px match at cosine ≥ 0.34; smaller faces must clear 0.45 (tiny faces blur-converge — measured impostor similarity 0.41 at 24 px). Far assignments additionally need 2 consistent frames.
5. **Sticky identity** — once a track is named, a face that shrinks, turns away, or dips below threshold is treated as *no evidence*, never eviction — tracking carries the identity out to 7 m and beyond. Eviction requires 6 consecutive **strong** contradictions (a clear ≥40 px face with similarity < 0.25 — plainly a different person), and re-attach after a genuine loss takes a few seconds once the face turns camera-ward again. One live track per name (highest similarity wins).
6. **MediaPipe at distance** — holistic's detectors fail on small distant crops (engagement collapsed to 0 beyond ~2–3 m). Person crops are now upscaled to ≥384 px height (max 4×) before landmark extraction; landmarks are crop-relative so downstream gaze/model geometry is unchanged.
7. **Diagnostics** — every run self-documents: the app log records the capture negotiation and every identity assign / evict / unmatched event with similarity and face size; the session JSONL gains per-person `face_px`. Box labels show `f:NNpx` live.

### Measured results (Logitech C930e, 1080p, kitchen tape test)

- Face **detection**: solid at ~7 m, worked even at 640×480 (~10 px faces).
- Fresh-gallery **recognition**: similarity 0.76–0.99 near, identity held unbroken through face-size dips to 88 px and a full walk-away/return; one genuine eviction (sim 0.15, face obscured) self-healed in 3.5 s.
- **Engagement keypoints**: buffer stays full at range after the upscale fix.
- Known remaining item: the **action model's score decays with distance/walking** (0.95 @ 0.7 m → 0.28 @ 1.8 m with a full buffer) — model bias, addressed by the planned retraining, not by this pipeline. The 10 s context window also means the score lags ~10 s after returning close.

### Camera guidance for the real venue

A 1080p/80° webcam yields ~24 px faces at 7 m: reliable *detection* + tracked identity, but fresh *recognition* only to ~4–5 m. For recognition at 7 m a face needs ~100 px → a **4K camera at ≤45° FOV or a PTZ with ≥5× optical zoom** (e.g. Logitech Rally / PTZ Pro 2 class). For a U-shaped audience, plan on a wide camera for the near arms plus a zoom camera for the far arm; enrollment at ~1 m + sticky tracking covers the gap in the meantime.

---

## �📡 Redis Channels — What Gets Published

All AR data flows over Redis pub/sub. By default the broker is auto-discovered over mDNS (falling back to `localhost:6379`); point every client at a specific broker with `--redis-host <IP>` (see *Running Redis on a separate PC* above). Channel names and payloads are identical wherever the broker runs.

### Engagement (Vision system)

| Channel | Payload | Description |
|---------|---------|-------------|
| `engagement_score` | `0.5342` (bare float string, crowd average 0–1) | Legacy **crowd-average** channel, published ~1 Hz. In the default **registered-people-only** mode it is the confidence-weighted average over enrolled (EmotiBit-wearing) participants only — bystanders are tracked cheaply by YOLO for enrolment but are not scored or aggregated. Unchanged, keep using it for the single crowd number. |
| `device:{serial}:engagement` | `{device, engagement, confirmed, confidence, source, timestamp}` | **Per-participant** engagement, one channel per EmotiBit, published ~1 Hz. `{serial}` is the same EmotiBit id the physio publisher uses (e.g. `MD-V5-0000334`), so this drops straight into the `device:*` namespace. See below. |

#### Per-participant engagement — `device:{serial}:engagement`

For each registered participant the vision system publishes their individual engagement on a channel keyed by their EmotiBit serial — the **same id** the physio publisher streams on `device:{serial}:physio_metrics`, so one participant's engagement and physiology share a key. Subscribe with a pattern:

```
PSUBSCRIBE device:*:engagement
```

Payload (JSON string, one message per participant per ~1 s tick):

```json
{
  "device": "MD-V5-0000334",
  "engagement": 0.72,
  "confirmed": true,
  "confidence": 0.61,
  "source": "face",
  "timestamp": 1756113600.0
}
```

| Field | Meaning |
|-------|---------|
| `device` | EmotiBit serial — matches `device:{serial}:physio_metrics` for the same person. |
| `engagement` | Engagement score `0.0–1.0` (same scale as the crowd average). |
| `confirmed` | `true` when the identity is a live or recently-seen **face match** (`source` = `face`/`coast`) — safe to attribute. `false` when it is an **appearance-inferred guess** after the person's track was lost and re-bound by body/clothing colour. Use this to drive a **green (confirmed) vs red (uncertain) indicator** in the AR overlay. |
| `confidence` | Numeric match strength `0.0–1.0` behind the id (face cosine similarity for `face`/`coast`, histogram correlation for an inferred guess). |
| `source` | `face` = live face match this second · `coast` = same track, face briefly unseen (still the same person) · `inferred` = re-bound by appearance after track churn (a guess). |
| `timestamp` | Unix epoch seconds when published. |

Only registered (enrolled) participants are published; unenrolled bystanders never appear. A participant whose track is lost and never re-bound simply stops publishing until they are seen again.

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

The bundled `UnityRedisSubscriber.cs` subscribes to the legacy `engagement_score` crowd float. To drive **per-participant** AR overlays, add a pattern subscription to `device:*:engagement` and parse the JSON payload — the `device` field tells you which participant and `confirmed` drives the green/red certainty indicator:

```csharp
// Per-participant engagement (one message per EmotiBit per ~1 s).
subscriber.Subscribe("device:*:engagement", (channel, message) =>
{
    // Runs on a Redis background thread — do NOT call Unity APIs here.
    // message is JSON: {"device","engagement","confirmed","confidence","source","timestamp"}
    var e = JsonUtility.FromJson<ParticipantEngagement>(message);
    // e.device      → EmotiBit serial (matches device:{serial}:physio_metrics)
    // e.engagement  → 0..1 score
    // e.confirmed   → true = positive face ID (green dot); false = inferred guess (red dot)
    // Buffer e by e.device and apply it on the Unity main thread in Update().
});

[System.Serializable]
public class ParticipantEngagement
{
    public string device;
    public float  engagement;
    public bool   confirmed;
    public float  confidence;
    public string source;
    public double timestamp;
}
```

Because the serial in `device:{serial}:engagement` is the **same** id as `device:{serial}:physio_metrics`, you can key one AR marker per participant and merge their engagement with their HR/EDA/valence/arousal from the physio channels.

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
| `FaceIdentifier init failed` | Re-run `SETUP.bat` so it downloads the face models (`model/face_detection_yunet_2023mar.onnx`, `model/face_recognition_sface_2021dec.onnx`); the legacy fallback also needs `model/20180402-114759-vggface2.pt` |
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
- **Biometric identification** — face matching / enrolment (`face_identifier.py`, SFace embeddings; legacy VGGFace2 fallback).
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
