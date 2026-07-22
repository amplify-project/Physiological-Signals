# Real-Time Audio Detection with CSV + WAV Recording

> **Location note:** this is the project-proper copy of the audio detection component
> (ported from `Handover_JulySession/`). Run it from the repo root with
> `python audio/audio_monitor_csv.py`; the `4_START_AUDIO_REALTIME.bat` launcher
> referenced below lives in `Handover_JulySession/`. Only the models required at
> runtime are in `audio/models/` — the unused Essentia `discogs-effnet-*.pb` models
> (Windows-incompatible, Linux-only future work) remain in `Handover_JulySession/models/`.

**Status: LIVE DETECTION READY** 🎵✅

## Overview

Real-time music and singing detection with CSV output and WAV recording for concert engagement AR system.

**Detection Capabilities:**
- **Music presence** (YamNet-based, 0.96s window, 0.48s hop)
- **Singing detection** (MLP on YamNet embeddings)
- **State tracking**: PAUSE | MUSIC | SINGING (priority: MUSIC > SINGING > PAUSE)

**Output:**
- **CSV file**: `emotibit_recordings/audio_YYYY-MM-DD_HH-MM-SS.csv` (detection log with human-readable timestamps)
- **WAV file**: `emotibit_recordings/audio_YYYY-MM-DD_HH-MM-SS.wav` (full audio recording)
- Saved in same directory as EmotiBit physiological data for easy matching

---

## Quick Start

### 1. Install Audio Detection Dependencies

Already included in `requirements.txt`. If needed manually:

```powershell
pip install sounddevice soundfile onnxruntime
```

### 2. Run Real-Time Audio Detection

```powershell
.\4_START_AUDIO_REALTIME.bat
```

**Prompts:**
1. Select audio input device ID (or press Enter for system default)
2. Detection starts immediately

**Console Output (Quiet Mode):**
```
[PAUSE    ] music=0.123 singing=0.045
[MUSIC    ] music=0.876 singing=0.234  ** MUSIC **
[MUSIC    ] music=0.923 singing=0.456
...
[Quiet mode active - showing state changes only. WAV auto-saving every 5s.]
[SINGING  ] music=0.234 singing=0.612  ** SINGING **
```

Shows first 10 detections (to verify it's working), then only state changes.

**Stop:** Ctrl+C OR just close the window (files save automatically!)

**Safe for long recordings:** WAV file auto-saves every 5 seconds — no data loss even if you forget to press Ctrl+C or just close the window!

### 3. Output Files

Both files saved with same timestamp for easy matching:

- **`audio_2026-07-21_14-30-15.csv`** — Detection log
- **`audio_2026-07-21_14-30-15.wav`** — Full audio recording (16kHz, 16-bit by default)

Located in `emotibit_recordings/` alongside:
- `MD-V5-0000334_2026-07-21_14-30-12.csv` (EmotiBit device data)
- `MD-V5-0000448_2026-07-21_14-30-12.csv`
- etc.

**File sizes (default settings):**
- 1-minute recording: ~1.9 MB (WAV) + ~5 KB (CSV)
- 2-hour recording: ~230 MB (WAV) + ~600 KB (CSV)

**Need higher quality?** Use `--wav-rate 48000` (files will be 3× larger)

**Auto-save:** WAV file writes continuously every 5 seconds. CSV writes + flushes every detection. Both files are safe even if program crashes or you forget to stop!

---

## CSV Output Structure

### Audio Detection CSV Format

**Filename:** `audio_YYYY-MM-DD_HH-MM-SS.csv`

**Columns:**
1. `unix_time` — Unix timestamp (seconds since epoch, with decimals)
2. `iso_time` — Human-readable timestamp (e.g., "2026-07-21T14:30:15.123+01:00")
3. `elapsed_sec` — Seconds since recording started
4. `music_score` — Music detection confidence (0.0-1.0)
5. `music_detected` — Binary: 1=music present, 0=absent
6. `singing_score` — Singing detection confidence (0.0-1.0)
7. `singing_detected` — Binary: 1=singing present, 0=absent
8. `state` — Current state (PAUSE | MUSIC | SINGING)

**Example rows:**
```csv
unix_time,iso_time,elapsed_sec,music_score,music_detected,singing_score,singing_detected,state
1721567415.123,2026-07-21T14:30:15.123+01:00,0.96,0.1234,0,0.0456,0,PAUSE
1721567415.603,2026-07-21T14:30:15.603+01:00,1.44,0.8765,1,0.2341,0,MUSIC
1721567416.083,2026-07-21T14:30:16.083+01:00,1.92,0.9234,1,0.6789,1,SINGING
```

**Update Frequency:** Every ~0.48 seconds (hop size)

### Audio WAV File

**Filename:** `audio_YYYY-MM-DD_HH-MM-SS.wav`

**Format:**
- Sample rate: 16,000 Hz (default, smaller files) or 48,000 Hz (high quality)
- Channels: Mono (1 channel)
- Bit depth: 16-bit PCM (default) or 32-bit float (high quality)
- Duration: Full recording length

**File Size:**
- **Default (16kHz, 16-bit)**: ~1.9 MB/minute → **~230 MB for 2 hours**
- High quality (48kHz, 16-bit): ~5.8 MB/minute → ~690 MB for 2 hours
- Maximum (48kHz, 32-bit): ~11.5 MB/minute → ~1.4 GB for 2 hours

**Change quality:**
```powershell
# Default: small files (16kHz, 16-bit) - recommended for long recordings
python audio_monitor_csv.py

# High quality: full sample rate (48kHz, 16-bit)
python audio_monitor_csv.py --wav-rate 48000

# Maximum quality: full rate + 32-bit (large files)
python audio_monitor_csv.py --wav-rate 48000 --wav-format FLOAT
```

**Usage:** Can be opened in any audio software (Audacity, Adobe Audition, etc.) for review/analysis

**Note:** Detection always uses 16kHz internally, so the default 16kHz recording preserves all information used by the detection models!

---

## Configuration

### Detection Thresholds

Edit [audio_config.py](audio_config.py):

```python
# Detection thresholds
YAMNET_THRESHOLD = 0.2      # Music detection (lower = more sensitive)
SINGING_THRESHOLD = 0.5     # Singing detection

# Detection windows
YAMNET_WINDOW_SEC = 0.96    # Analysis window length
YAMNET_HOP_SEC = 0.48       # Time between detections

# Stability (prevent flickering)
STABILITY_COUNT = 3         # Consecutive detections needed to change state
```

### Command Line Options

Full control via command line:

```powershell
# Specify audio device
python audio_monitor_csv.py --device 2

# Verbose mode (show all detections, not just state changes)
python audio_monitor_csv.py --device 2  # Add --no-quiet or remove --quiet flag

# Adjust thresholds on the fly
python audio_monitor_csv.py --yamnet-threshold 0.3 --singing-threshold 0.6

# Change output directory
python audio_monitor_csv.py --output-dir my_recordings

# High quality audio (48kHz, larger files)
python audio_monitor_csv.py --wav-rate 48000

# Maximum quality (48kHz, 32-bit float - very large files)
python audio_monitor_csv.py --wav-rate 48000 --wav-format FLOAT

# List available audio devices
python audio_monitor_csv.py --list-devices
```

**Note:** The batch file uses `--quiet` by default. To see all detections, run Python directly without the flag.

---

## State Machine Logic

**Priority:** MUSIC > SINGING > PAUSE

**Stability:** Prevents rapid flickering between states
- **Onset:** Requires `STABILITY_COUNT` consecutive detections (default: 3)
- **Offset:** 
  - Music: same as onset (3 frames)
  - Singing: 2× onset (6 frames) for smoother transitions

**Example State Transitions:**

```
[PAUSE] → 3× music detected → [MUSIC]
[MUSIC] → 3× singing detected (music still present) → [MUSIC] (priority)
[MUSIC] → 3× no music → [PAUSE]
[PAUSE] → 3× singing detected → [SINGING]
[SINGING] → 6× no singing → [PAUSE] (longer offset)
```

---

## Models & Files

**Required Files:** (already copied to `models/`)
- `yamnet_model.onnx` — Google YamNet music detection (16kHz audio)
- `sing_detection_head.pt` — Singing classifier (PyTorch MLP)
- `yamnet_class_map.csv` — Class labels (music = class 132)

**Total Size:** ~16MB

**Models NOT needed for real-time detection:**
- `discogs-effnet-*.pb` — Genre/mood/instrument (Essentia, Windows incompatible)
- Those are only used in full pipeline on Linux

---

## Troubleshooting

### "No module named 'audio_config'"

Run from the repo root (or the `audio/` folder):
```powershell
cd D:\Wspc\Python\Amplify\Physiological-Signals
python audio\audio_monitor_csv.py --list-devices
```

### "ONNX model not found"

Check that `models/` folder exists in the same directory as `audio_monitor_csv.py`.

### No audio input / "Invalid device ID"

List devices first:
```powershell
python audio_monitor_csv.py --list-devices
```

Then specify correct ID:
```powershell
python audio_monitor_csv.py --device 2
```

### "sounddevice not installed"

```powershell
pip install sounddevice soundfile
```

### CSV file is empty

Make sure you wait at least 1 second before stopping (Ctrl+C). First detection takes ~0.96s (window size).

### WAV file is incomplete or corrupted

This should not happen anymore! The program now properly handles:
- Ctrl+C (KeyboardInterrupt)
- Window close (Windows Console Control Handler for CTRL_CLOSE_EVENT)
- Ctrl+Break (SIGBREAK)
- Emergency exit (atexit handlers)

All methods properly finalize the WAV file. If you still have issues, check that you waited at least 5 seconds before closing (first auto-save interval).

### Program crashes - will I lose my data?

No! Both files are auto-saved:
- CSV: Flushed after every detection (~0.5s)
- WAV: Flushed every 5 seconds

You'll only lose the last 5 seconds of audio at most. The WAV file is properly closed even on crash/window close.

---

## July Session Usage

For the July concert trial:

1. **Pre-test at lab:**
   ```powershell
   # Quick test with any music playing
   .\4_START_AUDIO_REALTIME.bat
   # Play music from phone/computer near mic
   # Verify state changes: PAUSE → MUSIC → SINGING (if vocals)
   # Stop with Ctrl+C, check emotibit_recordings/ for CSV + WAV
   ```

2. **At concert venue:**
   ```powershell
   # Start EmotiBit first
   .\2_START_EMOTIBIT.bat
   
   # Then start audio detection
   .\4_START_AUDIO_REALTIME.bat
   # Select microphone device
   
   # Let run throughout concert (auto-saves continuously)
   # Safe to run for hours - files saved every 5 seconds
   # Stop with Ctrl+C or just close window - both save properly!
   ```

3. **Post-analysis:**
   - Audio data: `emotibit_recordings/audio_YYYY-MM-DD_HH-MM-SS.csv`
   - Audio recording: `emotibit_recordings/audio_YYYY-MM-DD_HH-MM-SS.wav`
   - Physio data: `emotibit_recordings/MD-V5-XXXXXXX_YYYY-MM-DD_HH-MM-SS.csv` (one per device)
   
   **Match by timestamp:** All files have timestamps in filename for easy alignment!

---

## Performance Notes

**CPU Usage:** ~5-10% (single core) for detection at 0.48s hop
**Latency:** ~0.5 seconds (analysis window + model inference)
**Memory:** ~500MB (models + 5s audio buffer) — constant memory usage even for long recordings!
**Disk Space (default 16kHz, 16-bit):** 
- WAV: ~1.9 MB per minute → **~230 MB for 2-hour session**
- CSV: ~5 KB per minute → ~600 KB for 2 hours

**For longer recordings:** Default settings are optimized for file size. Use `--wav-rate 48000` only if you need the full audio quality.

**Data Safety:** Audio writes every 5s, CSV writes every detection. Windows console control handlers ensure proper cleanup even when window is closed.

**Tested on:** Windows 11, Python 3.12, Intel i7-10th gen

---

## Post-Analysis Workflow

After collecting data, you'll have multiple files in `emotibit_recordings/`:

1. **Audio detection:** `audio_2026-07-21_20-30-15.csv`
2. **Audio recording:** `audio_2026-07-21_20-30-15.wav`
3. **Physio device 1:** `MD-V5-0000334_2026-07-21_20-30-12.csv`
4. **Physio device 2:** `MD-V5-0000448_2026-07-21_20-30-12.csv`
5. **Physio device 3:** `MD-V5-0001019_2026-07-21_20-30-12.csv`

**Analysis Strategy:**

1. **Load CSV files** into Python/R/Excel
2. **Align by timestamp:** Use `unix_time` or `iso_time` columns
3. **Merge datasets:** Join audio + physio data by nearest timestamp
4. **Analyze correlations:**
   - Music intensity vs. heart rate
   - Singing moments vs. EDA arousal
   - State transitions vs. physiological responses
5. **Listen to WAV:** Review audio at key moments identified in analysis

**Python Example:**
```python
import pandas as pd

# Load data
audio = pd.read_csv('emotibit_recordings/audio_2026-07-21_20-30-15.csv')
physio1 = pd.read_csv('emotibit_recordings/MD-V5-0000334_2026-07-21_20-30-12.csv')

# Convert to datetime
audio['datetime'] = pd.to_datetime(audio['iso_time'])
physio1['datetime'] = pd.to_datetime(physio1['LocalTimestamp'])

# Merge by nearest timestamp
merged = pd.merge_asof(audio.sort_values('datetime'), 
                       physio1.sort_values('datetime'),
                       on='datetime', direction='nearest')

# Analyze
print(merged[['state', 'HeartRate', 'EDA']].head())
```

---

## Next Steps / Future Enhancements

- [ ] Add BPM (tempo) estimation (requires Essentia, Linux only currently)
- [ ] Add musical key detection
- [ ] Add genre classification
- [ ] Multi-microphone support (spatial audio)
- [ ] ASIO driver support for professional audio interfaces
- [ ] Real-time synchronization with Redis Streams (if needed for live dashboards)

For now, **music + singing detection with CSV + WAV recording is fully operational** for the July session! 🎵✅
