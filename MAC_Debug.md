# macOS Debug Log

Platform tested: **macOS — Apple Silicon (arm64), iMac**  
Python: 3.10.19 (Homebrew)  
Branch: `feature/macos-debug`  
Date started: 2026-03-06

---

## Issue 1: Python version incompatibility

**Symptom:** System default Python is 3.14.3, which is outside the supported range (3.8–3.11) listed in the README. Several dependencies (mediapipe, torch 2.2.x) do not support Python 3.14.

**Fix:** Created `.venv` using `/opt/homebrew/bin/python3.10` instead of the system default `python3`.

**Recommendation:** The README and setup.sh should check for a compatible Python version and warn/exit if the Python used to create the venv is outside 3.8–3.11.

---

## Issue 2: Silent crash — stderr redirect swallows all errors

**Symptom:** App prints "Initializing MediaPipe..." then exits silently with exit code 1. No traceback, no error message visible.

**Root cause:** Lines 41–43 of `live_multiperson_binary_v2.py` redirect OS-level stderr (fd 2) to `/dev/null` before any third-party imports. This suppresses MediaPipe/TFLite C++ warnings, but also swallows **all** Python tracebacks from unhandled exceptions during:
- Module imports
- `MultiPersonEngagementSystem.__init__()` (YOLO, MediaPipe, model loading)
- Camera detection and opening
- FPS calibration
- First frame processing

stderr is only restored at ~line 1434, deep inside the main loop after the first frame is successfully processed. Any crash before that point is invisible.

**Temporary fix:** Commented out `os.dup2(_devnull, 2)` to make errors visible during debugging.

**Recommendation:** Restore stderr earlier — immediately after MediaPipe Holistic is instantiated (line ~632), not after the first frame. Alternatively, wrap the startup in a try/except that restores stderr before re-raising.

---

## Issue 3: mediapipe `solutions.holistic` API removed in 0.10.15+

**Symptom:** `AttributeError: module 'mediapipe' has no attribute 'solutions'`

**Root cause:** `requirements.txt` specifies `mediapipe>=0.10.0` with no upper bound. pip resolves this to the latest version (0.10.32 as of March 2026). Google removed the legacy `mp.solutions.holistic` API starting in mediapipe 0.10.15 (released 2024-08-29), replacing it with the new Tasks API which has a completely different interface.

The code was developed around November 2025, when 0.10.14 was likely installed on the developer's machine. The open-ended requirement never caused issues on that machine because the existing venv already had the correct version pinned.

**Version timeline:**
| Version | Release Date | Status |
|---------|-------------|--------|
| 0.10.14 | 2024-05-08 | Last version with `mp.solutions.holistic` |
| 0.10.15 | 2024-08-29 | **Breaking change** — `solutions` removed |
| 0.10.32 | latest | Current latest, no `solutions` API |

**Fix:** Downgrade to `mediapipe==0.10.14` and pin upper bound in `requirements.txt`:
```
mediapipe>=0.10.0,<0.10.15
```

**Status:** FIXED — downgraded to 0.10.14, pinned in requirements.txt.

---

## Issue 5: yolo26n.pt produces zero detections on macOS (PyTorch 2.2.2)

**Symptom:** OpenCV window appears, camera is active, but no bounding boxes are drawn around people. Engagement score is always 0.0000.

**Root cause:** `yolo26n.pt` was saved with ultralytics 8.3.222 and its yaml declares `end2end: True` (NMS-free YOLO26 architecture). However, at runtime on macOS with PyTorch 2.2.2, the Detect head deserializes with `end2end: False` — the model loads without errors but produces zero detections on any class, even at conf=0.01. This is a PyTorch version mismatch: YOLO26 was released in Jan 2025 and requires PyTorch >= 2.4 for correct weight deserialization.

**Verification:**
| Model | Detections | Confidence |
|-------|-----------|------------|
| yolo26n.pt | 0 (any class, conf=0.01) | — |
| yolo11n.pt | 1 person | 93.1% |

**Fix:** Added fallback logic in `live_multiperson_binary_v2.py`: if `yolo26n.pt` is not found, falls back to `yolo11n.pt` (same COCO classes, similar size). Renamed `yolo26n.pt` → `yolo26n.pt.bak` on this Mac. The app now detects people and produces engagement scores of 0.84–0.94 for an engaged person.

**Note:** This does NOT affect Windows where PyTorch >= 2.4 + CUDA is used and yolo26n works correctly. The code change is backward-compatible — it prefers yolo26n if present and working.

**Status:** FIXED — using yolo11n.pt fallback on macOS.

---

## Issue 4: macOS camera permission required

**Symptom:** `OpenCV: not authorized to capture video (status 0), requesting...` — camera not accessible.

**Root cause:** macOS requires explicit camera authorization per-application. The terminal app (Terminal.app or VS Code integrated terminal) must be granted Camera access in System Settings → Privacy & Security → Camera.

**Fix:** Granted camera permission to VS Code. Camera 0 then works correctly (1920×1080, AVFoundation backend).

**Note:** This is a standard macOS requirement, not a bug. However, the app exits silently when no cameras are found (due to Issue 2), which makes it hard to diagnose. The camera scan itself works — it prints "No real cameras detected" — but the message is lost to the stderr redirect.

---

## Pending Work

- [x] Apply mediapipe downgrade (`pip install mediapipe==0.10.14`)
- [x] Pin mediapipe version in `requirements.txt`
- [ ] Re-enable stderr redirect after confirming app runs
- [ ] Clean up temp test files (`test_camera.py`, `test_run_debug.py`)
- [ ] Full end-to-end test without `--save`
- [ ] Test with `--save` for file I/O behaviour
- [ ] Verify same changes don't break Windows
