#!/usr/bin/env python3
"""
Music/singing monitor with CSV + WAV recording (no Redis, no plotting).

Real-time detection (music + singing) saved to CSV alongside EmotiBit data.
Audio is also recorded to WAV file for later review.

Detection states: PAUSE | MUSIC | SINGING

Output:
  - CSV: emotibit_recordings/audio_YYYY-MM-DD_HH-MM-SS.csv
  - WAV: emotibit_recordings/audio_YYYY-MM-DD_HH-MM-SS.wav
"""

import argparse
import atexit
import csv
import os
import platform
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import soundfile as sf
import torch
import torch.nn as nn

# Windows console control handler
if platform.system() == 'Windows':
    import ctypes
    from ctypes import wintypes

from audio_config import (
    CAPTURE_RATE, TARGET_RATE,
    YAMNET_THRESHOLD, YAMNET_WINDOW_SEC, YAMNET_HOP_SEC, YAMNET_MUSIC_CLASS,
    SINGING_THRESHOLD,
    STABILITY_COUNT,
    MODEL_DIR,
)

YAMNET_MODEL = os.path.join(MODEL_DIR, "yamnet_model.onnx")
SINGING_MODEL = os.path.join(MODEL_DIR, "sing_detection_head.pt")

# Output directory (same as EmotiBit recordings)
OUTPUT_DIR = "emotibit_recordings"


def resample(audio, src_rate, dst_rate, target_len):
    if src_rate == dst_rate and len(audio) == target_len:
        return audio
    old_idx = np.linspace(0, len(audio) - 1, num=len(audio))
    new_idx = np.linspace(0, len(audio) - 1, num=target_len)
    return np.interp(new_idx, old_idx, audio).astype(np.float32)


def auto_select_device(probe_sec=1.0, min_rms=1e-4):
    """Pick the input device that actually hears something.

    Analogous to the camera auto-select (non-black frame test): record a short
    probe from every input device and choose the one with the highest RMS
    level. Devices that fail to open or are effectively silent are skipped.
    Returns a device ID, or None to fall back to the system default.

    Windows exposes each physical mic several times (MME / DirectSound /
    WASAPI / WDM-KS), so probing is restricted to the host API of the default
    input device, and virtual mapper entries are skipped — each real mic is
    probed once instead of 4x.
    """
    try:
        default_api = sd.query_devices(sd.default.device[0])['hostapi']
    except Exception:
        default_api = None
    _SKIP_NAMES = ('sound mapper', 'primary sound')

    candidates = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get('max_input_channels', 0) < 1:
            continue
        if default_api is not None and dev.get('hostapi') != default_api:
            continue
        if any(s in dev['name'].lower() for s in _SKIP_NAMES):
            continue
        try:
            rate = int(dev.get('default_samplerate') or CAPTURE_RATE)
            frames = int(rate * probe_sec)
            rec = sd.rec(frames, samplerate=rate, channels=1,
                         dtype='float32', device=idx)
            sd.wait()
            rms = float(np.sqrt(np.mean(rec[:, 0] ** 2)))
            candidates.append((rms, idx, dev['name']))
            print(f"  probe [{idx:2d}] {dev['name'][:40]:40s} rms={rms:.5f}")
        except Exception as e:
            print(f"  probe [{idx:2d}] {dev['name'][:40]:40s} skipped ({e})")
    live = [c for c in candidates if c[0] >= min_rms]
    if not live:
        print("  No device with signal detected - using system default.")
        return None
    rms, idx, name = max(live)
    print(f"  Selected [{idx}] {name} (rms={rms:.5f})")
    return idx


class MusicDetector:
    def __init__(self, model_path, music_class=YAMNET_MUSIC_CLASS):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.music_class = music_class

    def detect(self, audio_16k):
        outputs = self.session.run(None, {self.input_name: audio_16k})
        scores = outputs[0]
        embeddings = outputs[1]
        avg_scores = scores.mean(axis=0) if scores.ndim == 2 else scores.flatten()
        avg_embeddings = embeddings.mean(axis=0) if embeddings.ndim == 2 else embeddings.flatten()
        return float(avg_scores[self.music_class]), avg_embeddings


class SingingDetector:
    def __init__(self, model_path):
        raw = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dict = {k.removeprefix("net."): v for k, v in raw.items()}
        self.net = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 1),
        )
        self.net.load_state_dict(state_dict)
        self.net.eval()

    @torch.no_grad()
    def detect(self, yamnet_embeddings):
        x = torch.tensor(yamnet_embeddings).unsqueeze(0)
        return torch.sigmoid(self.net(x)).item()


class SignalStabilizer:
    def __init__(self, onset_count, offset_count=None):
        self.onset_count = onset_count
        self.offset_count = offset_count if offset_count is not None else onset_count
        self.stable = False
        self.counter = 0

    def update(self, detected):
        if detected != self.stable:
            self.counter += 1
            required = self.offset_count if self.stable else self.onset_count
            if self.counter >= required:
                self.stable = detected
                self.counter = 0
                return True
        else:
            self.counter = 0
        return False


STATE_PAUSE = "PAUSE"
STATE_MUSIC = "MUSIC"
STATE_SINGING = "SINGING"


class StateTracker:
    def __init__(self, stability_count):
        self.music = SignalStabilizer(stability_count)
        self.singing = SignalStabilizer(stability_count, offset_count=stability_count * 2)
        self.state = STATE_PAUSE

    def update(self, music_detected, singing_detected):
        self.music.update(music_detected)
        self.singing.update(singing_detected)

        self.prev_state = self.state
        if self.music.stable:
            self.state = STATE_MUSIC
        elif self.singing.stable:
            self.state = STATE_SINGING
        else:
            self.state = STATE_PAUSE

        return self.state != self.prev_state


def main():
    parser = argparse.ArgumentParser(
        description="Music/singing monitor with CSV + WAV recording",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = parser.add_argument_group("Detection")
    g.add_argument("--yamnet-model", default=YAMNET_MODEL, help="Path to YamNet ONNX model")
    g.add_argument("--yamnet-threshold", type=float, default=YAMNET_THRESHOLD, help="Music score threshold")
    g.add_argument("--yamnet-window", type=float, default=YAMNET_WINDOW_SEC, help="Detection window (s)")
    g.add_argument("--yamnet-hop", type=float, default=YAMNET_HOP_SEC, help="Detection hop (s)")
    g.add_argument("--singing-model", default=SINGING_MODEL, help="Path to singing detection head")
    g.add_argument("--singing-threshold", type=float, default=SINGING_THRESHOLD,
                    help="Singing head threshold")
    g.add_argument("--stability-count", type=int, default=STABILITY_COUNT,
                    help="Consecutive detections needed to switch state")

    g = parser.add_argument_group("Audio")
    g.add_argument("--device", type=int, default=None, help="Audio input device ID")
    g.add_argument("--auto-device", action="store_true",
                    help="Auto-select the input device: probe each one for ~1s and pick the loudest (like camera non-black-frame auto-select)")
    g.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    g.add_argument("--output-device", type=int, default=None,
                    help="Enable audio passthrough to this device ID (use --list-devices to find it; use headphones to avoid feedback)")

    g = parser.add_argument_group("Output")
    g.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for CSV and WAV files")
    g.add_argument("--wav-rate", type=int, default=16000, 
                    choices=[16000, 48000],
                    help="WAV file sample rate (16kHz=small files, 48kHz=full quality)")
    g.add_argument("--wav-format", default="PCM_16",
                    choices=["PCM_16", "PCM_24", "PCM_32", "FLOAT"],
                    help="WAV file format (PCM_16=16-bit for small files, FLOAT=32-bit high quality)")
    g.add_argument("--quiet", action="store_true",
                    help="Reduce console output (show only first 10 detections and state changes)")

    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.auto_device and args.device is None:
        print("Auto-selecting input device (make some noise near the mic)...")
        args.device = auto_select_device()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup output files
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = os.path.join(args.output_dir, f"audio_{timestamp}.csv")
    wav_path = os.path.join(args.output_dir, f"audio_{timestamp}.wav")

    # Derived sample counts
    yamnet_win = int(CAPTURE_RATE * args.yamnet_window)
    yamnet_hop = int(CAPTURE_RATE * args.yamnet_hop)
    yamnet_16k_len = int(TARGET_RATE * args.yamnet_window)

    # Load models
    print("Loading detection models...")
    music_detector = MusicDetector(args.yamnet_model)
    singing_detector = SingingDetector(args.singing_model)

    tracker = StateTracker(args.stability_count)

    lock = threading.Lock()
    yamnet_buf = np.array([], dtype=np.float32)
    
    # Audio recording buffer for resampling
    audio_buffer = []
    
    # Detection counter for quiet mode
    detection_count = 0

    def process_input(indata, status):
        nonlocal yamnet_buf
        if status:
            print(f"  [audio: {status}]")
        mono = indata[:, 0].copy()
        with lock:
            yamnet_buf = np.concatenate([yamnet_buf, mono])
            audio_buffer.append(mono)

    if args.output_device is not None:
        def audio_callback(indata, outdata, frames, time_info, status):
            process_input(indata, status)
            outdata[:] = indata
    else:
        def audio_callback(indata, frames, time_info, status):
            process_input(indata, status)

    # Open CSV file for writing
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'unix_time', 'iso_time', 'elapsed_sec',
        'music_score', 'music_detected', 
        'singing_score', 'singing_detected',
        'state'
    ])

    blocksize = int(CAPTURE_RATE * 0.1)
    device_info = sd.query_devices(args.device or sd.default.device[0], "input")

    print(f"\nMicrophone: {device_info['name']} (ID {device_info['index']})")
    print(f"Detection:  {args.yamnet_window}s window, {args.yamnet_hop}s hop")
    print(f"  Music:    threshold={args.yamnet_threshold}, onset={args.stability_count}, offset={args.stability_count}")
    print(f"  Singing:  threshold={args.singing_threshold}, onset={args.stability_count}, offset={args.stability_count * 2}")
    print(f"  Priority: MUSIC > SINGING > PAUSE")
    if args.output_device is not None:
        out_info = sd.query_devices(args.output_device, "output")
        print(f"Passthru:   on -> {out_info['name']} (ID {out_info['index']})")
    else:
        print(f"Passthru:   off")
    print(f"\nOutput:")
    print(f"  CSV: {csv_path}")
    print(f"  WAV: {wav_path} (saving continuously)")
    if args.quiet:
        print("\n[Quiet mode: showing first 10 detections + state changes only]")
    print("\nPress Ctrl+C to stop.\n")

    start_time = time.time()

    if args.output_device is not None:
        stream = sd.Stream(
            samplerate=CAPTURE_RATE,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=(args.device, args.output_device),
            callback=audio_callback,
        )
    else:
        stream = sd.InputStream(
            samplerate=CAPTURE_RATE,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=args.device,
            callback=audio_callback,
        )

    # Open WAV file for incremental writing
    wav_file = sf.SoundFile(wav_path, mode='w', samplerate=args.wav_rate, 
                            channels=1, subtype=args.wav_format)
    
    # Variables for auto-save
    last_wav_flush = time.time()
    WAV_FLUSH_INTERVAL = 5.0  # Flush every 5 seconds (was 10s)
    
    # Flag to track if cleanup already done
    cleanup_done = False
    
    # Cleanup function to ensure files are properly closed
    def cleanup():
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        
        try:
            # Flush remaining audio and close WAV file
            with lock:
                if audio_buffer:
                    audio_chunk = np.concatenate(audio_buffer)
                    audio_buffer.clear()
                    
                    if args.wav_rate != CAPTURE_RATE:
                        target_len = int(len(audio_chunk) * args.wav_rate / CAPTURE_RATE)
                        audio_chunk = resample(audio_chunk, CAPTURE_RATE, args.wav_rate, target_len)
                    
                    wav_file.write(audio_chunk)
            
            wav_file.close()
            csv_file.close()
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}", file=sys.stderr)
    
    # Register cleanup handlers
    atexit.register(cleanup)
    
    def signal_handler(signum, frame):
        elapsed = time.time() - start_time
        print(f"\n\nStopped after {elapsed:.1f}s (signal {signum})")
        cleanup()
        try:
            duration = wav_file.frames / args.wav_rate
            file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
            print(f"✓ CSV saved: {csv_path}")
            print(f"✓ WAV saved: {wav_path} ({duration:.1f}s, {file_size_mb:.1f}MB)")
        except:
            pass
        sys.exit(0)
    
    # Register signal handlers (Ctrl+C, Ctrl+Break, window close)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)
    
    # Windows-specific console control handler (catches window close)
    if platform.system() == 'Windows':
        def windows_console_handler(ctrl_type):
            """Handle Windows console events (including window close)"""
            if ctrl_type in (0, 1, 2, 5, 6):  # CTRL_C, CTRL_BREAK, CTRL_CLOSE, CTRL_LOGOFF, CTRL_SHUTDOWN
                print(f"\n\nWindows console event {ctrl_type} - cleaning up...")
                cleanup()
                try:
                    duration = wav_file.frames / args.wav_rate
                    file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
                    print(f"✓ CSV saved: {csv_path}")
                    print(f"✓ WAV saved: {wav_path} ({duration:.1f}s, {file_size_mb:.1f}MB)")
                except:
                    print(f"✓ Files saved to: {args.output_dir}")
                # Give time for output to be visible
                time.sleep(0.5)
                return 1  # Return 1 to indicate we handled it
            return 0
        
        # Register the handler (keep reference to prevent garbage collection)
        CTRL_HANDLER_ROUTINE = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
        global _windows_handler_ref  # Keep reference alive
        _windows_handler_ref = CTRL_HANDLER_ROUTINE(windows_console_handler)
        kernel32 = ctypes.windll.kernel32
        if not kernel32.SetConsoleCtrlHandler(_windows_handler_ref, 1):
            print("Warning: Could not register Windows console handler", file=sys.stderr)

    with stream:
        try:
            while True:
                # Auto-save WAV periodically
                current_time = time.time()
                if current_time - last_wav_flush >= WAV_FLUSH_INTERVAL:
                    with lock:
                        if audio_buffer:
                            # Resample and write buffered audio
                            audio_chunk = np.concatenate(audio_buffer)
                            audio_buffer.clear()
                            
                            if args.wav_rate != CAPTURE_RATE:
                                target_len = int(len(audio_chunk) * args.wav_rate / CAPTURE_RATE)
                                audio_chunk = resample(audio_chunk, CAPTURE_RATE, args.wav_rate, target_len)
                            
                            wav_file.write(audio_chunk)
                            wav_file.flush()
                    last_wav_flush = current_time
                
                with lock:
                    yamnet_avail = len(yamnet_buf)

                if yamnet_avail < yamnet_win:
                    time.sleep(0.01)
                    continue

                with lock:
                    yamnet_window = yamnet_buf[:yamnet_win].copy()
                    yamnet_buf = yamnet_buf[yamnet_hop:]

                audio_16k = resample(yamnet_window, CAPTURE_RATE, TARGET_RATE, yamnet_16k_len)

                music_score, yamnet_emb = music_detector.detect(audio_16k)
                rms = float(np.sqrt(np.mean(audio_16k ** 2)))
                if rms < 1e-4:
                    singing_score = 0.0
                else:
                    singing_score = singing_detector.detect(yamnet_emb)

                music_detected = music_score > args.yamnet_threshold
                singing_detected = singing_score > args.singing_threshold

                changed = tracker.update(music_detected, singing_detected)
                elapsed = time.time() - start_time
                unix_time = start_time + elapsed
                
                # Human-readable timestamp
                dt = datetime.fromtimestamp(unix_time, tz=timezone.utc).astimezone()
                iso_time = dt.isoformat()

                # Write to CSV
                csv_writer.writerow([
                    unix_time,
                    iso_time,
                    f"{elapsed:.2f}",
                    f"{music_score:.4f}",
                    1 if music_detected else 0,
                    f"{singing_score:.4f}",
                    1 if singing_detected else 0,
                    tracker.state
                ])
                csv_file.flush()

                # Console output (quiet mode: only first 10 + state changes)
                detection_count += 1
                show_output = not args.quiet or detection_count <= 10 or changed
                
                if show_output:
                    state_pad = tracker.state.ljust(9)
                    parts = [f"music={music_score:.3f}", f"singing={singing_score:.3f}"]

                    pending_parts = []
                    for name, sig, det in [
                        ("M", tracker.music, music_detected),
                        ("S", tracker.singing, singing_detected),
                    ]:
                        if det != sig.stable and sig.counter > 0:
                            required = sig.offset_count if sig.stable else sig.onset_count
                            arrow = "OFF" if sig.stable else "ON"
                            pending_parts.append(f"{name}→{arrow}:{sig.counter}/{required}")

                    pending = f" ({' '.join(pending_parts)})" if pending_parts else ""

                    print(f"[{state_pad}] {' '.join(parts)}{pending}", end="")

                    if changed:
                        print(f"  ** {tracker.state} **", end="")

                    print()
                
                # Show periodic status in quiet mode
                if args.quiet and detection_count == 11:
                    print("[Quiet mode active - showing state changes only. WAV auto-saving every 5s.]")

        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            print(f"\n\nStopped after {elapsed:.1f}s")
            
        finally:
            # Ensure cleanup happens
            if not cleanup_done:
                cleanup()
                try:
                    duration = wav_file.frames / args.wav_rate
                    file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
                    print(f"✓ CSV saved: {csv_path}")
                    print(f"✓ WAV saved: {wav_path} ({duration:.1f}s, {file_size_mb:.1f}MB)")
                except Exception as e:
                    print(f"Files saved to: {args.output_dir}")
                    print(f"Note: {e}")


if __name__ == "__main__":
    main()
