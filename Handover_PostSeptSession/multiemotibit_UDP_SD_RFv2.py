# ===============================================================
# -*- coding: utf-8 -*-
# REAL-TIME EMOTIBIT: FILTERED EDA + HR PLOT + HR PUBLISH TO REDIS
# Direct UDP - no EmotiBit Oscilloscope / LSL dependency
#
# Features:
# - Multi-device support via UDP discovery
# - Per-wearer baseline tracking (Welford online variance)
# - Runtime reconnect capability
# - Redis pub/sub for live GUI + Redis Streams for time-sync
# - Compatible with redis_sync_consumer.py for audio-physio alignment
# ===============================================================

import atexit
import argparse
import csv
import queue
import time, threading, json
from datetime import datetime
from collections import defaultdict
import os
import platform
import select
import signal
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import redis

# Shared async logger (writes to data/logs/emotibit/<ts>.log).
# In the handover folder applog.py sits beside this script.
import sys as _sys
_here = Path(__file__).resolve().parent
if str(_here) not in _sys.path:
    _sys.path.insert(0, str(_here))
from applog import setup_logging, install_excepthook  # noqa: E402
import logging as _logging
log = _logging.getLogger('emotibit')

# ----------------------------- SETTINGS -----------------------------
COLUMNS = ["EDA", "HeartRate"]  # For feature extraction (used for predictions)
WINDOW_SECONDS = 5
EMIT_RATE_HZ = 25
SAVE_ALL_SIGNALS = True  # Capture all available EmotiBit signals

# Get script directory for relative path resolution (cross-platform compatible)
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH_VAL = str(SCRIPT_DIR / "rf_valence_full_v2.pkl")
MODEL_PATH_ARO = str(SCRIPT_DIR / "rf_arousal_full_v2.pkl")

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

CHANNEL_PHYSIO = "device:{src}:physio_metrics"
CHANNEL_RAW_SENSORS = "device:{src}:raw_sensors"  # Accelerometer, gyroscope, magnetometer, temperature, battery, PPG
CHANNEL_VAL = "device:{src}:valence_cont"
CHANNEL_ARO = "device:{src}:arousal_cont"
CHANNEL_RESET = "device:{src}:reset_baseline"  # GUI publishes here to reset a wearer's baseline
CHANNEL_RECONNECT = "emotibit:reconnect"        # GUI/operator publishes here to re-run device discovery at runtime
CHANNEL_RECONNECT_STATUS = "emotibit:reconnect_status"  # publisher reports the outcome back to the GUI

# Redis Streams settings: cap stream length so memory doesn't grow unbounded
# over long sessions (approximate trimming is cheap). This enables time-based
# synchronization with audio_redis_publisher.py via redis_sync_consumer.py.
STREAM_MAXLEN = 20000

# ---- Per-wearer session-z baseline / artefact-rejection settings ----
# Plausibility gates: samples outside these ranges never enter RF / SD / Welford buffers.
PLAUSIBLE_HR_BPM   = (40.0, 200.0)
PLAUSIBLE_IBI_MS   = (300.0, 1500.0)
# Skin temperature gates depend on which sensor sourced the sample:
#   - TH (Thermopile / MLX90632, MD hardware only): true non-contact skin temp.
#     Wider window to tolerate brief excursions while putting on / adjusting the
#     wearable; values that drift far below 24 C indicate the sensor is pointing
#     at clothing/air rather than skin.
#   - T1 (MAX30101 die temperature): board temp -- runs ~4-6 C below skin.
#     Only a fallback when TH isn't streaming; values 22-32 C are normal at rest.
#   - T0 (legacy V1-V3 sensor): retained for completeness only.
PLAUSIBLE_SKIN_TEMP_C  = (24.0, 42.0)
PLAUSIBLE_DIE_TEMP_C   = (22.0, 36.0)
PLAUSIBLE_TEMP_C   = (30.0, 38.0)  # legacy alias, used only by manual-T0 path
PLAUSIBLE_EDA_US   = (0.01, 100.0)
PLAUSIBLE_SCR_FREQ = (0.0, 20.0)
# EmotiBit emits a ~9999-10000 sentinel/placeholder for SCRAmplitude before a
# real reading is computable (observed in the first few seconds of a new
# wearer's calibration, alongside the "eda_z missing post-calibration" state).
# Real values in this dataset run 0-25; gate it like every other channel so
# the placeholder never reaches the raw buffer as a numeric value.
PLAUSIBLE_SCR_AMPLITUDE = (0.0, 25.0)
IBI_MEDIAN_KERNEL  = 3        # median pre-filter on per-beat IBI
# Warm-up before z-scores are published. A z-score needs a mean AND a standard
# deviation, so the hard floor is BASELINE_MIN_SAMPLES (below) -- you cannot
# compute a meaningful z from a single reading. With the baseline updated once
# per second this is ~5 s, which is the smallest statistically valid warm-up.
# We deliberately do NOT impose any extra fixed wall-clock wait on top of that,
# so plotting begins almost immediately once a sensor is fitted (first fit or
# any subsequent re-fit after a swap reset). The baseline is an EXPANDING window:
# z-scores publish as soon as the floor is met and keep getting more accurate as
# Welford accumulates more samples (it converges over ~60-90 s, but plotting no
# longer waits for that).
CALIBRATION_SECONDS = 2       # minimal warm-up window (matches the sample floor)
BASELINE_MIN_SAMPLES = 2      # Welford samples needed before a z-score is valid
# Hard cap on how long the expanding baseline keeps accumulating. After this the
# mean/SD freeze and every later sample is scored against that fixed reference.
# Without it, a wearer who stays aroused for the whole (often <=90 s) wear slowly
# drags their own mean up, so it takes ever more arousal to register the same z --
# the elevation normalises itself away. Welford has converged by ~60 s anyway, so
# freezing then keeps sustained elevations visible at negligible accuracy cost.
BASELINE_CAP_SECONDS = 60
# EDA-only delayed baseline. Tonic EDA drifts upward for minutes after fitting
# while the skin-electrode interface hydrates, so a baseline taken in the first
# minute freezes a too-low mean with a tiny SD and eda_z saturates >3 forever.
# Skip the settling minute entirely, then accumulate for the following minute
# (eda_z publishes from ~62 s once BASELINE_MIN_SAMPLES is met, frozen at 120 s).
# Kept short for infant wearers; a swap/off-wrist reset restarts this schedule.
EDA_SETTLE_SECONDS       = 60
EDA_BASELINE_CAP_SECONDS = EDA_SETTLE_SECONDS + 60
Z_SOFT_THRESHOLD = 1.5        # amber GUI indicator (~13% fire rate)
Z_HARD_THRESHOLD = 2.0        # red GUI indicator + logged event (~5%, matches published methodology)
# Wearer-swap auto-detection (pinned-floor heuristic).
# EmotiBits keep streaming when removed from skin -- they emit pinned junk
# (EDA near zero, PPG near zero, HR/IBI sample-and-hold). UDP silence never
# happens, so we detect off-wrist by sustained collapse of the contact-driven
# raw signals instead.
SWAP_PINNED_SECONDS     = 2.0    # off-wrist must persist >= this to count as a swap -> fresh baseline on re-fit
OFF_WRIST_SECONDS       = 2.0    # short threshold: drop *_z keys from physio_metrics and signal GUI watermark
EDA_OFFWRIST_THRESHOLD  = 0.10   # µS — below this the wrist is empty
PPG_OFFWRIST_THRESHOLD  = 1500.0 # PPGGreen counts — below this the LED has no skin to reflect off
# Frozen-channel fallback: some units (e.g. MD-V5-0000448 on 9 June) don't
# collapse to floor when removed -- the ADS1114 rails at a high pinned value
# (~2.6 µS identical to 5 dp for many seconds) and PPGGreen latches on a
# nearby fabric reflection (~2600 counts). Real skin contact ALWAYS produces
# measurable jitter, so zero variance over the detection window is a clean
# tell-tale of disconnected electrodes regardless of magnitude.
EDA_FROZEN_STD_THRESHOLD = 0.005 # µS — on-wrist swings 0.03-0.10 µS/s, off-wrist literally 0.000
PPG_FROZEN_STD_THRESHOLD = 5.0   # counts — on-wrist sees ~15-30 count pulse jitter
SWAP_TEMP_STEP_C        = 1.0    # post-return skin-temp shift confirms different wearer (optional)
SWAP_DATA_FRESH_SECONDS = 5.0    # "channel is live" threshold

# ---- EmotiBit UDP protocol settings ----
EMOTIBIT_CONTROL_PORT = 3131
EMOTIBIT_DATA_PORT    = 3132
EMOTIBIT_TCP_PORT     = 3133
DISCOVERY_TIMEOUT_S   = 20
RECONNECT_TIMEOUT_S   = 8     # non-interactive re-discovery window for runtime reconnects
HEARTBEAT_INTERVAL_S  = 1

# EmotiBit type-tag ’ canonical stream name
TYPE_TAG_MAP = {
    "AX": "AccelerometerX",  "AY": "AccelerometerY",  "AZ": "AccelerometerZ",
    "GX": "GyroscopeX",      "GY": "GyroscopeY",      "GZ": "GyroscopeZ",
    "MX": "MagnetometerX",   "MY": "MagnetometerY",   "MZ": "MagnetometerZ",
    "EA": "EDA",              "EL": "EDL",              "ER": "EDR",
    "HR": "HeartRate",        "BI": "InterBeatInterval",
    "PG": "PPGGreen",        "PI": "PPGInfrared",     "PR": "PPGRed",
    "SA": "SCRAmplitude",    "SF": "SCRFrequency",    "SR": "SCRRiseTime",
    "T0": "Temperature0",    "T1": "Temperature1",
    "TH": "Thermopile",      "H0": "Humidity",
    "O2": "SpO2",
    "BV": "BatteryVoltage",  "B%": "BatteryPercent",
}

# ----------------------------- FILTERS -----------------------------
from scipy.signal import butter, filtfilt, medfilt

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, cutoff / nyq, btype="low")

def lowpass_filter(data, cutoff, fs, order=4):
    if len(data) < max(3*order, 20):
        return data
    b, a = butter_lowpass(cutoff, fs, order)
    try:
        return filtfilt(b, a, data)
    except ValueError:
        return data

def filter_eda(arr):
    arr = medfilt(arr, kernel_size=5)
    return lowpass_filter(arr, cutoff=1.0, fs=EMIT_RATE_HZ)

def filter_hr(arr):
    arr = medfilt(arr, kernel_size=5)
    return lowpass_filter(arr, cutoff=0.5, fs=EMIT_RATE_HZ)

# ----------------------------- REDIS STREAMS HELPER -----------------------------
def _safe_xadd(redis_client, stream_name, payload):
    """Persist a flat dict to a Redis Stream (redis-py requires every field
    value to be a str/int/float/bytes - None isn't allowed, bools/dicts/lists must be converted)."""
    fields = {}
    for k, v in payload.items():
        if v is None:
            continue
        # Redis xadd only accepts primitives - convert complex types
        if isinstance(v, bool):
            fields[k] = int(v)  # True → 1, False → 0
        elif isinstance(v, (dict, list)):
            fields[k] = json.dumps(v)  # Serialize complex types to JSON string
        elif isinstance(v, (str, int, float, bytes)):
            fields[k] = v
        else:
            # Fallback: convert unknown types to string
            fields[k] = str(v)
    try:
        redis_client.xadd(stream_name, fields, maxlen=STREAM_MAXLEN, approximate=True)
    except Exception as e:
        log.warning(f"xadd to {stream_name} failed: {e}")

# ----------------------------- BASELINE STATE -----------------------------
class WelfordState:
    """Numerically-stable running mean/variance (Welford's online algorithm).
    One instance per (device, channel). Anchors the per-wearer session z-score.
    """
    __slots__ = ('n', 'mean', 'm2')

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x):
        if x is None:
            return
        try:
            xv = float(x)
        except (TypeError, ValueError):
            return
        if not np.isfinite(xv):
            return
        self.n += 1
        delta = xv - self.mean
        self.mean += delta / self.n
        delta2 = xv - self.mean
        self.m2 += delta * delta2

    @property
    def sd(self):
        if self.n < 2:
            return 0.0
        return float(np.sqrt(self.m2 / (self.n - 1)))

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0


def _plausible(value, lo_hi):
    """True if value is finite and within (lo, hi)."""
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(v):
        return False
    lo, hi = lo_hi
    return lo <= v <= hi

# ----------------------------- FEATURE EXTRACTION -----------------------------
def extract_features_window(df):
    feats = {}
    feats['EDA_mean'] = df['EDA'].mean()
    feats['EDA_std']  = df['EDA'].std()
    feats['EDA_min']  = df['EDA'].min()
    feats['EDA_max']  = df['EDA'].max()
    feats['EDA_range'] = df['EDA'].max() - df['EDA'].min()
    feats['EDA_slope'] = (df['EDA'].iloc[-1] - df['EDA'].iloc[0]) / max(1, len(df))
    feats['EDA_peaks'] = (df['EDA'] > df['EDA'].mean() + df['EDA'].std()).sum()
    feats['EDA_std_slope'] = df['EDA'].diff().std()

    feats['HR_mean'] = df['HeartRate'].mean()
    feats['HR_std']  = df['HeartRate'].std()
    feats['HR_min']  = df['HeartRate'].min()
    feats['HR_max']  = df['HeartRate'].max()

    rr = 60 / df['HeartRate'].replace(0, np.nan)
    rr = rr.dropna()
    feats['HRV_SDNN'] = rr.std() if len(rr) > 1 else 0
    feats['HRV_RMSSD'] = np.sqrt(np.mean(np.diff(rr)**2)) if len(rr) > 1 else 0

    return feats

# ----------------------------- DATA LOGGER -----------------------------
class DataLogger:
    FLUSH_INTERVAL = 1.0   # seconds between automatic os.fsync() calls

    def __init__(self, source_id, output_dir="emotibit_recordings", signal_types=None):
        self.source_id = source_id
        self.output_dir = output_dir
        self.signal_types = signal_types or {}

        # Create output directory (cross-platform)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = output_path / f"emotibit_{source_id}_{timestamp}.csv"

        # Initialize headers: timestamp, device, all signal types, and computed metrics
        signal_columns = sorted(self.signal_types.keys())
        computed_metrics = [
            "valence", "arousal",
            "edl_sd", "eda_sd", "temperature_roc_sd", "scr_frequency_sd", "hr_sd", "ibi_sd",
            # Per-wearer session-z baseline columns (post-canteen physio update).
            "calibrating", "calibration_remaining_s", "baseline_n",
            "hr_z", "eda_z", "ibi_z", "temperature_roc_z", "scr_frequency_z",
            "hr_event_hard", "eda_event_hard", "ibi_event_hard",
            "quality_hr", "quality_eda", "quality_ibi",
            "quality_temperature_roc", "quality_scr_frequency", "quality_scr_amplitude",
            "swap_detected",
            "off_wrist",
            # Join key from the engagement PC (Redis 'engagement:session_id') —
            # makes cross-stream alignment deterministic instead of clock-based.
            "engagement_session_id",
            "processing_time_ms"
            # Note: end_to_end_latency_ms removed - unreliable without proper clock sync
        ]
        self.headers = ["timestamp", "device"] + signal_columns + computed_metrics
        self.lock = threading.Lock()

        # Persistent file handle - kept open for the lifetime of the session
        self._file = open(self.filename, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.headers)
        self._file.flush()
        self._last_flush = time.time()
        self._closed = False

        print(f"=Ê DataLogger initialized for {source_id}")
        print(f"   Signals to save: {', '.join(self.signal_types.keys())}")
        print(f"   File: {self.filename}")

    def log_signal_data(self, timestamp, device, signal_dict):
        """Log all signal data - writes directly to the persistent CSV handle."""
        with self.lock:
            if self._closed:
                return
            row = [timestamp, device]
            for col in self.headers[2:]:   # signal columns in sorted order
                row.append(signal_dict.get(col, ""))
            self._writer.writerow(row)

            # Periodic fsync - flush Python buffers to OS, then OS to disk
            now = time.time()
            if now - self._last_flush >= self.FLUSH_INTERVAL:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._last_flush = now

    def flush(self):
        """Force flush Python buffers + os.fsync to disk."""
        with self.lock:
            if self._closed:
                return
            self._file.flush()
            os.fsync(self._file.fileno())
            self._last_flush = time.time()

    def close(self):
        """Final flush, fsync, and close the file handle."""
        with self.lock:
            if not self._closed:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
                self._closed = True

# ----------------------------- RAW MOTION LOGGER -----------------------------
class RawMotionLogger:
    """Logs every raw accelerometer/gyroscope/magnetometer sample at full rate,
    independent of process_window()'s 1 Hz cadence (which only ever persists the
    single latest sample per second - fine for HR/EDA, which are meant to be
    smoothed, but it silently drops ~24 of every 25 motion samples).

    update() runs on the single UDP thread shared by every connected EmotiBit
    (see udp_thread()), so log() must never block on disk I/O - with 8 devices
    live, any per-call file open/write/close (the earlier approach) stalls that
    one thread for all of them and has crashed the sensor stream before. log()
    only enqueues; a dedicated per-device thread drains the queue and does the
    actual write against a handle opened once for the session (mirrors
    DataLogger), fsync'd on a timer instead of every row.
    """
    FLUSH_INTERVAL = 1.0
    QUEUE_WARN_SIZE = 20000  # backlog large enough to mean the writer can't keep up

    def __init__(self, source_id, output_dir="emotibit_recordings"):
        self.source_id = source_id
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = output_path / f"raw_motion_{source_id}_{timestamp}.csv"

        self._file = open(self.filename, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "device", "signal", "value"])
        self._file.flush()
        self._last_flush = time.time()
        self._closed = False

        self._queue = queue.Queue()
        self._last_backlog_warn = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._writer_loop, daemon=True,
            name=f"raw-motion-writer-{source_id}",
        )
        self._thread.start()

        print(f"  Raw motion logger initialized for {source_id} -> {self.filename}")

    def log(self, timestamp, signal, value):
        """Non-blocking enqueue - safe to call from the shared UDP receive thread."""
        self._queue.put_nowait((timestamp, signal, value))
        qsize = self._queue.qsize()
        if qsize > self.QUEUE_WARN_SIZE:
            now = time.time()
            if now - self._last_backlog_warn > 10.0:
                log.warning(
                    "%s raw motion writer backlog at %d items - disk write falling behind",
                    self.source_id, qsize,
                )
                self._last_backlog_warn = now

    def _writer_loop(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:
                ts, signal, value = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._writer.writerow([ts, self.source_id, signal, value])

            now = time.time()
            if now - self._last_flush >= self.FLUSH_INTERVAL:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._last_flush = now

    def close(self):
        """Signal the writer thread to drain and stop, then close the handle."""
        self._stop.set()
        self._thread.join(timeout=5.0)
        if not self._closed:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._closed = True

# ----------------------------- DEVICE AGGREGATOR -----------------------------
class DeviceAggregator:
    def __init__(self, source_id, clf_val, clf_aro, redis_client, all_signal_types=None):
        self.source_id = source_id
        self.values = {col: [] for col in COLUMNS}
        self.all_signals = all_signal_types or {}  # Store all signal types
        self.all_signal_values = {sig: [] for sig in self.all_signals.keys()}  # Buffer for all signals
        # Reentrant: process_window holds the lock while calling _compute_baseline_extras,
        # which may call _reset_baseline_unlocked. Reset listener thread also acquires.
        self.lock = threading.RLock()
        self.clf_val = clf_val
        self.clf_aro = clf_aro
        self.redis = redis_client
        # Engagement session join key, re-fetched from Redis every ~10 s
        self._eng_session_id = None
        self._eng_session_check = 0.0
        self.filtered_hr = []
        self.filtered_eda = []
        # Track 5 signals for Redis SD metrics
        self.edl_values = []  # Tonic EDA (EDL / EL tag)  -- ADS1114
        self.temp_values = []  # Skin temperature buffer (post-source-selection)
        # Per-wearer selected temperature source. Filled lazily by update() on the
        # first plausible sample; logged once via log.info so the .log file proves
        # which physical sensor the z-score was actually computed from. MD hardware
        # should pick 'Thermopile' (TH / MLX90632 = medical-grade skin sensor).
        self.temp_source = None
        # Per-wearer selected EDA source. EDA (EA tag = composite) is the raw
        # ADS1114 channel and the documented EmotiBit signal; EDL (EL tag = tonic)
        # is a derived component that some firmware/wearable combos don't emit.
        self.eda_source = None
        self.scr_freq_values = []  # SCR Frequency
        self.hr_values = []  # Heart Rate
        self.ibi_values = []  # Inter-Beat Interval
        self.data_logger = DataLogger(source_id, signal_types=self.all_signals)
        self.raw_motion_logger = RawMotionLogger(source_id)
        # Print throttling to reduce terminal overhead
        self.start_time = time.time()
        self.last_print_time = 0
        self.print_interval = 30.0  # Print every 30 seconds after initial 3 seconds
        # Throttle for post-calibration "missing metric" warnings (per metric name -> last log ts).
        # GUI's EDA spline goes dark when these are None after calibration ends; surfacing it
        # in the log file is the only way the operator can diagnose it after the fact.
        self._missing_metric_last_warn = {}
        self._missing_metric_warn_interval = 30.0

        # ---- Per-wearer session-z baseline state ----
        # Welford running mean/SD per channel, anchored to wearer-session start.
        self.welford = {
            'hr': WelfordState(),
            'eda': WelfordState(),
            'ibi': WelfordState(),
            'temperature_roc': WelfordState(),
            'scr_frequency': WelfordState(),
        }
        self.baseline_session_start = time.time()
        self.baseline_temp_c = None  # captured at end of calibration window
        self.swap_offwrist_start = None  # wall-clock when pinned-floor (off-wrist) began
        # Median pre-filter buffer for raw per-beat IBI (sample-and-hold artefacts)
        self._ibi_raw_recent = []
        # Last wall-clock arrival per channel group (for stale-data + swap detection)
        self.last_data_time = {'eda': 0.0, 'ppg': 0.0, 'temperature': 0.0, 'accelerometer': 0.0}
        # Quality flag per published channel
        self.channel_quality = {
            'hr': 'ok', 'eda': 'ok', 'ibi': 'ok',
            'temperature_roc': 'ok', 'scr_frequency': 'ok', 'scr_amplitude': 'ok',
        }

    def reset_baseline(self, reason="manual"):
        """Reset Welford state and calibration window. Call on wearer swap."""
        with self.lock:
            self._reset_baseline_unlocked(reason)

    def _engagement_session_id(self):
        """Current engagement session folder name from Redis (10 s cache).
        Engagement may start/stop mid-physio-run, so poll rather than read once."""
        now = time.time()
        if now - self._eng_session_check >= 10.0:
            self._eng_session_check = now
            try:
                v = self.redis.get('engagement:session_id')
                self._eng_session_id = v.decode() if isinstance(v, bytes) else v
            except Exception:
                pass  # keep last known value on transient Redis errors
        return self._eng_session_id

    def _reset_baseline_unlocked(self, reason):
        for w in self.welford.values():
            w.reset()
        self.baseline_session_start = time.time()
        self.baseline_temp_c = None
        self.swap_offwrist_start = None
        self._ibi_raw_recent = []
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.source_id} baseline reset ({reason})")
        log.info('baseline_reset', extra={'device': self.source_id, 'reason': reason})

    def update(self, col, value):
        """Update buffers for one signal sample.

        Raw samples are always logged to the CSV buffer (unfiltered record).
        Plausibility-gated copies populate the RF / SD / Welford buffers so
        impossible EmotiBit values (e.g. IBI 80 ms / 6,160 ms, HR > 220 BPM)
        never reach the statistics. IBI is additionally de-duplicated against
        the sample-and-hold UDP stream and median-pre-filtered over the most
        recent plausible beats.
        """
        with self.lock:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return
            if not np.isfinite(v):
                return

            max_samples = int(WINDOW_SECONDS * EMIT_RATE_HZ)
            now_wall = time.time()

            # Always log raw to CSV buffer. SCRAmplitude is the one exception:
            # it's gated below so a placeholder sentinel can be swapped for NaN
            # before it ever lands in the buffer this reads from.
            if col in self.all_signal_values and col != "SCRAmplitude":
                self.all_signal_values[col].append(v)
                self.all_signal_values[col] = self.all_signal_values[col][-max_samples:]

            # Track arrival per channel group (powers stale-data flags + swap detection)
            if col in ("EDA", "EDL", "EDR") or col.startswith("SCR"):
                self.last_data_time['eda'] = now_wall
            elif col in ("HeartRate", "InterBeatInterval") or col.startswith("PPG"):
                self.last_data_time['ppg'] = now_wall
            elif col.startswith("Temperature") or col == "Thermopile":
                self.last_data_time['temperature'] = now_wall
            elif col.startswith("Accelerometer") or col.startswith("Gyroscope") or col.startswith("Magnetometer"):
                self.last_data_time['accelerometer'] = now_wall
                self.raw_motion_logger.log(now_wall, col, v)

            # Plausibility gates before RF / SD / Welford buffers
            if col == "HeartRate":
                if not _plausible(v, PLAUSIBLE_HR_BPM):
                    self.channel_quality['hr'] = 'low'
                    return
                self.channel_quality['hr'] = 'ok'
                self.values['HeartRate'].append(v)
                self.values['HeartRate'] = self.values['HeartRate'][-max_samples:]
                self.hr_values.append(v)
                self.hr_values = self.hr_values[-max_samples:]
            elif col == "EDA":
                if not _plausible(v, PLAUSIBLE_EDA_US):
                    self.channel_quality['eda'] = 'low'
                    return
                self.channel_quality['eda'] = 'ok'
                self.values['EDA'].append(v)
                self.values['EDA'] = self.values['EDA'][-max_samples:]
                if self.eda_source is None:
                    self.eda_source = 'EDA'
                    log.info(
                        "%s EDA source = EDA (EA tag, raw ADS1114) -- primary path",
                        self.source_id,
                    )
            elif col == "EDL":
                if not _plausible(v, PLAUSIBLE_EDA_US):
                    self.channel_quality['eda'] = 'low'
                    return
                self.edl_values.append(v)
                self.edl_values = self.edl_values[-max_samples:]
                if self.eda_source is None:
                    # EDL only becomes primary if no EDA sample was seen first.
                    # _compute_baseline_extras falls back from edl_values -> values['EDA'],
                    # so logging this just records the first-arrival fact.
                    self.eda_source = 'EDL'
                    log.info(
                        "%s EDA source = EDL (EL tag, derived tonic) -- fallback path",
                        self.source_id,
                    )
            elif col == "InterBeatInterval":
                if not _plausible(v, PLAUSIBLE_IBI_MS):
                    self.channel_quality['ibi'] = 'low'
                    return
                # De-duplicate sample-and-hold values: only push when value changes
                if self._ibi_raw_recent and abs(self._ibi_raw_recent[-1] - v) < 1e-6:
                    return
                self._ibi_raw_recent.append(v)
                self._ibi_raw_recent = self._ibi_raw_recent[-IBI_MEDIAN_KERNEL:]
                v_filt = float(np.median(self._ibi_raw_recent)) if len(self._ibi_raw_recent) >= IBI_MEDIAN_KERNEL else v
                self.channel_quality['ibi'] = 'ok'
                self.ibi_values.append(v_filt)
                self.ibi_values = self.ibi_values[-max_samples:]
            elif col == "Thermopile":
                # TH = MLX90632 medical-grade thermopile (MD hardware). This is
                # the true non-contact SKIN temperature. Preferred source on MD;
                # if both TH and T1 arrive on the same wearer, TH wins and T1 is
                # ignored for the temp_values buffer to avoid mixing scales.
                if not _plausible(v, PLAUSIBLE_SKIN_TEMP_C):
                    self.channel_quality['temperature_roc'] = 'low'
                    return
                self.channel_quality['temperature_roc'] = 'ok'
                if self.temp_source != 'Thermopile':
                    if self.temp_source is not None:
                        # Upgrade path: a T0/T1 sample arrived first but TH is
                        # now available. Reset buffer so ROC isn't computed across
                        # mismatched scales.
                        log.info(
                            "%s upgrading skin-temp source: %s -> Thermopile (TH/MLX90632, true skin)",
                            self.source_id, self.temp_source,
                        )
                        self.temp_values = []
                    else:
                        log.info(
                            "%s skin-temp source = Thermopile (TH/MLX90632, medical-grade skin)",
                            self.source_id,
                        )
                    self.temp_source = 'Thermopile'
                self.temp_values.append(v)
                self.temp_values = self.temp_values[-max_samples:]
            elif col in ("Temperature0", "Temperature1"):
                # T1 (MAX30101 die) reads ~4-6 C below skin. T0 only exists on
                # legacy V1-V3 hardware. Used only when TH (Thermopile) isn't
                # streaming -- on MD hardware TH should always win.
                if self.temp_source == 'Thermopile':
                    # TH is the authoritative skin sensor; ignore die-temp samples
                    # so they don't contaminate the ROC buffer.
                    return
                if not _plausible(v, PLAUSIBLE_DIE_TEMP_C):
                    self.channel_quality['temperature_roc'] = 'low'
                    return
                self.channel_quality['temperature_roc'] = 'ok'
                if self.temp_source != col:
                    log.info(
                        "%s skin-temp source = %s (FALLBACK -- not true skin temp; "
                        "Thermopile/TH not received from this wearable)",
                        self.source_id, col,
                    )
                    self.temp_source = col
                self.temp_values.append(v)
                self.temp_values = self.temp_values[-max_samples:]
            elif col == "SCRFrequency":
                if not _plausible(v, PLAUSIBLE_SCR_FREQ):
                    self.channel_quality['scr_frequency'] = 'low'
                    return
                self.channel_quality['scr_frequency'] = 'ok'
                self.scr_freq_values.append(v)
                self.scr_freq_values = self.scr_freq_values[-max_samples:]
            elif col == "SCRAmplitude":
                if not _plausible(v, PLAUSIBLE_SCR_AMPLITUDE):
                    self.channel_quality['scr_amplitude'] = 'low'
                    v = float('nan')  # sentinel/placeholder -> NaN, never a
                                       # fabricated numeric reading
                else:
                    self.channel_quality['scr_amplitude'] = 'ok'
                if "SCRAmplitude" in self.all_signal_values:
                    self.all_signal_values["SCRAmplitude"].append(v)
                    self.all_signal_values["SCRAmplitude"] = self.all_signal_values["SCRAmplitude"][-max_samples:]

    def _compute_baseline_extras(self):
        """Update Welford running stats with current 1Hz representatives,
        compute per-channel signed z-scores, soft/hard event flags, quality
        flags, and auto wearer-swap detection.

        Returns a dict ready to merge into the physio_metrics payload, plus a
        'swap_detected' flag the caller acts on (outside the lock).
        """
        now_wall = time.time()
        age_s = now_wall - self.baseline_session_start
        calibrating = age_s < CALIBRATION_SECONDS
        sps = EMIT_RATE_HZ  # samples per second

        # 1Hz representative values per channel (last 1s of plausibility-gated buffer).
        # Mean of 25 samples gives independent-ish 1Hz draws into Welford rather than
        # the heavily-correlated 5s sliding-window mean.
        def _mean_tail(buf, n=sps):
            if not buf:
                return None
            tail = buf[-n:] if len(buf) >= n else buf
            try:
                m = float(np.mean(tail))
                return m if np.isfinite(m) else None
            except Exception:
                return None

        hr_repr = _mean_tail(self.hr_values)
        # EDL is the tonic component (EmotiBit short tag "EL"). Some firmware
        # builds / wearable units don't stream EDL at all -- only the composite
        # "EDA" tag (tonic + phasic). Fall back to plain EDA so the Welford
        # baseline and downstream eda_z never go permanently dark just because
        # the wearer's EmotiBit isn't emitting EDL packets.
        eda_repr = _mean_tail(self.edl_values)
        if eda_repr is None:
            eda_repr = _mean_tail(self.values.get('EDA', []))
        scr_repr = _mean_tail(self.scr_freq_values)
        # IBI: latest de-duplicated + median-filtered beat (one update per new beat).
        ibi_repr = float(self.ibi_values[-1]) if self.ibi_values else None
        # Temperature ROC: derivative on tail of temp buffer.
        temp_roc_repr = None
        if len(self.temp_values) >= 2:
            tail = self.temp_values[-sps:] if len(self.temp_values) >= sps else self.temp_values
            diffs = np.diff(tail)
            if len(diffs) > 0:
                m = float(np.mean(diffs) * EMIT_RATE_HZ)
                if np.isfinite(m):
                    temp_roc_repr = m

        # Update Welford with the per-channel representative -- but only while the
        # baseline is still expanding. Once BASELINE_CAP_SECONDS has elapsed the
        # mean/SD freeze, so sustained arousal is scored against the early-wear
        # reference instead of inflating it.
        if age_s < BASELINE_CAP_SECONDS:
            if hr_repr is not None: self.welford['hr'].update(hr_repr)
            if ibi_repr is not None: self.welford['ibi'].update(ibi_repr)
            if temp_roc_repr is not None: self.welford['temperature_roc'].update(temp_roc_repr)
            if scr_repr is not None: self.welford['scr_frequency'].update(scr_repr)
        # EDA runs its own delayed window (see EDA_SETTLE_SECONDS).
        if EDA_SETTLE_SECONDS <= age_s < EDA_BASELINE_CAP_SECONDS:
            if eda_repr is not None: self.welford['eda'].update(eda_repr)

        # Capture baseline skin temperature at end of calibration window
        # (used as the reference for swap detection's temperature step).
        if not calibrating and self.baseline_temp_c is None and self.temp_values:
            self.baseline_temp_c = _mean_tail(self.temp_values, sps)

        # Per-channel signed z-scores (None during calibration or insufficient data)
        def _z(val, w):
            if val is None or calibrating or w.n < BASELINE_MIN_SAMPLES or w.sd <= 0:
                return None
            return (val - w.mean) / w.sd

        z_scores = {
            'hr_z': _z(hr_repr, self.welford['hr']),
            'eda_z': _z(eda_repr, self.welford['eda']),
            'ibi_z': _z(ibi_repr, self.welford['ibi']),
            'temperature_roc_z': _z(temp_roc_repr, self.welford['temperature_roc']),
            'scr_frequency_z': _z(scr_repr, self.welford['scr_frequency']),
        }
        events = {}
        for k, z in z_scores.items():
            base = k[:-2]  # strip trailing '_z'
            events[f'{base}_event_soft'] = (z is not None and abs(z) > Z_SOFT_THRESHOLD)
            events[f'{base}_event_hard'] = (z is not None and abs(z) > Z_HARD_THRESHOLD)

        # Quality flags: stale data on a channel group => low quality on its derived metrics
        def _fresh(group):
            return (now_wall - self.last_data_time.get(group, 0.0)) < SWAP_DATA_FRESH_SECONDS
        quality = {
            'hr':              self.channel_quality['hr']              if _fresh('ppg')         else 'low',
            'ibi':             self.channel_quality['ibi']             if _fresh('ppg')         else 'low',
            'eda':             self.channel_quality['eda']             if _fresh('eda')         else 'low',
            'temperature_roc': self.channel_quality['temperature_roc'] if _fresh('temperature') else 'low',
            'scr_frequency':   self.channel_quality['scr_frequency']   if _fresh('eda')         else 'low',
        }

        # Wearer-swap auto-detection (pinned-floor heuristic).
        # EmotiBits keep streaming when removed from skin -- they don't go silent,
        # they emit pinned junk: EDA at the implausibility floor and PPGGreen at
        # near-zero (no skin to reflect the LED). The 5 June canteen test proved
        # the UDP-silence heuristic could never fire because the device was still
        # publishing packets the whole time it was off the wrist.
        # Off-wrist = sustained collapse of EDA AND PPGGreen below their floors.
        # When the wrist is re-attached and both rise back above their floors,
        # if the collapse lasted >= SWAP_PINNED_SECONDS we trigger a reset. The
        # temperature step is now an *optional* confirmer (skipped when the temp
        # stream is unavailable, as it was on Sowmya's MD-V5-0000334 with T0).
        eda_recent = self.all_signal_values.get('EDA', [])
        ppg_recent = self.all_signal_values.get('PPGGreen', [])
        # Mean over last second of raw samples (~25 samples each)
        eda_mean_recent = float(np.mean(eda_recent[-sps:])) if len(eda_recent) >= 3 else None
        ppg_mean_recent = float(np.mean(ppg_recent[-sps:])) if len(ppg_recent) >= 3 else None
        # Std over the same window -- used by the frozen-channel branch to catch
        # units whose electrodes rail/latch at a HIGH pinned value when removed.
        eda_std_recent = float(np.std(eda_recent[-sps:])) if len(eda_recent) >= sps else None
        ppg_std_recent = float(np.std(ppg_recent[-sps:])) if len(ppg_recent) >= sps else None
        offwrist_low = (
            eda_mean_recent is not None and ppg_mean_recent is not None and
            eda_mean_recent < EDA_OFFWRIST_THRESHOLD and
            ppg_mean_recent < PPG_OFFWRIST_THRESHOLD
        )
        # Frozen branch: both channels show no variation across the last
        # second of RAW samples. Real skin always produces jitter (EDA pulse
        # noise + PPG cardiac waveform), so zero std on both = no skin contact
        # regardless of mean magnitude. Requires BOTH (a single frozen channel
        # could be an ADC glitch; both frozen for a full second is unambiguous).
        offwrist_frozen = (
            eda_std_recent is not None and ppg_std_recent is not None and
            eda_std_recent < EDA_FROZEN_STD_THRESHOLD and
            ppg_std_recent < PPG_FROZEN_STD_THRESHOLD
        )
        currently_offwrist = offwrist_low or offwrist_frozen
        swap_detected = False
        # Short off-wrist event (>= OFF_WRIST_SECONDS, well below the SWAP_PINNED_SECONDS
        # baseline-reset threshold). Surfaces an `off_wrist` boolean so the GUI can
        # drop / watermark the splines without faking values, and the publisher omits
        # the *_z keys from the payload to keep stale baseline-relative readings off
        # the operator screen the moment skin contact is lost.
        off_wrist = False
        if currently_offwrist:
            if self.swap_offwrist_start is None:
                self.swap_offwrist_start = now_wall
            off_wrist = (now_wall - self.swap_offwrist_start) >= OFF_WRIST_SECONDS
        elif self.swap_offwrist_start is not None:
            offwrist_duration = now_wall - self.swap_offwrist_start
            # Any sustained off-wrist gap >= SWAP_PINNED_SECONDS forces a fresh
            # baseline the moment the sensor is re-fitted. We intentionally do NOT
            # require a skin-temp step here: the operator may re-fit the same OR a
            # different wearer, and in both cases the z-score baseline must be
            # recomputed from scratch so a new wearer is never plotted against the
            # previous wearer's mean/SD.
            if offwrist_duration >= SWAP_PINNED_SECONDS:
                swap_detected = True
            self.swap_offwrist_start = None

        return {
            'calibrating': calibrating,
            'calibration_remaining_s': round(max(0.0, CALIBRATION_SECONDS - age_s), 1),
            'session_age_s': round(age_s, 1),
            'baseline_n': int(self.welford['hr'].n),
            'z_scores': z_scores,
            'events': events,
            'quality': quality,
            'swap_detected': swap_detected,
            'off_wrist': off_wrist,
        }

    def process_window(self):
        processing_start = time.time()  # Start latency measurement
        
        with self.lock:
            if len(self.values["HeartRate"]) < 10 or len(self.values["EDA"]) < 10:
                return

            hr_raw = np.array(self.values["HeartRate"]).copy()
            eda_raw = np.array(self.values["EDA"]).copy()

            hr = filter_hr(hr_raw)
            eda = filter_eda(eda_raw)

            L = min(len(hr), len(eda))
            hr = hr[-L:]
            eda = eda[-L:]

            self.filtered_hr = hr
            self.filtered_eda = eda

            df = pd.DataFrame({"EDA": eda, "HeartRate": hr})

        # --- Features & continuous predictions ---
        feats = extract_features_window(df)
        X = pd.DataFrame([feats])

        try:
            val_proba = self.clf_val.predict_proba(X)[0]
            aro_proba = self.clf_aro.predict_proba(X)[0]
            val_cont = np.dot(val_proba, [0, 1, 2])
            aro_cont = np.dot(aro_proba, [0, 1, 2])

            ts = datetime.now().isoformat()

            # --- Calculate SD metrics and performance metrics ---
            # SD of 5 signals
            # SD of tonic EDA buffer. Falls back to the composite EDA stream
            # when the wearer's EmotiBit doesn't emit the EDL short-tag.
            if len(self.edl_values) > 1:
                edl_sd = float(np.std(self.edl_values))
            elif len(self.values.get('EDA', [])) > 1:
                edl_sd = float(np.std(self.values['EDA']))
            else:
                edl_sd = None
            
            temp_roc_sd = None
            if len(self.temp_values) > 2:
                temp_roc = np.diff(self.temp_values) * EMIT_RATE_HZ
                temp_roc_sd = float(np.std(temp_roc))
            
            scr_freq_sd = float(np.std(self.scr_freq_values)) if len(self.scr_freq_values) > 1 else None
            hr_sd = float(np.std(self.hr_values)) if len(self.hr_values) > 1 else None
            ibi_sd = float(np.std(self.ibi_values)) if len(self.ibi_values) > 1 else None
            
            # Performance metric: processing time
            processing_time_ms = (time.time() - processing_start) * 1000

            # --- Log all signals + SD metrics + performance metrics ---
            signal_dict = {}
            for sig_type, _ in self.all_signals.items():
                if sig_type in self.all_signal_values and len(self.all_signal_values[sig_type]) > 0:
                    signal_dict[sig_type] = float(self.all_signal_values[sig_type][-1])
                else:
                    signal_dict[sig_type] = None
            
            # Add predictions
            signal_dict["valence"] = float(val_cont)
            signal_dict["arousal"] = float(aro_cont)
            
            # Add SD metrics (for CSV storage)
            signal_dict["edl_sd"] = edl_sd
            signal_dict["eda_sd"] = edl_sd
            signal_dict["temperature_roc_sd"] = temp_roc_sd
            signal_dict["scr_frequency_sd"] = scr_freq_sd
            signal_dict["hr_sd"] = hr_sd
            signal_dict["ibi_sd"] = ibi_sd

            # --- Per-wearer session-z baseline extras (computed once, used by both CSV + Redis) ---
            with self.lock:
                extras = self._compute_baseline_extras()
                if extras['swap_detected']:
                    self._reset_baseline_unlocked(reason='auto_swap_detected')
                    # Recompute against the fresh state so this cycle reports calibrating.
                    extras = self._compute_baseline_extras()
                    extras['swap_detected'] = True  # preserve event for CSV / downstream

            # Persist baseline state into the CSV row so detector behaviour can be
            # audited from the file (the 5 June canteen test was inconclusive
            # because these columns weren't being written).
            signal_dict["calibrating"] = bool(extras['calibrating'])
            signal_dict["calibration_remaining_s"] = extras['calibration_remaining_s']
            signal_dict["baseline_n"] = extras['baseline_n']
            for k, v in extras['z_scores'].items():
                signal_dict[k] = float(v) if v is not None else ""
            signal_dict["hr_event_hard"] = bool(extras['events'].get('hr_event_hard', False))
            signal_dict["eda_event_hard"] = bool(extras['events'].get('eda_event_hard', False))
            signal_dict["ibi_event_hard"] = bool(extras['events'].get('ibi_event_hard', False))
            q = extras['quality']
            signal_dict["quality_hr"] = q.get('hr', '')
            signal_dict["quality_eda"] = q.get('eda', '')
            signal_dict["quality_ibi"] = q.get('ibi', '')
            signal_dict["quality_temperature_roc"] = q.get('temperature_roc', '')
            signal_dict["quality_scr_frequency"] = q.get('scr_frequency', '')
            # Not part of the freshness-gated z-score quality dict (no Welford/z
            # exists for amplitude) - sourced straight from the plausibility gate.
            signal_dict["quality_scr_amplitude"] = self.channel_quality.get('scr_amplitude', '')
            signal_dict["swap_detected"] = bool(extras['swap_detected'])
            signal_dict["off_wrist"] = bool(extras.get('off_wrist', False))
            signal_dict["engagement_session_id"] = self._engagement_session_id() or ""

            # Add performance metric (for CSV storage)
            signal_dict["processing_time_ms"] = processing_time_ms
            
            self.data_logger.log_signal_data(
                timestamp=ts,
                device=self.source_id,
                signal_dict=signal_dict
            )

            # --- Calculate and publish standard deviations of 5 signals to Redis ---
            physio_metrics = {"device": self.source_id, "timestamp": ts}
            
            # 1. Standard deviation of Tonic EDA (EDL)
            if edl_sd is not None:
                physio_metrics["edl_sd"] = edl_sd
                physio_metrics["eda_sd"] = edl_sd
            
            # 2. Standard deviation of Temperature Rate of Change
            if temp_roc_sd is not None:
                physio_metrics["temperature_roc_sd"] = temp_roc_sd
            
            # 3. Standard deviation of SCR Frequency
            if scr_freq_sd is not None:
                physio_metrics["scr_frequency_sd"] = scr_freq_sd
            
            # 4. Standard deviation of Heart Rate
            if hr_sd is not None:
                physio_metrics["hr_sd"] = hr_sd
            
            # 5. Standard deviation of Inter-Beat Interval
            if ibi_sd is not None:
                physio_metrics["ibi_sd"] = ibi_sd

            # --- Per-wearer session-z baseline extras (additive schema) ---
            physio_metrics['calibrating'] = bool(extras['calibrating'])
            physio_metrics['calibration_remaining_s'] = extras['calibration_remaining_s']
            physio_metrics['session_age_s'] = extras['session_age_s']
            physio_metrics['baseline_n'] = extras['baseline_n']
            # Off-wrist: drop *_z keys from the payload entirely. Faking a value
            # (e.g. 0) would mislead the operator into reading skin-contact
            # "baseline" when the device is on a table. The GUI subscriber's
            # "drop stale key" logic then greys the readouts, and the explicit
            # boolean below lets it render an OFF-WRIST watermark.
            off_wrist = bool(extras.get('off_wrist', False))
            physio_metrics['off_wrist'] = off_wrist
            if not off_wrist:
                for k, v in extras['z_scores'].items():
                    if v is not None:
                        physio_metrics[k] = round(float(v), 3)
            for k, v in extras['events'].items():
                physio_metrics[k] = bool(v)
            physio_metrics['quality'] = extras['quality']

            # Note: Performance metrics (processing_time_ms, sample counts) are only saved to CSV, not published to Redis
            
            # Publish consolidated metrics (pub/sub for live GUI)
            self.redis.publish(CHANNEL_PHYSIO.format(src=self.source_id), json.dumps(physio_metrics))

            # Also persist to Redis Stream for timestamp-based synchronization with audio
            _safe_xadd(self.redis, CHANNEL_PHYSIO.format(src=self.source_id), physio_metrics)

            # Post-calibration sanity check: if the GUI-facing EDA channels are still empty
            # after the warm-up window expired, the spline will render as "Physio: no signal".
            # Log a throttled warning so the cause (e.g. EDA off-wrist, dead electrodes,
            # plausibility gate rejecting every sample) is captured in the .log file.
            if not extras['calibrating']:
                eda_z_val = extras['z_scores'].get('eda_z')
                now_ts = time.time()
                # eda_z is legitimately absent during the EDA settle window.
                eda_expected = (now_ts - self.baseline_session_start) >= (EDA_SETTLE_SECONDS + 5)
                if eda_z_val is None and eda_expected:
                    last = self._missing_metric_last_warn.get('eda_z', 0.0)
                    if now_ts - last >= self._missing_metric_warn_interval:
                        log.warning(
                            "eda_z missing post-calibration for %s (baseline_n=%d, "
                            "edl_samples=%d, eda_samples=%d). GUI spline will show "
                            "'Physio: no signal'. Likely off-wrist, dead electrodes, "
                            "or EDA samples rejected by plausibility gate (%s).",
                            self.source_id, extras['baseline_n'],
                            len(self.edl_values), len(self.values.get('EDA', [])),
                            PLAUSIBLE_EDA_US,
                        )
                        self._missing_metric_last_warn['eda_z'] = now_ts
                if edl_sd is None:
                    last = self._missing_metric_last_warn.get('edl_sd', 0.0)
                    if now_ts - last >= self._missing_metric_warn_interval:
                        log.warning(
                            "edl_sd missing post-calibration for %s "
                            "(edl_samples=%d, eda_samples=%d). Fallback EDA-SD "
                            "spline path is also unavailable.",
                            self.source_id, len(self.edl_values),
                            len(self.values.get('EDA', [])),
                        )
                        self._missing_metric_last_warn['edl_sd'] = now_ts

            # --- Redis publish: continuous val/arousal (pub/sub + streams) ---
            val_payload = {"device": self.source_id, "valence": float(val_cont), "timestamp": ts}
            aro_payload = {"device": self.source_id, "arousal": float(aro_cont), "timestamp": ts}

            self.redis.publish(CHANNEL_VAL.format(src=self.source_id), json.dumps(val_payload))
            self.redis.publish(CHANNEL_ARO.format(src=self.source_id), json.dumps(aro_payload))

            # Persist valence/arousal to streams for sync with audio
            _safe_xadd(self.redis, CHANNEL_VAL.format(src=self.source_id), val_payload)
            _safe_xadd(self.redis, CHANNEL_ARO.format(src=self.source_id), aro_payload)

            # --- Publish raw sensor data (accelerometer, gyroscope, etc.) ---
            # Get latest values from all_signal_values buffer for motion/orientation/context analysis
            raw_sensors = {"device": self.source_id, "timestamp": ts}
            sensor_keys = ['AccelerometerX', 'AccelerometerY', 'AccelerometerZ',
                          'GyroscopeX', 'GyroscopeY', 'GyroscopeZ',
                          'MagnetometerX', 'MagnetometerY', 'MagnetometerZ',
                          'PPGGreen', 'PPGInfrared', 'PPGRed',
                          'Thermopile', 'Temperature0', 'Temperature1',
                          'BatteryVoltage', 'BatteryPercent',
                          'Humidity', 'SpO2']
            for key in sensor_keys:
                if key in self.all_signal_values and len(self.all_signal_values[key]) > 0:
                    raw_sensors[key] = float(self.all_signal_values[key][-1])  # Latest value
            _safe_xadd(self.redis, CHANNEL_RAW_SENSORS.format(src=self.source_id), raw_sensors)

            # Print status with SD metrics and performance (throttled to reduce overhead)
            current_time = time.time()
            elapsed = current_time - self.start_time
            # Print for first 3 seconds, then every 30 seconds
            if elapsed <= 3.0 or (current_time - self.last_print_time) >= self.print_interval:
                sd_metrics = [f"{k}={v:.3f}" for k, v in [
                    ("eda_sd", edl_sd), ("temp_roc_sd", temp_roc_sd), ("scr_freq_sd", scr_freq_sd),
                    ("hr_sd", hr_sd), ("ibi_sd", ibi_sd)
                ] if v is not None]
                metrics_str = ", ".join(sd_metrics) + f", proc={processing_time_ms:.1f}ms"
                if extras['calibrating']:
                    base_str = f"calibrating({extras['calibration_remaining_s']:.0f}s left,n={extras['baseline_n']})"
                else:
                    zs = extras['z_scores']
                    z_parts = [f"{k[:-2]}={zs[k]:+.2f}" for k in ('hr_z', 'eda_z', 'ibi_z') if zs[k] is not None]
                    base_str = " ".join(z_parts) if z_parts else f"baseline_n={extras['baseline_n']}"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.source_id} {metrics_str}, {base_str}, Val={val_cont:.2f}, Aro={aro_cont:.2f}")
                self.last_print_time = current_time

        except Exception as e:
            print("[WARN] Prediction skipped:", e)

# ----------------------------- UDP PROTOCOL HELPERS -----------------------------

_pkt_counter      = 0
_pkt_counter_lock = threading.Lock()
_start_time       = time.time()


def _next_pn():
    global _pkt_counter
    with _pkt_counter_lock:
        n = _pkt_counter; _pkt_counter += 1
    return n


def make_pkt(tag, data=None):
    d = [str(x) for x in (data or [])]
    ts = int((time.time() - _start_time) * 1000)
    parts = [str(ts), str(_next_pn()), str(len(d)), tag, "1", "100"] + d
    return (",".join(parts) + "\n").encode("ascii")


def _parse_text(text: str):
    results = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 6:
            continue
        type_tag = parts[3].strip()
        if type_tag not in TYPE_TAG_MAP:
            continue
        try:
            data_len = int(parts[2])
            values = [float(v) for v in parts[6:6 + data_len] if v.strip()]
            results.append({
                "stream_name": TYPE_TAG_MAP[type_tag],
                "type_tag":    type_tag,
                "values":      values,
            })
        except (ValueError, IndexError):
            continue
    return results


def _broadcast_targets():
    targets = {"255.255.255.255"}
    try:
        import netifaces
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET, [])
            for a in addrs:
                bcast = a.get('broadcast')
                if bcast:
                    targets.add(bcast)
    except ImportError:
        try:
            hostname = socket.gethostname()
            for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ip = info[4][0]
                if ip.startswith("127."):
                    continue
                parts = ip.split('.')
                if len(parts) == 4:
                    targets.add(f"{parts[0]}.{parts[1]}.{parts[2]}.255")
        except Exception:
            pass
    return list(targets)


def discover_devices(adv_sock, timeout_s, own_ips=None, interactive=True):
    """Discover EmotiBit devices via HE/HH/EC/PO handshake on UDP 3131.

    interactive=False is used for runtime reconnects (triggered from the GUI
    reconnect button): it skips the keyboard 'press ENTER' wait and just
    auto-accepts whatever responds within the window, so it can run unattended
    from a background thread.
    """
    _IS_WINDOWS = platform.system() == 'Windows'
    if _IS_WINDOWS:
        import msvcrt

    def _enter_pressed():
        if not interactive or not sys.stdin.isatty():
            return False
        if _IS_WINDOWS:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                return ch in ('\r', '\n')
            return False
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        if rlist:
            sys.stdin.readline()
            return True
        return False

    hh_seen   = {}
    connected = set()
    deadline  = time.time() + timeout_s
    next_he   = 0.0
    last_new_device_t = None
    AUTO_ACCEPT_DELAY = 5.0
    _warned_bcast = set()

    print(f"\nDiscovering EmotiBit devices (timeout {timeout_s}s)&")
    print(f"  Advertising port: {EMOTIBIT_CONTROL_PORT}")
    print(f"  Data port:        {EMOTIBIT_DATA_PORT}")
    print(f"  TCP control port: {EMOTIBIT_TCP_PORT}")
    print("Power on devices or ensure they are already broadcasting.")
    if interactive:
        print(">>> Press ENTER once all devices are found to continue <<<\n")

    while time.time() < deadline:
        if connected and _enter_pressed():
            print(f"\n  Accepted {len(connected)} device(s).")
            break

        if (last_new_device_t is not None
                and time.time() - last_new_device_t >= AUTO_ACCEPT_DELAY):
            print(f"\n  No new devices for {AUTO_ACCEPT_DELAY:.0f}s - "
                  f"auto-accepting {len(connected)} device(s).")
            break

        now = time.time()

        if now >= next_he:
            targets = _broadcast_targets()
            for bcast_addr in targets:
                try:
                    adv_sock.sendto(make_pkt("HE"),
                                    (bcast_addr, EMOTIBIT_CONTROL_PORT))
                except OSError as e:
                    if bcast_addr not in _warned_bcast:
                        _warned_bcast.add(bcast_addr)
                        print(f"  Warning: broadcast to {bcast_addr} failed: {e}")
            print(f"  ’ HE broadcast sent ({len(targets)} target(s))&")
            next_he = now + 3.0

        for ip in set(hh_seen) - connected:
            try:
                ec = make_pkt("EC", ["CP", EMOTIBIT_TCP_PORT, "DP", EMOTIBIT_DATA_PORT])
                adv_sock.sendto(ec, (ip, EMOTIBIT_CONTROL_PORT))
            except Exception:
                pass

        try:
            adv_sock.settimeout(0.1)
            pkt, (src_ip, _) = adv_sock.recvfrom(4096)
            if own_ips and src_ip in own_ips:
                pass
            else:
                text = pkt.decode('ascii', errors='ignore').strip()
                for line in text.split('\n'):
                    fields = line.strip().split(',')
                    if len(fields) < 6:
                        continue
                    tag = fields[3].strip()

                    if tag == "HH" and src_ip not in hh_seen:
                        device_id = "unknown"
                        for i, f in enumerate(fields):
                            if f.strip() == "DI" and i + 1 < len(fields):
                                device_id = fields[i + 1].strip()
                        hh_seen[src_ip] = device_id
                        print(f"  -  Device available: {src_ip} (ID: {device_id})")
                        ec = make_pkt("EC", ["CP", EMOTIBIT_TCP_PORT,
                                             "DP", EMOTIBIT_DATA_PORT])
                        adv_sock.sendto(ec, (src_ip, EMOTIBIT_CONTROL_PORT))

                    elif tag == "PO" and src_ip in hh_seen:
                        for i, f in enumerate(fields):
                            if f.strip() == "DP" and i + 1 < len(fields):
                                if fields[i + 1].strip() == str(EMOTIBIT_DATA_PORT):
                                    if src_ip not in connected:
                                        last_new_device_t = time.time()
                                        connected.add(src_ip)
                                        print(f"  -  Connection confirmed: {src_ip}")
                                        print(f"  [{len(connected)} device(s) found] "
                                              f"Press ENTER to finish, or wait for more&")
                                    break
        except socket.timeout:
            pass

        remaining = max(0, int(deadline - time.time()))
        print(f"  &waiting ({remaining}s, found: {len(connected)})    ", end='\r')

    print()
    devices = {ip: hh_seen.get(ip, "unknown") for ip in connected}
    for ip in hh_seen:
        if ip not in devices:
            print(f"     {ip} replied HH but no PO - adding anyway")
            devices[ip] = hh_seen[ip]
    return devices

# ----------------------------- LIVE PLOT (removed) -----------------------------
# The local matplotlib live plot was removed: the engagement GUI
# (live_multiperson_binary_v2.py) now subscribes to Redis and renders all
# physio signals in its sidebar, so plotting here was duplicate work and a
# blocking Tk main loop on the publisher process. The publisher now runs
# headless and blocks on a wait loop while the UDP / prediction daemon
# threads continue to push data to Redis.

# ========================= FIREWALL CHECK =======================

def _ensure_firewall_rules():
    """On Windows, verify that inbound firewall rules exist for the
    three EmotiBit ports.  On macOS the Application Firewall auto-prompts."""
    if platform.system() != 'Windows':
        return

    RULES = [
        ("AMPLIFY EmotiBit UDP 3131", "UDP", EMOTIBIT_CONTROL_PORT),
        ("AMPLIFY EmotiBit UDP 3132", "UDP", EMOTIBIT_DATA_PORT),
        ("AMPLIFY EmotiBit TCP 3133", "TCP", EMOTIBIT_TCP_PORT),
    ]

    missing = []
    for name, proto, port in RULES:
        try:
            result = subprocess.run(
                ["netsh", "advfirewall", "firewall", "show", "rule",
                 f"name={name}"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                missing.append((name, proto, port))
        except Exception:
            missing.append((name, proto, port))

    if not missing:
        print("Firewall: all EmotiBit rules present -")
        return

    print(f"Firewall: {len(missing)} rule(s) missing - attempting to add&")
    failed = []
    for name, proto, port in missing:
        cmd = [
            "netsh", "advfirewall", "firewall", "add", "rule",
            f"name={name}", "dir=in", f"action=allow",
            f"protocol={proto}", f"localport={port}",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print(f"  -  Added: {name}")
            else:
                failed.append((name, proto, port))
        except Exception:
            failed.append((name, proto, port))

    if failed:
        print("\n   Could not add the following rules (admin privileges required).")
        print("  Run these commands in an Administrator PowerShell, then restart:\n")
        for name, proto, port in failed:
            print(f'    netsh advfirewall firewall add rule '
                  f'name="{name}" dir=in action=allow '
                  f'protocol={proto} localport={port}')
        print()


# ----------------------------- MAIN -----------------------------
def main():
    # Redis is the comms hub: it may live on this machine or on a different
    # device (e.g. the AR/engagement laptop). Resolve the target from CLI flags,
    # then environment variables, then the localhost defaults — and write the
    # result back to the module globals so every connection and the log context
    # below use the same host/port.
    global REDIS_HOST, REDIS_PORT
    parser = argparse.ArgumentParser(
        description='EmotiBit UDP → Redis physiological publisher')
    parser.add_argument('--redis-host', default=os.environ.get('REDIS_HOST', 'auto'),
                        help='Redis server host/IP. Default "auto" finds the broker over the '
                             'network via mDNS (the advertiser must run on the hub, see '
                             '1B_START_REDIS_ADVERTISER.bat), then falls back to localhost. '
                             'Pass an explicit IP to skip discovery.')
    parser.add_argument('--redis-port', type=int,
                        default=int(os.environ.get('REDIS_PORT', REDIS_PORT)),
                        help='Redis server port (default: env REDIS_PORT or 6379).')
    parser.add_argument('--no-discover', action='store_true',
                        help='Disable mDNS auto-discovery; use localhost when no host is given.')
    args, _ = parser.parse_known_args()
    REDIS_PORT = args.redis_port
    if args.redis_host == 'auto':
        found = None
        if not args.no_discover:
            print("Looking for the Redis broker on the network (mDNS)...")
            try:
                from redis_discovery import resolve_redis_broker
                found = resolve_redis_broker(timeout=6.0)
            except Exception as e:
                print(f"   Discovery unavailable ({e}); falling back to localhost.")
        if found:
            REDIS_HOST, REDIS_PORT = found
            print(f"> Discovered Redis broker at {REDIS_HOST}:{REDIS_PORT}")
        else:
            REDIS_HOST = 'localhost'
            if not args.no_discover:
                print("> No advertised broker found; using localhost "
                      "(pass --redis-host <ip> to target another device).")
    else:
        REDIS_HOST = args.redis_host

    setup_logging('emotibit', extra_context={
        'redis': f'{REDIS_HOST}:{REDIS_PORT}',
        'window_seconds': str(WINDOW_SECONDS),
        'emit_rate_hz': str(EMIT_RATE_HZ),
    })
    install_excepthook(log)
    log.info('EmotiBit pipeline starting')
    _ensure_firewall_rules()

    print("Loading models...")

    # Check if model files exist
    if not os.path.exists(MODEL_PATH_VAL):
        print(f"L Model not found: {MODEL_PATH_VAL}")
        print(f"   Make sure '{Path(MODEL_PATH_VAL).name}' is in: {SCRIPT_DIR}")
        return
    if not os.path.exists(MODEL_PATH_ARO):
        print(f"L Model not found: {MODEL_PATH_ARO}")
        print(f"   Make sure '{Path(MODEL_PATH_ARO).name}' is in: {SCRIPT_DIR}")
        return

    clf_val = joblib.load(MODEL_PATH_VAL)
    clf_aro = joblib.load(MODEL_PATH_ARO)
    print("> Models loaded successfully")

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    try:
        r.ping()
        print(f"> Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        print(f"!! Could not reach Redis at {REDIS_HOST}:{REDIS_PORT}: {e}")
        print("   If Redis runs on another device, pass --redis-host <ip> (and open port 6379).")

    # ---- NETWORK SOCKETS ----
    # NOTE: no SO_REUSEADDR on the UDP sockets. UDP has no TIME_WAIT so it is
    # never needed for restarts — but with it set, a second publisher instance
    # binds the same ports silently and Windows delivers each packet to only
    # ONE of the sockets, so a forgotten stale instance steals all EmotiBit
    # traffic and discovery appears dead with no error. Better to fail loudly.
    def _bind_or_die(sock, port, label):
        try:
            sock.bind(("", port))
        except OSError:
            print(f"L Port {port} ({label}) is already in use.")
            print("   Another EmotiBit publisher is probably still running.")
            print("   Close it (check Task Manager for python.exe) and retry:")
            print(f"     Get-NetUDPEndpoint -LocalPort {port} | "
                  "Select-Object OwningProcess")
            sys.exit(1)

    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    _bind_or_die(ctrl_sock, EMOTIBIT_CONTROL_PORT, "EmotiBit control")
    ctrl_sock.settimeout(1.0)

    data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
    except OSError:
        pass
    _bind_or_die(data_sock, EMOTIBIT_DATA_PORT, "EmotiBit data")
    data_sock.settimeout(1.0)
    if platform.system() == 'Windows':
        import ctypes, ctypes.wintypes
        _ws2 = ctypes.windll.ws2_32
        _ret = ctypes.wintypes.DWORD()
        _ws2.WSAIoctl(
            data_sock.fileno(), 0x9800000C,
            ctypes.byref(ctypes.c_bool(False)), ctypes.sizeof(ctypes.c_bool),
            None, 0, ctypes.byref(_ret), None, None
        )

    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, 'SO_REUSEPORT'):
        try:
            tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
    tcp_sock.bind(("", EMOTIBIT_TCP_PORT))
    tcp_sock.listen(8)
    tcp_sock.settimeout(1.0)

    stop_flag = threading.Event()
    # Serialises all ctrl_sock access (heartbeat vs runtime reconnect handshake)
    # so the two threads never steal each other's UDP control-port packets.
    ctrl_lock = threading.Lock()
    # Guards the live device collections (device_aggs / agg_list / aggs_by_source /
    # discovered) which a runtime reconnect can mutate while the UDP + prediction
    # threads are iterating them.
    agg_lock = threading.Lock()

    def _tcp_accept():
        while True:
            try:
                conn, _ = tcp_sock.accept()
                threading.Thread(target=_tcp_handler, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                return

    def _tcp_handler(conn):
        # Keep TCP alive - EmotiBit disconnects data if TCP drops.
        conn.settimeout(5.0)
        try:
            while not stop_flag.is_set():
                try:
                    if not conn.recv(4096):
                        break
                except socket.timeout:
                    continue
                except Exception:
                    break
        finally:
            conn.close()

    threading.Thread(target=_tcp_accept, daemon=True).start()

    # Own IPs (to ignore broadcast loopback)
    own_ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            own_ips.add(info[4][0])
    except Exception:
        pass
    own_ips.update({"127.0.0.1", "::1"})

    # ---- SESSION TIMESTAMPS (for metadata) ----
    session_start_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_start_iso = datetime.now().isoformat()

    # ---- DISCOVER DEVICES ----
    discovered = discover_devices(ctrl_sock, DISCOVERY_TIMEOUT_S, own_ips)
    if not discovered:
        print("L No EmotiBit devices found. Exiting.")
        ctrl_sock.close(); data_sock.close(); tcp_sock.close()
        return

    print(f"> Found {len(discovered)} device(s)")

    # ---- EMERGENCY DISCONNECT (atexit + SIGTERM) ----
    # Ensures DC is sent to all EmotiBits even if the script is killed
    # or exits abnormally, preventing the need to power-cycle devices.
    _cleanup_done = threading.Event()

    def _emergency_disconnect():
        if _cleanup_done.is_set():
            return
        _cleanup_done.set()
        for device_ip in discovered:
            try:
                dc = make_pkt("DC")
                ctrl_sock.sendto(dc, (device_ip, EMOTIBIT_CONTROL_PORT))
            except Exception:
                pass

    atexit.register(_emergency_disconnect)

    def _sigterm_handler(signum, frame):
        _emergency_disconnect()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Closing the console window on Windows kills the process WITHOUT running
    # atexit, so the DC packet never reached the devices and they stayed
    # paired to a dead session (deaf to the next discovery until their own
    # timeout / power-cycle). A console control handler catches window close /
    # logoff / shutdown and sends DC first.
    if platform.system() == 'Windows':
        import ctypes
        import ctypes.wintypes

        def _win_console_handler(ctrl_type):
            # CTRL_CLOSE_EVENT=2, CTRL_LOGOFF_EVENT=5, CTRL_SHUTDOWN_EVENT=6
            if ctrl_type in (2, 5, 6):
                _emergency_disconnect()
                time.sleep(0.2)  # let the DC packets leave the NIC
            return False  # continue with default termination

        _HANDLER_ROUTINE = ctypes.WINFUNCTYPE(
            ctypes.wintypes.BOOL, ctypes.wintypes.DWORD)
        global _win_ctrl_handler_ref  # keep alive for process lifetime
        _win_ctrl_handler_ref = _HANDLER_ROUTINE(_win_console_handler)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_win_ctrl_handler_ref, 1)

    # All possible signal types for logging
    signal_types = {name: True for name in TYPE_TAG_MAP.values()}
    print(f"> Signal types: {', '.join(sorted(signal_types.keys()))}")

    # Create one DeviceAggregator per discovered device
    device_aggs = {}   # ip ’ DeviceAggregator
    agg_list    = []
    for dev_ip, dev_id in discovered.items():
        src_label = dev_id if dev_id != "unknown" else dev_ip.replace('.', '_')
        agg = DeviceAggregator(src_label, clf_val, clf_aro, r,
                               all_signal_types=signal_types)
        device_aggs[dev_ip] = agg
        agg_list.append(agg)

    # ---- HEARTBEAT ----

    def heartbeat_thread():
        while not stop_flag.is_set():
            pkt = make_pkt("PN", ["DP", EMOTIBIT_DATA_PORT])
            with ctrl_lock:
                for dev_ip in list(discovered):
                    try:
                        ctrl_sock.sendto(pkt, (dev_ip, EMOTIBIT_CONTROL_PORT))
                    except Exception:
                        pass
                # Drain PO responses so ctrl_sock buffer doesn't fill up
                try:
                    ctrl_sock.settimeout(0.05)
                    while True:
                        try:
                            ctrl_sock.recvfrom(4096)
                        except socket.timeout:
                            break
                except Exception:
                    pass
            time.sleep(HEARTBEAT_INTERVAL_S)

    threading.Thread(target=heartbeat_thread, daemon=True).start()

    # ---- BASELINE-RESET LISTENER ----
    # GUI / operator publishes an empty JSON to `device:{serial}:reset_baseline`
    # whenever a sensor is reassigned to a new wearer. We pattern-subscribe and
    # dispatch to the matching aggregator. Auto-swap detection in
    # _compute_baseline_extras is the safety net; this is the manual override.
    aggs_by_source = {agg.source_id: agg for agg in agg_list}

    def reset_listener_thread():
        try:
            pubsub = r.pubsub()
            pubsub.psubscribe('device:*:reset_baseline')
            for message in pubsub.listen():
                if stop_flag.is_set():
                    return
                if message.get('type') != 'pmessage':
                    continue
                try:
                    channel = message.get('channel')
                    if isinstance(channel, (bytes, bytearray)):
                        channel = channel.decode('utf-8', errors='ignore')
                    # channel = "device:{src}:reset_baseline"
                    parts = (channel or '').split(':')
                    if len(parts) >= 3:
                        src = parts[1]
                        agg = aggs_by_source.get(src)
                        if agg is not None:
                            agg.reset_baseline(reason='manual_redis')
                        else:
                            print(f"[WARN] reset_baseline received for unknown device: {src}")
                except Exception as e:
                    print(f"[WARN] reset listener dispatch: {e}")
        except Exception as e:
            print(f"[WARN] reset listener exited: {e}")

    threading.Thread(target=reset_listener_thread, daemon=True).start()

    # ---- RUNTIME RECONNECT ----
    # When an EmotiBit drops mid-session (Wi-Fi blip, power glitch) it stops
    # streaming and previously the only fix was to kill and relaunch the whole
    # pipeline. Instead, the GUI publishes to CHANNEL_RECONNECT and we re-run the
    # discovery handshake live: known devices resume streaming and any brand-new
    # device gets its own aggregator with a fresh per-wearer baseline. No restart.
    def do_reconnect(trigger='redis'):
        print(f"\n[RECONNECT] requested ({trigger}) - re-scanning for EmotiBit devices...")
        log.info('reconnect_requested', extra={'trigger': trigger})
        # Hold ctrl_lock for the whole handshake so the heartbeat thread doesn't
        # consume the HH/PO replies discovery is waiting for.
        with ctrl_lock:
            try:
                found = discover_devices(ctrl_sock, RECONNECT_TIMEOUT_S, own_ips,
                                         interactive=False)
            except Exception as e:
                found = {}
                print(f"[RECONNECT] discovery error: {e}")
            finally:
                ctrl_sock.settimeout(1.0)
        new_count = 0
        with agg_lock:
            for ip, dev_id in found.items():
                if ip in device_aggs:
                    continue  # known device -- EC handshake already re-sent, streaming resumes
                src_label = dev_id if dev_id != "unknown" else ip.replace('.', '_')
                agg = DeviceAggregator(src_label, clf_val, clf_aro, r,
                                       all_signal_types=signal_types)
                device_aggs[ip] = agg
                agg_list.append(agg)
                aggs_by_source[agg.source_id] = agg
                discovered[ip] = dev_id
                new_count += 1
                print(f"[RECONNECT] new device online: {ip} (ID: {dev_id}) - fresh baseline")
        print(f"[RECONNECT] done. {len(found)} device(s) responded, {new_count} new.")
        log.info('reconnect_done', extra={'responded': len(found), 'new': new_count})
        try:
            r.publish(CHANNEL_RECONNECT_STATUS,
                      json.dumps({'responded': len(found), 'new': new_count,
                                  'ts': time.time()}))
        except Exception:
            pass

    def reconnect_listener_thread():
        try:
            pubsub = r.pubsub()
            pubsub.subscribe(CHANNEL_RECONNECT)
            for message in pubsub.listen():
                if stop_flag.is_set():
                    return
                if message.get('type') != 'message':
                    continue
                do_reconnect(trigger='redis')
        except Exception as e:
            print(f"[WARN] reconnect listener exited: {e}")

    threading.Thread(target=reconnect_listener_thread, daemon=True).start()

    # ---- UDP RECEIVE THREAD ----
    def udp_thread():
        while not stop_flag.is_set():
            try:
                data, (src_ip, src_port) = data_sock.recvfrom(65535)
            except socket.timeout:
                continue
            except ConnectionResetError:
                continue
            except OSError:
                if stop_flag.is_set():
                    break
                continue

            try:
                agg = device_aggs.get(src_ip)
                if agg is None:
                    continue

                text = data.decode('ascii', errors='ignore')

                # Respond to RD with timestamp
                for line in text.strip().split('\n'):
                    fields = line.strip().split(',')
                    if len(fields) >= 4 and fields[3].strip() == "RD":
                        try:
                            tl = make_pkt("TL", [str(int(time.time() * 1000))])
                            data_sock.sendto(tl, (src_ip, src_port))
                        except OSError:
                            pass
                        break

                parsed_list = _parse_text(text)
                
                for parsed in parsed_list:
                    sname = parsed["stream_name"]
                    for value in parsed["values"]:
                        if SAVE_ALL_SIGNALS or sname in COLUMNS:
                            agg.update(sname, value)
            except Exception as e:
                print(f"    UDP processing error: {e}")

    threading.Thread(target=udp_thread, daemon=True).start()

    # ---- PREDICTION LOOP ----
    def prediction_loop():
        while not stop_flag.is_set():
            with agg_lock:
                aggs = list(agg_list)
            for agg in aggs:
                agg.process_window()
                agg.data_logger.flush()
            time.sleep(1.0)

    threading.Thread(target=prediction_loop, daemon=True).start()

    fig, ani = (None, None)  # plotting removed; engagement GUI renders signals

    try:
        print("<¬ Publisher running headless. Open the engagement GUI to view live plots.")
        print("   Press Ctrl+C in this window to stop publishing.")
        while not stop_flag.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nL Error: {e}")
    finally:
        print("\nFlushing data - please wait&")
        stop_flag.set()

        # 1. Drain remaining packets from the UDP socket buffer
        data_sock.settimeout(0.2)
        drained = 0
        while True:
            try:
                data, (src_ip, _) = data_sock.recvfrom(65535)
                agg = device_aggs.get(src_ip)
                if agg is not None:
                    parsed_list, _ = _parse_text(data.decode('ascii', errors='ignore'))
                    ts = datetime.now().isoformat()
                    for parsed in parsed_list:
                        sname = parsed["stream_name"]
                        for value in parsed["values"]:
                            agg.update(sname, value)
                            drained += 1
            except socket.timeout:
                break
            except Exception:
                break
        if drained:
            print(f"  Drained {drained} remaining samples from socket buffer.")

        # 2. Send DC (EMOTIBIT_DISCONNECT) to each device so the sensor
        #    releases the connection immediately.
        _emergency_disconnect()
        time.sleep(0.2)   # brief pause so the DC packet reaches the device

        # 3. Close sockets
        ctrl_sock.close()
        data_sock.close()
        tcp_sock.close()

        # 4. Flush and close every logger file handle
        for agg in list(agg_list):
            try:
                agg.data_logger.close()
            except Exception:
                pass
            try:
                agg.raw_motion_logger.close()
            except Exception:
                pass

        # 5. Save session metadata
        session_meta = {
            "script": Path(__file__).name,
            "session_start": session_start_iso,
            "session_end": datetime.now().isoformat(),
            "session_unix_end": time.time(),
            "devices": {
                ip: {"device_id": dev_id,
                     "source_label": device_aggs[ip].source_id}
                for ip, dev_id in discovered.items()
            },
            "signal_types": sorted(signal_types.keys()),
        }
        meta_path = Path(agg_list[0].data_logger.output_dir) / f"session_metadata_{session_start_ts}.json"
        with open(meta_path, 'w') as f:
            json.dump(session_meta, f, indent=4)

        print(f"> All data saved ’ {agg_list[0].data_logger.output_dir}/")

if __name__=="__main__":
    print(f"=¥   Running on: {sys.platform}")
    main()
