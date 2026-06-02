# ===============================================================
# -*- coding: utf-8 -*-
# REAL-TIME EMOTIBIT: FILTERED EDA + HR PLOT + HR PUBLISH TO REDIS
# Direct UDP - no EmotiBit Oscilloscope / LSL dependency
# ===============================================================

import atexit
import csv
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
CHANNEL_VAL = "device:{src}:valence_cont"
CHANNEL_ARO = "device:{src}:arousal_cont"

# ---- EmotiBit UDP protocol settings ----
EMOTIBIT_CONTROL_PORT = 3131
EMOTIBIT_DATA_PORT    = 3132
EMOTIBIT_TCP_PORT     = 3133
DISCOVERY_TIMEOUT_S   = 20
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

# ----------------------------- DEVICE AGGREGATOR -----------------------------
class DeviceAggregator:
    def __init__(self, source_id, clf_val, clf_aro, redis_client, all_signal_types=None):
        self.source_id = source_id
        self.values = {col: [] for col in COLUMNS}
        self.all_signals = all_signal_types or {}  # Store all signal types
        self.all_signal_values = {sig: [] for sig in self.all_signals.keys()}  # Buffer for all signals
        self.lock = threading.Lock()
        self.clf_val = clf_val
        self.clf_aro = clf_aro
        self.redis = redis_client
        self.filtered_hr = []
        self.filtered_eda = []
        # Track 5 signals for Redis SD metrics
        self.edl_values = []  # Tonic EDA (EDL)
        self.temp_values = []  # Temperature
        self.scr_freq_values = []  # SCR Frequency
        self.hr_values = []  # Heart Rate
        self.ibi_values = []  # Inter-Beat Interval
        self.data_logger = DataLogger(source_id, signal_types=self.all_signals)
        # Print throttling to reduce terminal overhead
        self.start_time = time.time()
        self.last_print_time = 0
        self.print_interval = 30.0  # Print every 30 seconds after initial 3 seconds

    def update(self, col, value):
        """Update values for both prediction columns and all signals"""
        with self.lock:
            # Update prediction columns (EDA, HeartRate)
            if col in COLUMNS:
                self.values[col].append(float(value))
                max_samples = int(WINDOW_SECONDS * EMIT_RATE_HZ)
                for c in COLUMNS:
                    self.values[c] = self.values[c][-max_samples:]
            
            # Update all signals
            if col in self.all_signal_values:
                self.all_signal_values[col].append(float(value))
                max_samples = int(WINDOW_SECONDS * EMIT_RATE_HZ)
                self.all_signal_values[col] = self.all_signal_values[col][-max_samples:]
            
            # Track 5 specific signals for Redis SD metrics
            max_samples = int(WINDOW_SECONDS * EMIT_RATE_HZ)
            if col == "EDL":  # Tonic EDA
                self.edl_values.append(float(value))
                self.edl_values = self.edl_values[-max_samples:]
            elif col in ["Temperature0", "Temperature1"]:
                self.temp_values.append(float(value))
                self.temp_values = self.temp_values[-max_samples:]
            elif col == "SCRFrequency":
                self.scr_freq_values.append(float(value))
                self.scr_freq_values = self.scr_freq_values[-max_samples:]
            elif col == "HeartRate":
                self.hr_values.append(float(value))
                self.hr_values = self.hr_values[-max_samples:]
            elif col == "InterBeatInterval":
                self.ibi_values.append(float(value))
                self.ibi_values = self.ibi_values[-max_samples:]

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
            edl_sd = float(np.std(self.edl_values)) if len(self.edl_values) > 1 else None
            
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
            
            # Note: Performance metrics (processing_time_ms, sample counts) are only saved to CSV, not published to Redis
            
            # Publish consolidated metrics
            self.redis.publish(CHANNEL_PHYSIO.format(src=self.source_id), json.dumps(physio_metrics))

            # --- Redis publish: continuous val/arousal ---
            self.redis.publish(CHANNEL_VAL.format(src=self.source_id),
                               json.dumps({"device": self.source_id, "valence": float(val_cont), "timestamp": ts}))
            self.redis.publish(CHANNEL_ARO.format(src=self.source_id),
                               json.dumps({"device": self.source_id, "arousal": float(aro_cont), "timestamp": ts}))

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
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.source_id} {metrics_str}, Val={val_cont:.2f}, Aro={aro_cont:.2f}")
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


def discover_devices(adv_sock, timeout_s, own_ips=None):
    """Discover EmotiBit devices via HE/HH/EC/PO handshake on UDP 3131."""
    _IS_WINDOWS = platform.system() == 'Windows'
    if _IS_WINDOWS:
        import msvcrt

    def _enter_pressed():
        if not sys.stdin.isatty():
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

# ----------------------------- LIVE PLOT -----------------------------
import matplotlib
# Use platform-appropriate backend
if sys.platform == 'darwin':  # macOS
    matplotlib.use('MacOSX')
elif sys.platform == 'win32':  # Windows
    matplotlib.use('TkAgg')
else:  # Linux and other platforms
    matplotlib.use('TkAgg')
    
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_plot(devices):
    n = len(devices)
    fig, axes = plt.subplots(2, n, figsize=(5*n,7))
    if n==1: axes = np.array([[axes[0]],[axes[1]]])

    eda_lines, hr_lines = [], []

    for i, dev in enumerate(devices):
        l_eda, = axes[0][i].plot([], [], lw=2)
        l_hr,  = axes[1][i].plot([], [], lw=2)
        eda_lines.append(l_eda)
        hr_lines.append(l_hr)
        axes[0][i].set_title(f"EDA - {dev.source_id}")
        axes[1][i].set_title(f"HR - {dev.source_id}")
        axes[0][i].set_ylim(0, 5)
        axes[1][i].set_ylim(40, 140)

    def update(_):
        for i, dev in enumerate(devices):
            with dev.lock:
                eda = dev.filtered_eda
                hr = dev.filtered_hr
            if len(eda)>5:
                eda_lines[i].set_data(np.arange(len(eda)), eda)
                axes[0][i].set_xlim(0, len(eda))
            if len(hr)>5:
                hr_lines[i].set_data(np.arange(len(hr)), hr)
                axes[1][i].set_xlim(0, len(hr))
        return eda_lines + hr_lines

    def _on_key(event):
        if event.key == 'q':
            print("\n'q' pressed - shutting down gracefully...")
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', _on_key)
    fig.text(0.5, 0.01, "Press 'q' to quit", ha='center', fontsize=9, color='gray')

    ani = animation.FuncAnimation(fig, update, interval=500)
    return fig, ani

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

    # ---- NETWORK SOCKETS ----
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    if hasattr(socket, 'SO_REUSEPORT'):
        try:
            ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
    ctrl_sock.bind(("", EMOTIBIT_CONTROL_PORT))
    ctrl_sock.settimeout(1.0)

    data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, 'SO_REUSEPORT'):
        try:
            data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
    try:
        data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
    except OSError:
        pass
    data_sock.bind(("", EMOTIBIT_DATA_PORT))
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
            for dev_ip in discovered:
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
            for agg in agg_list:
                agg.process_window()
                agg.data_logger.flush()
            time.sleep(1.0)

    threading.Thread(target=prediction_loop, daemon=True).start()

    fig, ani = create_plot(agg_list)

    try:
        print("<¬ Starting live plot...")
        plt.show()
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
        for agg in agg_list:
            try:
                agg.data_logger.close()
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
