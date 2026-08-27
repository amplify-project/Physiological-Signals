# ===============================================================
# EMOTIBIT SIMULATOR  --  fake physio publisher for GUI testing
# ---------------------------------------------------------------
# Publishes the same Redis messages the real publisher
# (multiemotibit_UDP_SD_RFv2.py) emits, for N synthetic devices,
# so you can see how the engagement GUI sidebar looks with many
# wearers (e.g. 8) without owning that many EmotiBits.
#
# It writes to the exact channels the GUI subscribes to:
#     device:<serial>:physio_metrics   (JSON, ~1 Hz)
#     device:<serial>:hr_filtered      (JSON, raw HR for the deque)
#     device:<serial>:eda_filtered     (JSON, raw EDA for the deque)
#
# Usage (inside the venv):
#     python simulate_emotibits.py                # 8 fake devices @ 1 Hz
#     python simulate_emotibits.py --count 8      # explicit
#     python simulate_emotibits.py --calibrate 8  # 8 s calibrating first
#
# Run it in its own terminal, alongside (or instead of) the real
# EmotiBit publisher, then start the engagement app. Ctrl+C to stop.
# ===============================================================

import argparse
import json
import random
import signal
import sys
import time

import redis


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class FakeWearer:
    """One synthetic EmotiBit producing smoothly drifting z-scores."""

    def __init__(self, serial):
        self.serial = serial
        # Mean-reverting random walks for the three plotted traces.
        self.hr_z = random.uniform(-0.5, 0.5)
        self.eda_z = random.uniform(-0.5, 0.5)
        self.temp_z = random.uniform(-0.5, 0.5)
        # Absolute values (for the raw HR/EDA deques + realism).
        self.hr_bpm = random.uniform(65, 95)
        self.eda_us = random.uniform(2.0, 8.0)
        # Occasional off-wrist demo state.
        self.off_wrist = False
        self._off_until = 0.0

    def _step(self, z, drift=0.18):
        """Ornstein-Uhlenbeck-ish step: revert to 0, add noise, clip to +/-3.2."""
        z += -0.05 * z + random.gauss(0, drift)
        return _clamp(z, -3.2, 3.2)

    def update(self, now, allow_off_wrist):
        self.hr_z = self._step(self.hr_z)
        self.eda_z = self._step(self.eda_z, 0.22)
        self.temp_z = self._step(self.temp_z, 0.10)
        # Track loose absolute values off the z drift for the raw deques.
        self.hr_bpm = _clamp(self.hr_bpm + self.hr_z * 0.6 + random.gauss(0, 0.4), 45, 180)
        self.eda_us = _clamp(self.eda_us + self.eda_z * 0.05 + random.gauss(0, 0.05), 0.1, 40)

        # Randomly drop one device off-wrist for a few seconds to show the
        # OFF-WRIST watermark + spline gapping behaviour in the GUI.
        if allow_off_wrist:
            if self.off_wrist and now >= self._off_until:
                self.off_wrist = False
            elif not self.off_wrist and random.random() < 0.002:
                self.off_wrist = True
                self._off_until = now + random.uniform(4, 9)

    def physio_payload(self, ts, calibrating, calib_remaining, session_age, baseline_n):
        """Mirror the real publisher's physio_metrics dict."""
        p = {
            "device": self.serial,
            "timestamp": ts,
            # Standard deviations (small positive numbers, as the real device sends).
            "edl_sd": round(random.uniform(0.02, 0.12), 4),
            "eda_sd": round(random.uniform(0.02, 0.12), 4),
            "temperature_roc_sd": round(random.uniform(0.005, 0.05), 4),
            "scr_frequency_sd": round(random.uniform(0.01, 0.2), 4),
            "hr_sd": round(random.uniform(0.5, 3.0), 4),
            "ibi_sd": round(random.uniform(10, 40), 4),
            "calibrating": bool(calibrating),
            "calibration_remaining_s": round(calib_remaining, 1),
            "session_age_s": round(session_age, 1),
            "baseline_n": int(baseline_n),
            "off_wrist": bool(self.off_wrist),
            "quality": {
                "hr": 1.0,
                "eda": 1.0,
                "ibi": 1.0,
                "temperature_roc": 1.0,
                "scr_frequency": 1.0,
            },
        }
        # The real publisher OMITS all *_z keys while calibrating or off-wrist.
        if not calibrating and not self.off_wrist:
            p["hr_z"] = round(self.hr_z, 3)
            p["eda_z"] = round(self.eda_z, 3)
            p["ibi_z"] = round(-self.hr_z + random.gauss(0, 0.15), 3)
            p["temperature_roc_z"] = round(self.temp_z, 3)
            p["scr_frequency_z"] = round(_clamp(self.eda_z + random.gauss(0, 0.3), -3.2, 3.2), 3)
        return p


def main():
    ap = argparse.ArgumentParser(description="Simulate N EmotiBit devices publishing to Redis.")
    ap.add_argument("--count", type=int, default=8, help="Number of fake devices (default 8).")
    ap.add_argument("--rate", type=float, default=1.0, help="Publish rate in Hz (default 1.0).")
    ap.add_argument("--host", default="localhost", help="Redis host (default localhost).")
    ap.add_argument("--port", type=int, default=6379, help="Redis port (default 6379).")
    ap.add_argument("--prefix", default="MD-V5-SIM", help="Serial prefix (default MD-V5-SIM).")
    ap.add_argument("--calibrate", type=float, default=6.0,
                    help="Seconds of 'calibrating' before z-scores stream (default 6).")
    ap.add_argument("--no-off-wrist", action="store_true",
                    help="Disable the random off-wrist demo events.")
    args = ap.parse_args()

    try:
        r = redis.Redis(host=args.host, port=args.port, db=0)
        r.ping()
    except Exception as e:
        print(f"ERROR: could not connect to Redis at {args.host}:{args.port} -- is it running? ({e})")
        sys.exit(1)

    serials = [f"{args.prefix}{i:04d}" for i in range(1, args.count + 1)]
    wearers = [FakeWearer(s) for s in serials]

    print(f"Simulating {args.count} EmotiBit device(s) -> Redis {args.host}:{args.port}")
    print("  Serials: " + ", ".join(serials))
    print(f"  Rate: {args.rate} Hz   Calibration: {args.calibrate}s")
    print("  Press Ctrl+C to stop.\n")

    running = {"go": True}

    def _stop(*_):
        running["go"] = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    start = time.time()
    period = 1.0 / max(args.rate, 0.1)
    tick = 0
    while running["go"]:
        now = time.time()
        ts = now
        session_age = now - start
        calibrating = session_age < args.calibrate
        calib_remaining = max(0.0, args.calibrate - session_age)
        baseline_n = min(180, int(session_age))

        for w in wearers:
            w.update(now, allow_off_wrist=not args.no_off_wrist)
            # Raw filtered channels feed the GUI's rolling deques.
            r.publish(f"device:{w.serial}:hr_filtered",
                      json.dumps({"device": w.serial, "timestamp": ts,
                                  "HR_filtered": round(w.hr_bpm, 2)}))
            r.publish(f"device:{w.serial}:eda_filtered",
                      json.dumps({"device": w.serial, "timestamp": ts,
                                  "EDA_filtered": round(w.eda_us, 3)}))
            # Consolidated metrics (drives the z-score header + spline panel).
            r.publish(f"device:{w.serial}:physio_metrics",
                      json.dumps(w.physio_payload(ts, calibrating, calib_remaining,
                                                  session_age, baseline_n)))

        tick += 1
        if tick % 5 == 0:
            state = "calibrating" if calibrating else "streaming z-scores"
            print(f"[{time.strftime('%H:%M:%S')}] tick {tick}: {len(wearers)} devices ({state})")

        # Keep a steady cadence.
        sleep_for = period - (time.time() - now)
        if sleep_for > 0:
            time.sleep(sleep_for)

    print("\nStopped. Simulated devices will disappear from the GUI after their panels time out.")


if __name__ == "__main__":
    main()
