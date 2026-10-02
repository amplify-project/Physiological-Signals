"""
LIGHT INSTRUMENTS SIMULATOR -- replays Tom's recorded sample CSVs
(light_instruments_sample_data/{drum,maracas,rainstick,keys,touch}.csv) onto
Redis at their original pacing, so the full pipeline (EmotiBit, camera/
engagement, audio, light instruments) can be integration-tested end-to-end,
all streams timestamp-synced, before real hardware or Tom's own laptop is
available.

DEV/TEST TOOL ONLY. On the real day, real light instruments publish through
Tom's node-editor app (Serial Input -> ... -> Redis PubSub node) directly --
this script is not part of that path and isn't needed once his app is
running (whether on this same machine or his own laptop).
`redis_subscriber_light_instruments.py` (the CSV logger) is agnostic to
which of the two is publishing; it just subscribes to the channel either
way, so it's the same subscriber for both this simulator and the real app.

Each sample CSV's rows are `timestamp,device,port,value` -- the *relative*
gaps between a file's own timestamps are preserved on replay (so a real
maracas shake pattern still sounds/looks like a real maracas shake pattern),
but the absolute times are re-anchored to "now" at the moment this script
starts, matching how simulate_emotibits.py and Tom's own GUI "Simulate" node
both re-anchor to current time rather than replaying original clock time.

Published message shape matches the real RedisNode.tsx exactly:
    {"device": "<name>", "port": "<id>", "value": <int>, "timestamp": <unix float>}
(the "timestamp" here is the publish-time clock, i.e. this script's replay
time -- NOT the original sample file's timestamp column, same as the real
node editor's Redis node behaviour; see redis_subscriber_light_instruments.py's
docstring.)

Usage:
    python simulate_light_instruments.py                     # replay all 5 sample files concurrently, once, at 1x speed
    python simulate_light_instruments.py --instruments maracas,drum
    python simulate_light_instruments.py --speed 4            # 4x faster (compress waiting time for quick tests)
    python simulate_light_instruments.py --loop               # repeat indefinitely
"""
import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

import redis

SAMPLE_DIR = Path(__file__).resolve().parent / "light_instruments_sample_data"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_CHANNEL = "amplify"  # unconfirmed for the real Sep graph -- see redis_subscriber_light_instruments.py

ALL_INSTRUMENTS = ["drum", "maracas", "rainstick", "keys", "touch"]


def load_events(csv_path):
    """Return [(offset_s_from_first_row, device, port, value), ...]."""
    rows = []
    with open(csv_path, newline="") as f:
        for line in csv.reader(f):
            if len(line) != 4:
                continue
            ts_str, device, port, value_str = line
            try:
                rows.append((float(ts_str), device, port, int(value_str)))
            except ValueError:
                continue
    if not rows:
        return []
    t0 = rows[0][0]
    return [(ts - t0, device, port, value) for ts, device, port, value in rows]


def replay_one(r, channel, name, events, speed, loop, stop_event, replay_start):
    if not events:
        print(f"[{name}] no events to replay, skipping")
        return
    n_published = 0
    while not stop_event.is_set():
        for offset_s, device, port, value in events:
            if stop_event.is_set():
                break
            due = replay_start + offset_s / speed
            now = time.time()
            if due > now:
                stop_event.wait(due - now)
            payload = {"device": device, "port": port, "value": value, "timestamp": time.time()}
            r.publish(channel, json.dumps(payload))
            n_published += 1
        if not loop:
            break
        replay_start = time.time()  # re-anchor for the next lap
    print(f"[{name}] done -- {n_published} events published")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=REDIS_HOST)
    parser.add_argument("--port", type=int, default=REDIS_PORT)
    parser.add_argument("--channel", default=REDIS_CHANNEL)
    parser.add_argument("--instruments", default=",".join(ALL_INSTRUMENTS),
                         help="comma-separated subset of: " + ",".join(ALL_INSTRUMENTS))
    parser.add_argument("--sample-dir", default=str(SAMPLE_DIR))
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (2.0 = 2x faster)")
    parser.add_argument("--loop", action="store_true", help="repeat each instrument's file indefinitely")
    args = parser.parse_args()

    instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]
    unknown = set(instruments) - set(ALL_INSTRUMENTS)
    if unknown:
        print(f"[ERROR] unknown instrument(s): {sorted(unknown)} -- choose from {ALL_INSTRUMENTS}")
        sys.exit(1)

    stop_event = threading.Event()
    replay_start = time.time()

    threads = []
    for name in instruments:
        csv_path = Path(args.sample_dir) / f"{name}.csv"
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found, skipping {name}")
            continue
        events = load_events(csv_path)
        print(f"[{name}] loaded {len(events)} events from {csv_path.name}, "
              f"spans {events[-1][0]:.1f}s" if events else f"[{name}] empty file")
        # Each thread gets its own connection -- a single redis.Redis() object
        # is not safe for concurrent .publish() calls from multiple threads
        # (confirmed by testing: sharing one connection measurably increased
        # drop rate on top of the baseline PubSub loss, see
        # redis_subscriber_light_instruments.py).
        thread_r = redis.Redis(host=args.host, port=args.port, db=0)
        t = threading.Thread(target=replay_one, args=(thread_r, args.channel, name, events, args.speed, args.loop,
                                                        stop_event, replay_start), daemon=True)
        threads.append(t)

    print(f"\nPublishing to Redis channel '{args.channel}' ({args.host}:{args.port}), speed={args.speed}x"
          + (", looping" if args.loop else ""))
    print("Ctrl+C to stop.\n")

    for t in threads:
        t.start()

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        for t in threads:
            t.join(timeout=2)


if __name__ == "__main__":
    main()
