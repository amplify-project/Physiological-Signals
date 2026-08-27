"""
Subscribe to the Light Instruments' Redis PubSub channel and log every event
to a timestamped CSV in emotibit_recordings/, the same folder and role
audio_monitor_csv.py plays for the audio-detection stream and
multiemotibit_UDP_SD_RFv2.py plays for EmotiBit data -- all three sources
land in one shared recordings folder so they can be aligned by timestamp.

Verified end-to-end against a real redis-server + Tom's sample data via
`simulate_light_instruments.py`: Redis PubSub has no delivery guarantee for
a busy subscriber (unlike Streams/lists, an unread message is not buffered
for a slow consumer). Per-event `print()` to console was enough added
latency in the receive loop to silently drop messages during genuine
rapid-fire bursts -- confirmed even at real (1x, unsped-up) replay of
`keys.csv`, whose original hardware timestamps include presses as close as
~6-8ms apart. Fixed by making console output opt-in (`--verbose`) rather
than the default -- confirm zero loss again if this loop grows any other
per-event work (extra logging, a second write target, etc.).

Upstream source: Tom's light-instrument-node-editor app. A "Serial Input"
node (or "Simulate" node when replaying a sample CSV) reads each physical
instrument's raw events, an optional processing chain runs, and a
"Redis PubSub" node publishes to a channel -- payload is whatever JSON
object reaches that node's input, with a `timestamp` field (Unix seconds,
added at publish time on Tom's machine -- NOT the original event time)
merged in before JSON.stringify.

The message actually received here:
    {"device": "<instrument-name>", "port": "<port-id>", "value": <int>,
     "timestamp": <unix-seconds-float-at-publish-time-on-TOM'S-clock>}

This script additionally records `received_at`, a timestamp on THIS
machine's own clock at the moment the message arrives -- if this subscriber
runs on the same PC as the EmotiBit/camera/audio capture, `received_at` is
directly comparable to their timestamps without depending on Tom's laptop
clock being in sync. Prefer `received_at` for cross-modal alignment; use
`timestamp` only if Tom's app is confirmed to run on this same machine (in
which case the two columns should already agree).

Per-instrument value semantics (from Tom's email):
  - maracas: value always 1 (event = shake)
  - rainstick: value 0-1023 (IR-barrier interruption amount, treat as event)
  - keys: value 0/1 (press/release), one of 3 ports per key
  - touch: value 0-1023 (contact amount) -- ports observed as "r"/"g"/"b" in
    the sample data, NOT "one port per touch surface" as the email describes;
    unconfirmed whether that's 3 colour-coded touch zones or something else
    -- flag to Tom before trusting `touch` port semantics in analysis.
  - drum: value 0-1023 (strike/vibration strength) -- sample data's own
    `device` field says "percussion_small", not "drum".

NOT YET CONFIRMED for the real September deployment: the actual Redis
channel name. The node editor's own default is "amplify" (RedisNode.tsx,
`data.channel ?? "amplify"`), used below as a placeholder -- confirm the
live graph's actual Redis node configuration (and Host, if Tom's app runs
on a different machine than this script) before relying on this.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import redis

from applog import setup_logging, install_excepthook
import logging as _logging

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_CHANNEL = "amplify"  # unconfirmed for Sep -- see module docstring

OUTPUT_DIR = "emotibit_recordings"  # shared with EmotiBit + audio recordings
CSV_FLUSH_INTERVAL_S = 2.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=REDIS_HOST,
                         help="Redis host -- if Tom's app runs on his own laptop, this is that laptop's "
                              "network address, not localhost")
    parser.add_argument("--port", type=int, default=REDIS_PORT)
    parser.add_argument("--channel", default=REDIS_CHANNEL,
                         help="Redis PubSub channel the node editor's Redis node publishes to")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true",
                         help="print every event to console (off by default -- see module docstring: "
                              "per-event console I/O was found to cause dropped PubSub messages during "
                              "rapid-fire bursts, e.g. a real double key-press only ~7ms apart)")
    args = parser.parse_args()

    log = _logging.getLogger("light_instruments_subscriber")
    setup_logging("light_instruments_subscriber", extra_context={
        "redis": f"{args.host}:{args.port}",
        "channel": args.channel,
    })
    install_excepthook(log)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_label = time.strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = out_dir / f"light_instruments_{ts_label}.csv"

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "device", "port", "value", "received_at"])
    csv_file.flush()

    r = redis.Redis(host=args.host, port=args.port, db=REDIS_DB)
    pubsub = r.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(args.channel)

    print(f"Listening on Redis channel '{args.channel}' ({args.host}:{args.port})")
    print(f"Writing to {csv_path}")
    log.info("Subscribed to channel=%s, logging to %s", args.channel, csv_path)

    last_flush = time.time()
    n_events = 0
    try:
        for message in pubsub.listen():
            try:
                raw = message["data"]
                data = json.loads(raw)
            except Exception as e:
                print("[WARN] Could not parse message:", e)
                log.warning("Could not parse message: %s", e, exc_info=True)
                continue

            device = data.get("device")
            port = data.get("port")
            value = data.get("value")
            evt_timestamp = data.get("timestamp")
            received_at = time.time()

            if device is None or port is None or value is None:
                print(f"[WARN] Unexpected payload shape, logging raw: {data}")
                log.warning("Unexpected payload shape: %s", data)

            csv_writer.writerow([evt_timestamp, device, port, value, received_at])
            n_events += 1

            if args.verbose:
                print(f"[{device}:{port}] {value}")

            now = time.time()
            if now - last_flush >= CSV_FLUSH_INTERVAL_S:
                csv_file.flush()
                last_flush = now
    except KeyboardInterrupt:
        print(f"\nStopping -- {n_events} events logged to {csv_path}")
        log.info("Stopped by user, %d events logged", n_events)
    finally:
        csv_file.flush()
        csv_file.close()


if __name__ == "__main__":
    main()
