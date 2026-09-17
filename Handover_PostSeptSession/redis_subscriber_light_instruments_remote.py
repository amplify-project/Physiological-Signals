"""
Subscribe to Light Instruments data when Tom's laptop is running BOTH his
node-editor app AND its own Redis server -- i.e. his laptop is fully
self-contained, and this machine is a pure remote subscriber over the
network. This is a different topology from redis_subscriber_light_instruments.py
(which assumes a shared Redis on THIS machine that Tom's app publishes into).
Use whichever script matches how the two laptops are actually set up that day.

What Tom needs to do on his laptop for this to work:
  1. Run his bundled Redis server with network access enabled, e.g.
     redis-server.exe --bind 0.0.0.0 --protected-mode no
     (the default bind is 127.0.0.1-only, which only accepts connections
     from his own machine)
  2. Allow inbound TCP on his Redis port (6379 by default) through Windows
     Firewall.
  3. Point his app's Redis node at HIS OWN machine (localhost is fine on
     his end, since his app and his Redis are on the same laptop).
  4. Be on the same Wi-Fi/LAN as this machine, and give you his laptop's
     IP address (ipconfig -> IPv4 Address).

Then run this script with that address:
    python redis_subscriber_light_instruments_remote.py --host 192.168.1.23

Everything else (CSV output, columns, retry-safe against a slow consumer,
per-instrument value semantics) matches redis_subscriber_light_instruments.py
exactly -- see that script's docstring for the full data-format notes
(touch r/g/b ambiguity, channel name not yet confirmed, etc.). The one
addition here is connection retry: a two-machine link over real-world Wi-Fi
can drop briefly or not be ready the instant this script starts, so this
version reconnects automatically instead of crashing.

The CSV's `received_at` column (this machine's own clock at receive time)
is what to use for aligning against EmotiBit/camera/audio, which are also
timestamped on this machine's clock -- NOT the `timestamp` field, which is
stamped on Tom's laptop's clock and may be offset if the two machines'
clocks aren't synced.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import redis

from applog import setup_logging, install_excepthook
import logging as _logging

REDIS_PORT = 6379
REDIS_DB = 0
REDIS_CHANNEL = "amplify"  # unconfirmed for Sep -- see docstring

OUTPUT_DIR = "emotibit_recordings"  # shared with EmotiBit + audio recordings
CSV_FLUSH_INTERVAL_S = 2.0
RECONNECT_DELAY_S = 3.0


def connect(host, port):
    r = redis.Redis(host=host, port=port, db=REDIS_DB,
                     socket_connect_timeout=5, socket_timeout=None)
    r.ping()  # fail fast here rather than silently hanging in pubsub.listen()
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True,
                         help="Tom's laptop's network address, e.g. 192.168.1.23 -- "
                              "find it on his machine with 'ipconfig'. No default on purpose: "
                              "guessing localhost here would silently connect to nothing.")
    parser.add_argument("--port", type=int, default=REDIS_PORT)
    parser.add_argument("--channel", default=REDIS_CHANNEL,
                         help="Redis PubSub channel Tom's Redis node publishes to")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true",
                         help="print every event to console (off by default -- per-event console "
                              "I/O was found to cause dropped PubSub messages during rapid-fire bursts)")
    args = parser.parse_args()

    log = _logging.getLogger("light_instruments_subscriber_remote")
    setup_logging("light_instruments_subscriber_remote", extra_context={
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

    print(f"Connecting to Tom's Redis at {args.host}:{args.port} ...")
    log.info("Attempting connection to %s:%s", args.host, args.port)

    n_events = 0
    last_flush = time.time()

    try:
        while True:
            try:
                r = connect(args.host, args.port)
                pubsub = r.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(args.channel)
                print(f"Connected. Listening on channel '{args.channel}'.")
                print(f"Writing to {csv_path}")
                log.info("Connected, subscribed to channel=%s, logging to %s", args.channel, csv_path)

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

            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
                csv_file.flush()
                print(f"[WARN] Lost connection to {args.host}:{args.port} ({e}). "
                      f"Retrying in {RECONNECT_DELAY_S:.0f}s -- check Tom's laptop is on, "
                      f"his Redis is bound to 0.0.0.0, and both machines are on the same network.")
                log.warning("Connection error, will retry: %s", e)
                time.sleep(RECONNECT_DELAY_S)
    except KeyboardInterrupt:
        print(f"\nStopping -- {n_events} events logged to {csv_path}")
        log.info("Stopped by user, %d events logged", n_events)
    finally:
        csv_file.flush()
        csv_file.close()


if __name__ == "__main__":
    main()
