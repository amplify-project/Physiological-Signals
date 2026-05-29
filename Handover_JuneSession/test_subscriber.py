"""
Test Redis Subscriber — verify engagement_score channel is live.

Run this on the AR machine (or any machine on the same network) to confirm
that scores are arriving from the Python inference script.

Usage:
    python test_subscriber.py
    python test_subscriber.py --host 192.168.1.100   # remote server
"""

import redis
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Test subscriber for engagement_score channel")
    parser.add_argument("--host", type=str, default="localhost", help="Redis host (default: localhost)")
    parser.add_argument("--port", type=int, default=6379, help="Redis port (default: 6379)")
    parser.add_argument("--channel", type=str, default="engagement_score", help="Channel name")
    args = parser.parse_args()

    try:
        r = redis.Redis(host=args.host, port=args.port, db=0, decode_responses=True)
        r.ping()
        print(f"✅ Connected to Redis at {args.host}:{args.port}")
    except redis.ConnectionError as e:
        print(f"❌ Could not connect to Redis: {e}")
        return

    pubsub = r.pubsub()
    pubsub.subscribe(args.channel)

    print(f"📡 Subscribed to channel: '{args.channel}'")
    print(f"🔊 Listening for messages... (Press Ctrl+C to stop)\n")
    print("-" * 60)

    last_time = None

    for message in pubsub.listen():
        if message["type"] == "message":
            now = datetime.now()
            value = message["data"]

            if last_time:
                delta = (now - last_time).total_seconds()
                freq = f"(Δ {delta:.2f}s)"
            else:
                freq = "(first message)"

            # Colour-code the value: red < 0.3, yellow 0.3–0.6, green > 0.6
            try:
                score = float(value)
                if score > 0.6:
                    indicator = "🟢"
                elif score > 0.3:
                    indicator = "🟡"
                else:
                    indicator = "🔴"
            except ValueError:
                indicator = "❓"

            print(f"[{now.strftime('%H:%M:%S.%f')[:-3]}] {indicator} {value}  {freq}")
            last_time = now


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Subscriber stopped.")
