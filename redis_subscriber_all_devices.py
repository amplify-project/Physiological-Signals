import redis
import json

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
pubsub = r.pubsub(ignore_subscribe_messages=True)

# Subscribe to all device channels dynamically
pubsub.psubscribe(["device:*:hr_filtered",
                   "device:*:eda_filtered",
                   "device:*:valence_cont",
                   "device:*:arousal_cont"])

print("Listening for all devices' HR filtered, EDA filtered, Valence, and Arousal streams...")
for message in pubsub.listen():
    try:
        channel = message['channel'].decode() if isinstance(message['channel'], bytes) else message['channel']
        data = json.loads(message['data'])
        print(f"[{channel}] {data}")
    except Exception as e:
        print("[WARN] Could not parse message:", e)
