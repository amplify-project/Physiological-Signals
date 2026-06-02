import redis
import json
from pathlib import Path

# Shared async logger (writes to data/logs/subscriber/<ts>.log).
# In the handover folder applog.py sits beside this script.
import sys as _sys
_here = Path(__file__).resolve().parent
if str(_here) not in _sys.path:
    _sys.path.insert(0, str(_here))
from applog import setup_logging, install_excepthook  # noqa: E402
import logging as _logging
log = _logging.getLogger('subscriber')

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

setup_logging('subscriber', extra_context={
    'redis': f'{REDIS_HOST}:{REDIS_PORT}',
})
install_excepthook(log)

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
pubsub = r.pubsub(ignore_subscribe_messages=True)

# Subscribe to all device channels dynamically
pubsub.psubscribe(["device:*:hr_filtered",
                   "device:*:eda_filtered",
                   "device:*:physio_metrics",
                   "device:*:valence_cont",
                   "device:*:arousal_cont"])

print("Listening for all devices' physio metrics, filtered HR/EDA, Valence, and Arousal streams...")
log.info("Subscribed to device:*:{hr_filtered,eda_filtered,physio_metrics,valence_cont,arousal_cont}")
for message in pubsub.listen():
    try:
        channel = message['channel'].decode() if isinstance(message['channel'], bytes) else message['channel']
        data = json.loads(message['data'])
        print(f"[{channel}] {data}")
        log.debug("msg %s %s", channel, data)
    except Exception as e:
        print("[WARN] Could not parse message:", e)
        log.warning("Could not parse message: %s", e, exc_info=True)
