# Redis Integration for AMPLIFY Pilot 0

This module integrates the engagement estimation system with the AMPLIFY Pilot 0 infrastructure using Redis pub/sub.

## Architecture

The AMPLIFY Pilot 0 infrastructure uses Redis as a message broker for communication between input devices (data producers) and output devices (data consumers):

- **Input Devices**: Produce data and publish to Redis channels
  - `amplify.emotion` - Emotion recognition from webcam
  - `amplify.accelerometer` - Accelerometer data
  - `amplify.pressure` - Pressure sensor data
  - `amplify.trigger` - Sleeping mat status
  - **`amplify.engagement`** - Engagement estimations (this module)

- **Output Devices**: Subscribe to channels and visualize/act on data
  - Inscore visualization
  - LED strip control
  - Custom visualization tools

## Setup

### 1. Start Redis Server

Using Docker (recommended):

```bash
docker compose up
```

This starts Redis on `localhost:6379` (default port).

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and set your Redis URL:

```env
REDIS_URL=redis://localhost:6379
ENGAGEMENT_CHANNEL=amplify.engagement
```

### 3. Install Dependencies

```bash
pip install redis python-dotenv
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Publishing Engagement Estimations

```python
from src.dwpose_engagement.redis_publisher import EngagementPublisher

# Initialize publisher (reads REDIS_URL from environment)
publisher = EngagementPublisher()

# Publish single estimation
publisher.publish_engagement(
    engagement=2.5,
    boredom=0.3,
    confusion=0.5,
    frustration=0.2,
    score=0.92,  # Model confidence
    video_id='session_001'  # Optional, for logging
)

# Clean up
publisher.close()
```

### Message Format

Messages are published to `amplify.engagement` channel in JSON format:

```json
{
    "timestamp": 1699200000.123,
    "score": 0.92,
    "engagement": 2.5,
    "boredom": 0.3,
    "confusion": 0.5,
    "frustration": 0.2
}
```

This format is consistent with other AMPLIFY channels (e.g., `amplify.emotion`).

### Subscribing to Engagement Data

Other applications can subscribe to receive engagement updates:

```python
import redis
import json

r = redis.Redis.from_url('redis://localhost:6379')
pubsub = r.pubsub()
pubsub.subscribe('amplify.engagement')

for message in pubsub.listen():
    if message['type'] == 'message':
        data = json.loads(message['data'])
        print(f"Engagement: {data['engagement']:.2f}, Score: {data['score']:.2f}")
```

## Real-time Streaming

For real-time webcam processing with MediaPipe:

```python
from src.dwpose_engagement.redis_publisher import EngagementStream

# Initialize stream (TODO: implement model loading)
stream = EngagementStream(
    model_path='models/best_model.pth',
    publisher=publisher,
    device='cuda'
)

# Process video frames
# stream.process_frame_batch(pose_features, video_id='live_session')
```

## Integration with AMPLIFY Infrastructure

### Channels

- Input: `amplify.engagement` (this module publishes here)
- Related: `amplify.emotion`, `amplify.accelerometer`, `amplify.pressure`

### Compatible Output Devices

Output devices can subscribe to `amplify.engagement` to:
- Visualize engagement levels in real-time
- Trigger alerts for low engagement
- Control ambient lighting based on engagement
- Log engagement data for analysis

### Example Visualization Integration

```python
# In an output device (e.g., emotion-receiver style)
import redis
import json

r = redis.Redis.from_url('redis://localhost:6379')
pubsub = r.pubsub()
pubsub.subscribe('amplify.engagement', 'amplify.emotion')

for message in pubsub.listen():
    if message['channel'] == 'amplify.engagement':
        data = json.loads(message['data'])
        # Send to visualization via OSC, WebSocket, etc.
        update_visualization(data)
```

## Testing

Test the publisher with mock data:

```bash
python src/dwpose_engagement/redis_publisher.py
```

This runs the example in `main()` which publishes test data to Redis.

## Troubleshooting

### Cannot connect to Redis

```
Error: Failed to connect to Redis: Error 111 connecting to localhost:6379. Connection refused.
```

**Solution**: Make sure Redis is running:
```bash
docker compose up
```

### Channel not receiving messages

**Check subscribers**:
```bash
docker exec -it <redis_container> redis-cli
> PUBSUB CHANNELS amplify.*
```

### Environment variable not found

**Solution**: Create `.env` file from `.env.example` and set `REDIS_URL`.

## References

- [AMPLIFY Pilot 0 Infrastructure](https://github.com/amplify-project/pilot0-infrastructure)
- [Redis Pub/Sub Documentation](https://redis.io/docs/manual/pubsub/)
- Original emotion recognition: `pilot0-infrastructure/input/emotion-recognition/`
