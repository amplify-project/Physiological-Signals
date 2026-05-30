"""
Redis Subscriber - Echo Engagement Predictions to Console
Subscribes to AMPLIFY engagement channel and displays predictions in real-time

Usage:
    python scripts/console_subscriber.py
"""

import redis
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class EngagementSubscriber:
    """Subscribe to engagement predictions from Redis."""
    
    def __init__(self, redis_url: str = None, channel: str = 'amplify.engagement'):
        """
        Initialize engagement subscriber.
        
        Args:
            redis_url: Redis connection URL (default: from .env or localhost)
            channel: Redis channel to subscribe to
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.channel = channel
        
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.pubsub = self.redis_client.pubsub()
            print(f"✓ Connected to Redis at {self.redis_url}")
        except Exception as e:
            print(f"✗ Failed to connect to Redis: {e}")
            raise
    
    def format_prediction(self, data: dict) -> str:
        """
        Format prediction data for display.
        
        Args:
            data: Prediction dictionary
            
        Returns:
            Formatted string
        """
        timestamp = datetime.fromtimestamp(data['timestamp']).strftime('%H:%M:%S')
        
        lines = [
            "─" * 60,
            f"[{timestamp}] Engagement Prediction",
            "─" * 60,
        ]
        
        # Main predictions
        if 'engagement' in data:
            lines.append(f"  Engagement:   {data['engagement']:.2f}")
        if 'boredom' in data:
            lines.append(f"  Boredom:      {data['boredom']:.2f}")
        if 'confusion' in data:
            lines.append(f"  Confusion:    {data['confusion']:.2f}")
        if 'frustration' in data:
            lines.append(f"  Frustration:  {data['frustration']:.2f}")
        
        # Confidence/score
        if 'score' in data:
            lines.append(f"  Confidence:   {data['score']:.2%}")
        
        # Metadata
        if 'metadata' in data:
            metadata = data['metadata']
            lines.append("  " + "─" * 56)
            for key, value in metadata.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def listen(self):
        """Listen for engagement predictions and echo to console."""
        print(f"\n{'='*60}")
        print(f"Listening on channel: {self.channel}")
        print(f"{'='*60}")
        print("Waiting for predictions... (Press Ctrl+C to stop)\n")
        
        # Subscribe to channel
        self.pubsub.subscribe(self.channel)
        
        try:
            for message in self.pubsub.listen():
                if message['type'] == 'message':
                    try:
                        # Parse message
                        data = json.loads(message['data'])
                        
                        # Display formatted prediction
                        print(self.format_prediction(data))
                        print()  # Blank line between predictions
                        
                    except json.JSONDecodeError as e:
                        print(f"✗ Error decoding message: {e}")
                    except Exception as e:
                        print(f"✗ Error processing message: {e}")
        
        except KeyboardInterrupt:
            print("\n\n✓ Stopped listening")
        finally:
            self.pubsub.unsubscribe(self.channel)
            self.redis_client.close()


def main():
    """Run engagement subscriber."""
    try:
        subscriber = EngagementSubscriber()
        subscriber.listen()
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
