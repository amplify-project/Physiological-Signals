"""
Redis Publisher for Engagement Estimations
Publishes real-time engagement scores to Redis pub/sub for AMPLIFY Pilot 0 infrastructure.

Compatible with the amplify-project/pilot0-infrastructure Redis pub/sub architecture.
Publishes to channel: amplify.engagement
"""
import redis
import json
import time
import numpy as np
import os
from typing import Dict, Any, Optional
from datetime import datetime


class EngagementPublisher:
    """Publishes engagement estimations to Redis pub/sub (AMPLIFY infrastructure)."""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        channel: str = 'amplify.engagement'
    ):
        """
        Initialize Redis publisher for AMPLIFY infrastructure.
        
        Args:
            redis_url: Redis connection URL (e.g., 'redis://localhost:6379')
                      Falls back to REDIS_URL environment variable
            channel: Channel name for publishing engagement data
                    Default: 'amplify.engagement' (following AMPLIFY convention)
        """
        self.channel = channel
        
        # Get Redis URL from parameter or environment
        if redis_url is None:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        # Connect to Redis
        try:
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            print(f"✓ Connected to Redis: {redis_url}")
            print(f"✓ Publishing to channel: {channel}")
        except redis.ConnectionError as e:
            print(f"✗ Failed to connect to Redis: {e}")
            raise
    
    def publish_engagement(
        self,
        engagement: float,
        boredom: float,
        confusion: float,
        frustration: float,
        score: Optional[float] = None,
        video_id: Optional[str] = None
    ) -> bool:
        """
        Publish engagement estimation to Redis (AMPLIFY format).
        
        Message format matches AMPLIFY convention (similar to amplify.emotion):
        {
            "timestamp": <unix_timestamp>,
            "score": <confidence_score>,
            "engagement": <engagement_value>,
            "boredom": <boredom_value>,
            "confusion": <confusion_value>,
            "frustration": <frustration_value>
        }
        
        Args:
            engagement: Engagement score (0-3)
            boredom: Boredom score (0-3)
            confusion: Confusion score (0-3)
            frustration: Frustration score (0-3)
            score: Optional model confidence score (0-1)
            video_id: Optional video identifier (for logging only)
            
        Returns:
            True if published successfully, False otherwise
        """
        message = {
            'timestamp': time.time(),  # Unix timestamp for consistency with emotion channel
            'engagement': float(engagement),
            'boredom': float(boredom),
            'confusion': float(confusion),
            'frustration': float(frustration)
        }
        
        if score is not None:
            message['score'] = float(score)
        
        try:
            # Publish to Redis channel
            subscribers = self.redis_client.publish(self.channel, json.dumps(message))
            vid_str = f" ({video_id})" if video_id else ""
            print(f"Published to {subscribers} subscribers{vid_str}: engagement={engagement:.2f}")
            return True
        except Exception as e:
            print(f"Error publishing to Redis: {e}")
            return False
    
    def publish_batch(
        self,
        estimations: list[Dict[str, Any]]
    ) -> int:
        """
        Publish batch of engagement estimations.
        
        Args:
            estimations: List of estimation dictionaries
            
        Returns:
            Number of successfully published messages
        """
        success_count = 0
        for est in estimations:
            if self.publish_engagement(**est):
                success_count += 1
        return success_count
    
    def close(self):
        """Close Redis connection."""
        self.redis_client.close()
        print("✓ Redis connection closed")


class EngagementStream:
    """Real-time engagement estimation stream from video."""
    
    def __init__(
        self,
        model_path: str,
        publisher: EngagementPublisher,
        device: str = 'cuda'
    ):
        """
        Initialize engagement stream.
        
        Args:
            model_path: Path to trained model
            publisher: EngagementPublisher instance
            device: Device to run inference on
        """
        self.publisher = publisher
        self.device = device
        
        # TODO: Load model
        # self.model = load_model(model_path, device)
        print(f"TODO: Load model from {model_path}")
    
    def process_frame_batch(
        self,
        frames: np.ndarray,
        video_id: str
    ) -> Dict[str, float]:
        """
        Process batch of frames and return engagement estimation.
        
        Args:
            frames: Numpy array of pose features (T, keypoints, 3)
            video_id: Video identifier
            
        Returns:
            Dictionary with engagement scores
        """
        # TODO: Run inference
        # predictions = self.model.predict(frames)
        
        # Placeholder - replace with actual model inference
        predictions = {
            'engagement': 2.0,
            'boredom': 0.5,
            'confusion': 0.8,
            'frustration': 0.3
        }
        
        # Publish to Redis
        self.publisher.publish_engagement(
            video_id=video_id,
            **predictions,
            confidence=0.85
        )
        
        return predictions


def main():
    """Example usage."""
    # Initialize publisher (uses REDIS_URL from environment or defaults to localhost)
    publisher = EngagementPublisher(
        redis_url='redis://localhost:6379',
        channel='amplify.engagement'
    )
    
    # Example: Publish single estimation
    publisher.publish_engagement(
        engagement=2.5,
        boredom=0.3,
        confusion=0.5,
        frustration=0.2,
        score=0.92,
        video_id='test_video_001'
    )
    
    # Example: Publish batch
    estimations = [
        {
            'engagement': 2.0 + 0.1 * i,
            'boredom': 0.5,
            'confusion': 0.3,
            'frustration': 0.2,
            'score': 0.85,
            'video_id': f'video_{i}'
        }
        for i in range(5)
    ]
    
    published_count = publisher.publish_batch(estimations)
    print(f"Published {published_count}/{len(estimations)} messages")
    
    # Close connection
    publisher.close()


if __name__ == '__main__':
    main()
