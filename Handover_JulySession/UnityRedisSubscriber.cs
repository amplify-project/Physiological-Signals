using System;
using UnityEngine;
using StackExchange.Redis; // Requires StackExchange.Redis NuGet package or DLL

/// <summary>
/// Subscribes to the AudiencePose Redis channel and exposes a live
/// crowd-average engagement float (0.0–1.0) for AR augmentation.
///
/// Setup:
///   1. Import StackExchange.Redis DLLs into Assets/Plugins/.
///   2. Attach this script to a GameObject (e.g. "EngagementManager").
///   3. Set redisHost to the IP of the machine running the Python inference
///      (use "localhost" if both run on the same device).
///
/// Usage in other scripts:
///   float score = FindObjectOfType<EngagementSubscriber>().currentEngagement;
/// </summary>
public class EngagementSubscriber : MonoBehaviour
{
    [Header("Redis Configuration")]
    [Tooltip("IP or hostname of the machine running the Python inference script")]
    public string redisHost = "localhost";

    [Tooltip("Redis server port (default 6379)")]
    public int redisPort = 6379;

    [Tooltip("Must match REDIS_CHANNEL in the Python script")]
    public string channelName = "engagement_score";

    [Header("Live Data (read-only at runtime)")]
    [Range(0f, 1f)]
    public float currentEngagement = 0.0f;

    /// <summary>Fires on the Unity main thread whenever a new score arrives.</summary>
    public event Action<float> OnEngagementUpdated;

    private ConnectionMultiplexer redis;
    private ISubscriber subscriber;

    // Thread-safe float buffer (written by Redis thread, read by Update)
    private volatile float _pendingScore = -1f;

    void Start()
    {
        ConnectToRedis();
    }

    void ConnectToRedis()
    {
        try
        {
            var config = ConfigurationOptions.Parse($"{redisHost}:{redisPort}");
            config.AbortOnConnectFail = false;   // Retry silently
            config.ConnectTimeout = 5000;

            redis = ConnectionMultiplexer.Connect(config);
            subscriber = redis.GetSubscriber();

            Debug.Log($"[Engagement] Connected to Redis at {redisHost}:{redisPort}");

            subscriber.Subscribe(channelName, (channel, message) =>
            {
                // Runs on a Redis background thread — do NOT call Unity APIs here.
                if (float.TryParse(message, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float score))
                {
                    _pendingScore = Mathf.Clamp01(score);
                }
            });

            Debug.Log($"[Engagement] Subscribed to channel: {channelName}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[Engagement] Redis connection failed: {e.Message}");
        }
    }

    void Update()
    {
        // Drain the latest value from the Redis thread onto the main thread.
        float pending = _pendingScore;
        if (pending >= 0f)
        {
            currentEngagement = pending;
            _pendingScore = -1f;
            OnEngagementUpdated?.Invoke(currentEngagement);
        }
    }

    void OnDestroy()
    {
        if (redis != null)
        {
            redis.Close();
            redis.Dispose();
        }
    }
}
