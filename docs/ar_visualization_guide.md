# Real-Time Engagement AR Visualization

## Composite Metrics Explained

The model predicts 4 raw dimensions (0-3 scale):
- **Engagement**: Attention, interest, involvement
- **Confusion**: Struggling to understand, cognitive load
- **Frustration**: Emotional stress, irritation
- **Boredom**: Disengagement, lack of interest

These are combined into 3 **composite metrics** optimized for concert performance:

### 1. 🔥 Flow State (Primary Metric)
```
flow_state = engagement × (1 - confusion) × (1 - frustration)
```
**What it means**: Optimal audience experience - fully engaged, understanding, and enjoying without frustration.

**For musicians**: This is your **target metric**. High flow state = audience in the zone, loving the performance.

**Range**: 0-100%
- 90-100%: Audience in perfect flow ⭐⭐⭐⭐⭐
- 70-89%: High engagement, minor issues ⭐⭐⭐⭐
- 50-69%: Moderate engagement ⭐⭐⭐
- 30-49%: Low engagement, losing them ⭐⭐
- 0-29%: Not engaged ⭐

### 2. 👀 Attention Quality
```
attention = engagement × (1 - confusion)
```
**What it means**: Paying attention AND understanding what's happening.

**For musicians**: High attention but low flow = they're watching but something is bothering them (frustration).

**Range**: 0-100%

### 3. ⚡ Energy Level
```
energy = (engagement + frustration - boredom) / 2
```
**What it means**: Emotional intensity in the audience - are they excited or passive?

**For musicians**: 
- High energy + high flow = Excited, engaged audience 🔥
- High energy + low flow = Frustrated, stressed audience ⚠️
- Low energy = Bored, passive audience 😴

**Range**: 0-100%

## Smart Glasses AR Display Layout

```
┌─────────────────────────────────┐
│  AUDIENCE ENGAGEMENT            │
│                                 │
│  🔥 Flow State:      87%  ████▓ │  ← PRIMARY METRIC
│  👀 Attention:       92%  █████ │
│  ⚡ Energy:          78%  ████░ │
│                                 │
│  RAW METRICS:                   │
│    Engagement:     2.67         │  ← 0-3 scale
│    Confusion:      0.36         │
│    Frustration:    0.24         │
│    Boredom:        0.15         │
└─────────────────────────────────┘
```

## Usage

### 1. Test on existing video with extracted features:

```bash
python scripts/realtime_engagement_viz.py \
    --model ~/models/engagement_transformer/final_model.pth \
    --video ~/datasets/daisee/Test/1100011002.avi \
    --pose-features ~/datasets/daisee/features/Test/1100011002.npy \
    --output ~/results/engagement_viz_test.mp4
```

### 2. For live concert (future):

Would require:
- DWPose running in real-time on concert video feed
- Streaming pose keypoints to engagement model
- AR overlay sent to smart glasses display

**Processing pipeline**:
```
Concert Camera → DWPose Extraction → Pose Buffer (300 frames) 
              ↓
         Transformer Model → Engagement Metrics → AR Overlay → Smart Glasses
```

## Interpreting Results During Performance

**Scenario 1: Perfect Performance**
- Flow: 85-95%
- Attention: 90-95%
- Energy: 75-85%
- **Action**: Keep doing what you're doing! 🎉

**Scenario 2: Losing Them**
- Flow: 40-60%
- Attention: 45-65%
- Energy: 30-50%
- Boredom: 1.5-2.5
- **Action**: Change tempo, interact with audience, build energy

**Scenario 3: Confused Audience**
- Flow: 50-70%
- Attention: 80%+
- Confusion: 1.5-2.5
- **Action**: Simplify, explain, reduce complexity

**Scenario 4: Frustrated Audience**
- Flow: 30-50%
- Attention: 60-80%
- Frustration: 2.0+
- **Action**: Technical issue? Sound problems? Address immediately

**Scenario 5: High Energy, Low Flow**
- Flow: 40-60%
- Energy: 80%+
- Frustration: 1.5+
- **Action**: Audience is pumped but something is bothering them (sound, wait time, etc)

## Multi-Person Aggregation (Future Enhancement)

For concerts with multiple visible audience members:

1. **Spatial heatmap**: Show engagement zones in venue
2. **Mean metrics**: Average across all detected people
3. **Worst-case**: Show minimum flow state (find unhappy people)
4. **Best-case**: Show maximum flow state (find super fans)
5. **Distribution**: Show % of audience in each engagement range

Example multi-person overlay:
```
┌─────────────────────────────────┐
│  AUDIENCE ENGAGEMENT (N=47)    │
│                                 │
│  🔥 Avg Flow:        72%  ████░ │
│  📊 Distribution:               │
│     High (>70%):     68% ████▓  │
│     Med (40-70%):    28% ██▓░   │
│     Low (<40%):       4% ▓░     │
└─────────────────────────────────┘
```

## Performance Notes

- **Latency**: ~50ms inference time on GPU (real-time capable)
- **Buffer**: Needs 300 frames of history (~10 seconds at 30fps)
- **Smoothing**: 10-frame rolling average for stable display
- **Update rate**: Metrics updated every 5 frames to reduce jitter
