# multiemotibit_UDP_SD - Technical Documentation

**Scripts**: 
- `multiemotibit_UDP_SD.py` - Original version with RF models v1
- `multiemotibit_UDP_SD_RFv2.py` - Updated version with RF models v2 (sklearn 1.3.0+)

**Date**: May 27, 2026  
**Version**: SD Metrics Implementation with Latency Tracking

## Overview
This document explains the physiological metrics sent to Redis, performance monitoring features, and the changes made to the multiemotibit_UDP_SD scripts. This document is maintained and updated as changes are made to the scripts.

---

## Script Versions

### multiemotibit_UDP_SD.py (Original)
- **RF Models**: `rf_valence_full.pkl`, `rf_arousal_full.pkl`
- **Compatible with**: Python 3.7-3.9, older scikit-learn versions
- **Use case**: Legacy systems with older Python/sklearn installations

### multiemotibit_UDP_SD_RFv2.py (Updated - Recommended)
- **RF Models**: `rf_valence_full_v2.pkl`, `rf_arousal_full_v2.pkl`
- **Compatible with**: Python 3.9-3.12, scikit-learn 1.3.0+
- **Use case**: New deployments, modern Python environments
- **Advantages**: 
  - Updated dependencies for better compatibility
  - Same functionality as original version
  - Better long-term support

**Note**: Both scripts have identical functionality (SD metrics, latency tracking, Redis publishing, CSV storage). The only difference is the RF model files they load.

---

## RF Model Information

### Model Training
Both model versions use the same Random Forest architecture trained on the same dataset:
- **Features**: EDA and Heart Rate derived metrics (HRV, temporal features, statistical features)
- **Output**: Valence and Arousal predictions (continuous scale 0-2)
- **Window**: 5-second windows at 25 Hz sampling rate

### Model Files

| Model | File Name | sklearn Version | Python Version | Status |
|-------|-----------|----------------|----------------|--------|
| V1 - Valence | `rf_valence_full.pkl` | <1.3.0 | 3.7-3.9 | Legacy |
| V1 - Arousal | `rf_arousal_full.pkl` | <1.3.0 | 3.7-3.9 | Legacy |
| V2 - Valence | `rf_valence_full_v2.pkl` | ≥1.3.0 | 3.9-3.12 | **Recommended** |
| V2 - Arousal | `rf_arousal_full_v2.pkl` | ≥1.3.0 | 3.9-3.12 | **Recommended** |

### Choosing the Right Version

**Use multiemotibit_UDP_SD_RFv2.py if:**
- ✅ New installation
- ✅ Python 3.9 or newer
- ✅ Modern package versions
- ✅ Sending to clients (better long-term support)

**Use multiemotibit_UDP_SD.py if:**
- ⚠️ Existing system with Python 3.7-3.8
- ⚠️ Cannot upgrade scikit-learn
- ⚠️ Must maintain compatibility with legacy systems

### Package Dependencies

**For multiemotibit_UDP_SD_RFv2.py (Recommended):**
```
Python: 3.9, 3.10, 3.11, or 3.12
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
scipy>=1.10.0,<2.0.0
scikit-learn>=1.3.0,<2.0.0
joblib>=1.3.0
redis>=4.5.0,<6.0.0
matplotlib>=3.7.0,<4.0.0
netifaces>=0.11.0 (optional)
```

**For multiemotibit_UDP_SD.py (Legacy):**
```
Python: 3.7, 3.8, or 3.9
numpy>=1.19.0,<1.24.0
pandas>=1.2.0,<2.0.0
scipy>=1.6.0,<1.10.0
scikit-learn>=0.24.0,<1.3.0
joblib>=1.0.0
redis>=3.5.0
matplotlib>=3.3.0
netifaces>=0.11.0 (optional)
```

**Installation:**
```bash
# For RFv2 (recommended)
pip install -r requirements.txt

# Or manually
pip install numpy>=1.24.0 pandas>=2.0.0 scipy>=1.10.0 scikit-learn>=1.3.0 joblib>=1.3.0 redis>=4.5.0 matplotlib>=3.7.0
```

**See Also**: `INSTALLATION_GUIDE.md` for complete setup instructions

---

## Changes Made from Previous Version

### Previous Implementation
- **Redis Channels**: `device:{src}:hr_filtered` and `device:{src}:eda_filtered`
- **Data Published**: Raw filtered HR and EDA values (5 samples per window)
- **Message Frequency**: 10 messages per window (5 HR + 5 EDA)
- **Metrics**: Instantaneous filtered values

### Current Implementation
- **Redis Channel**: `device:{src}:physio_metrics` (single consolidated channel)
- **Data Published**: Standard deviations of 5 physiological signals
- **Message Frequency**: 1 message per window
- **Metrics**: Variability measures over 5-second windows

### Why These Changes?
1. **More Informative**: Standard deviations capture **variability** rather than instantaneous values
2. **Better for ML/Analysis**: Variability metrics are stronger indicators of emotional/physiological states
3. **Reduced Network Load**: 1 message vs 10 messages per window
4. **Complementary Information**: Five different aspects of physiological response
5. **Research-Backed**: These metrics are commonly used in affective computing and stress detection

---

## Five Physiological Metrics Explained

**Note**: All features and metrics described in this document apply to both script versions (original and RFv2). The only difference between versions is the RF model files used for emotion prediction.

### 1. **Standard Deviation of Tonic EDA (EDL)**
**Metric Name**: `edl_sd`

**What it is:**
- EDL (Electrodermal Level) = the slow-moving baseline component of skin conductance
- Represents general arousal state
- Already decomposed by EmotiBit from raw EDA

**What SD measures:**
- How much the baseline arousal level varies over the 5-second window
- Low SD = stable arousal state
- High SD = fluctuating arousal level

**Example:**
- During calm focus: Low edl_sd (steady baseline)
- During anticipation: High edl_sd (baseline drifting up)

---

### 2. **Standard Deviation of Temperature Rate of Change**
**Metric Name**: `temperature_roc_sd`

**What it is:**
- Rate of Change (ROC) = how fast temperature is changing (°C per second)
- Calculated using: `np.diff(temperature_values) * sampling_rate`

**What SD measures:**
- Variability in the speed of temperature change
- Low SD = temperature changing at constant rate (or not at all)
- High SD = temperature fluctuating rapidly

**Example:**
- During rest: Low temperature_roc_sd (stable temp)
- During stress response: High temperature_roc_sd (rapid fluctuations)

**Why ROC instead of temperature:**
- Temperature itself changes very slowly
- Rate of change is more sensitive to acute physiological responses
- Better for detecting moment-to-moment emotional changes

---

### 3. **Standard Deviation of SCR Frequency**
**Metric Name**: `scr_frequency_sd`

**What it is:**
- SCR (Skin Conductance Response) = fast peaks in EDA
- SCR Frequency = number of responses per unit time
- Already calculated by EmotiBit

**What SD measures:**
- How variable the rate of SCR events is
- Low SD = responses happening at steady rate
- High SD = bursts of activity alternating with calm

**Example:**
- During sustained attention: Low scr_frequency_sd (steady responses)
- During startle/surprise: High scr_frequency_sd (bursts then calm)

**Distinction from Tonic EDA:**
- Tonic EDA (EDL) = slow baseline drift (minutes)
- SCR Frequency = fast response events (seconds)
- They capture different aspects of electrodermal activity

---

### 4. **Standard Deviation of Heart Rate**
**Metric Name**: `hr_sd`

**What it is:**
- Heart Rate in beats per minute (BPM)

**What SD measures:**
- Beat-to-beat heart rate variability over the window
- Low SD = very stable heart rate
- High SD = heart rate changing rapidly

**Example:**
- During steady state: Low hr_sd
- During exercise or stress: High hr_sd

**Note:** This is different from HRV (Heart Rate Variability), which typically uses IBI directly.

---

### 5. **Standard Deviation of Inter-Beat Interval**
**Metric Name**: `ibi_sd`

**What it is:**
- IBI (Inter-Beat Interval) = time between consecutive heartbeats (milliseconds)
- Inverse of heart rate

**What SD measures:**
- Classic measure of **Heart Rate Variability (HRV)**
- Low SD = rigid, inflexible heart rhythm (often stress/fatigue)
- High SD = flexible, adaptive heart rhythm (often relaxed/healthy)

**Example:**
- During stress: Low ibi_sd (rigid rhythm)
- During relaxation: High ibi_sd (natural variability)

**Why IBI and HR:**
- IBI is the standard HRV measure in research
- HR provides complementary information
- Together they give complete cardiac variability picture

---

## Comparison: Raw EDA vs. These Metrics

| Metric Type | What It Tells You | Use Case |
|-------------|-------------------|----------|
| **Raw EDA** | Absolute conductance level | "How aroused is the person right now?" |
| **SD of Tonic EDA** | Baseline stability | "Is arousal level stable or drifting?" |
| **SD of SCR Frequency** | Response burst patterns | "Are responses steady or sporadic?" |

### Key Insight
- **Raw EDA** = single snapshot value
- **SD metrics** = information about **change and variability**
- Variability often more predictive of emotional/cognitive states than absolute values

---

## Redis Message Format

### Channel
```
device:{device_id}:physio_metrics
```

### Message Structure (JSON)
```json
{
  "device": "device_id",
  "timestamp": "2026-05-27T14:30:45.123456",
  "edl_sd": 0.123,
  "temperature_roc_sd": 0.045,
  "scr_frequency_sd": 0.234,
  "hr_sd": 2.567,
  "ibi_sd": 15.789
}
```

**Note**: Performance metrics (processing_time_ms, sample counts) are saved to CSV files only, not published to Redis to minimize network overhead.

### Publishing Frequency
- Every 1 second (after processing 5-second rolling window)
- All 5 metrics in single message
- Only published if sufficient data available (>1 or >2 samples depending on metric)

---

## How Standard Deviation is Calculated on Real-Time Signals

### Sliding Window Approach

The script uses a **rolling buffer** method to calculate SD in real-time:

**1. Buffering Strategy**
- Each signal maintains a rolling buffer (e.g., `self.edl_values = []`)
- Buffer size: **5 seconds × 25 Hz = 125 samples**
- As new data arrives, it's **appended** to the buffer
- Old data beyond 125 samples is **automatically removed** (keeps only latest window)

```python
# Implementation example:
max_samples = int(WINDOW_SECONDS * EMIT_RATE_HZ)  # 5 * 25 = 125
if col == "EDL":
    self.edl_values.append(float(value))
    self.edl_values = self.edl_values[-max_samples:]  # Keep only last 125
```

**2. SD Calculation**
- Every **1 second**, the `process_window()` function is called
- It calculates `np.std()` on the **current buffer contents**
- This gives a **rolling standard deviation** over the past 5 seconds

```python
if len(self.edl_values) > 1:
    physio_metrics["edl_sd"] = float(np.std(self.edl_values))
```

**3. Visual Timeline**
```
Time:        0s    1s    2s    3s    4s    5s    6s    7s
Buffer:     [----5 second window----]
                   [----5 second window----]
                          [----5 second window----]

At 5s: Calculate SD of samples from 0-5s
At 6s: Calculate SD of samples from 1-6s (0-1s dropped)
At 7s: Calculate SD of samples from 2-7s (1-2s dropped)
```

### Why This Approach?

**Current Method: 5-Second Window SD**

**What it measures:**
- Variability **within the current moment**
- "How much is this signal fluctuating right now?"

**Advantages:**
- ✅ No calibration needed - works immediately
- ✅ Adapts to changing contexts (activity, posture, environment)
- ✅ Captures **relative** emotional dynamics in real-time
- ✅ Good for detecting **state changes** (calm → stressed)
- ✅ Appropriate for emotion recognition where dynamics matter more than absolute levels

**Limitations:**
- ❌ Ignores individual differences (person A's "high" ≠ person B's "high")
- ❌ Doesn't capture **absolute deviation** from personal norm
- ❌ Someone with naturally high variability looks the same as someone stressed

### Alternative Approach: Baseline-Relative

**Not currently implemented**, but could be added if needed:

**What it would measure:**
- How **different** current values are from person's **normal baseline**
- "How far from baseline is this person right now?"

**Methods:**
1. **Deviation from baseline mean**: `current_value - baseline_mean`
2. **Z-score (normalized)**: `(current_value - baseline_mean) / baseline_sd`

**When baseline-relative would be better:**
- Stress/arousal alerting systems
- Cross-individual comparisons
- Clinical monitoring
- Detecting "unusually high/low for this person"

**Tradeoffs:**
- ✅ Personalized - accounts for individual differences
- ✅ Better for alerting systems
- ❌ Requires 2-5 minute calibration period
- ❌ Baseline may drift during long sessions
- ❌ Less responsive to rapid context changes

### Design Decision

**Current implementation uses 5-second window SD because:**
1. The ML models (valence/arousal prediction) are trained on relative dynamics
2. Emotions are about **changes** and **variability**, not absolute levels
3. No calibration delay means immediate usage upon device connection
4. Better suited for real-time emotion recognition applications

**Future enhancement possibility:** Could add baseline-relative metrics alongside current metrics for hybrid approach if cross-person comparison or alerting features are needed.

---

## Technical Implementation Details

### Signal Tracking
Each `DeviceAggregator` maintains 5 separate buffers:
- `edl_values`: Tonic EDA samples
- `temp_values`: Temperature samples
- `scr_freq_values`: SCR frequency samples
- `hr_values`: Heart rate samples
- `ibi_values`: Inter-beat interval samples

### Window Parameters
- **Window Duration**: 5 seconds
- **Sampling Rate**: 25 Hz
- **Buffer Size**: 125 samples per signal (5s × 25 Hz)
- **Update Frequency**: 1 Hz (every second)

### Calculation Method
```python
# Example: SD of Tonic EDA
if len(self.edl_values) > 1:
    physio_metrics["edl_sd"] = float(np.std(self.edl_values))

# Example: SD of Temperature ROC
if len(self.temp_values) > 2:
    temp_roc = np.diff(self.temp_values) * EMIT_RATE_HZ  # per second
    physio_metrics["temperature_roc_sd"] = float(np.std(temp_roc))
```

---

## Use Cases for These Metrics

### Stress Detection
- High `hr_sd` + Low `ibi_sd` + High `edl_sd` → Acute stress
- Low variability across all → Chronic stress/fatigue

### Emotional Engagement
- High `scr_frequency_sd` + High `temperature_roc_sd` → Active emotional processing
- Low across all → Disengagement

### Cognitive Load
- Decreasing `ibi_sd` + Increasing `hr_sd` → Mental effort
- Stable metrics → Comfortable task difficulty

### Arousal State Classification
- Use all 5 metrics as features for ML models (already done in script)
- Combined with valence/arousal predictions for emotion recognition

---

## Additional Notes

### Signal Sources (EmotiBit Type Tags)
- EDL: `"EL"` (already filtered by device)
- Temperature: `"T0"` or `"T1"` 
- SCR Frequency: `"SF"` (already calculated by device)
- Heart Rate: `"HR"` (already calculated by device)
- IBI: `"BI"` (already calculated by device)

### Data Logging
All raw signal values are still logged to CSV files for offline analysis. The SD metrics are only for real-time Redis publishing.

### Platform Compatibility
Script works on both Windows and macOS with automatic platform detection and adjustment.

---

## Data Storage vs Real-Time Publishing

**Note**: This section applies to both `multiemotibit_UDP_SD.py` and `multiemotibit_UDP_SD_RFv2.py`. Both scripts store and publish data identically; only the RF model files differ.

### Saved to CSV Files (Complete Dataset)
**Location**: `emotibit_recordings/emotibit_{device_id}_{timestamp}.csv`

**All data saved locally:**
- ✅ Raw physiological signals (20+ signals from EmotiBit)
- ✅ Valence and arousal predictions
- ✅ **5 SD metrics** (edl_sd, temperature_roc_sd, scr_frequency_sd, hr_sd, ibi_sd)
- ✅ **Performance metric**:
  - `processing_time_ms` - Time to compute features, predict, and publish (every second)
- ✅ Timestamps

**CSV File Format:**
- **Headers**: timestamp, device, [all EmotiBit signals], valence, arousal, [5 SD metrics], processing_time_ms
- **Update Frequency**: Every 1 second
- **Storage**: Persistent file handle with automatic flush every 1 second

**Recent Changes (May 28, 2026):**
- Fixed missing CSV headers - valence, arousal, SD metrics, and performance metrics now properly included
- Removed end-to-end latency measurement - unreliable without proper clock synchronization between EmotiBit device and computer
- Added terminal output throttling - prints every second for first 3 seconds, then once per 30 seconds to reduce system overhead

**Purpose**: Complete offline analysis and system diagnostics

### Published to Redis (Real-Time Streaming)
**Channels**: 
- `device:{device_id}:physio_metrics` - SD metrics only
- `device:{device_id}:valence_cont` - Valence predictions
- `device:{device_id}:arousal_cont` - Arousal predictions

**What's published:**
- ✅ **5 SD metrics** (edl_sd, temperature_roc_sd, scr_frequency_sd, hr_sd, ibi_sd)
- ✅ Valence and arousal predictions
- ❌ Performance metrics (not published to minimize network overhead)
- ❌ Raw signals (too much data for real-time streaming)

**Purpose**: Real-time emotion recognition and system integration

---

## Performance Monitoring

### Processing Time Measurement

**What it measures:**
- Time to compute features, make predictions, and publish to Redis
- Measured every 1 second for each processing window

**Implementation:**
```python
processing_start = time.time()
# ... feature extraction, prediction, Redis publish ...
processing_time_ms = (time.time() - processing_start) * 1000
```

**Expected Values:**
- **Typical**: 1-10 ms
- **Good**: <5 ms
- **Concerning**: >20 ms (indicates system overload)

**Saved to**: CSV file only (not published to Redis to minimize overhead)

### Terminal Output Throttling

**To reduce system overhead**, terminal printing is throttled:
- **First 3 seconds**: Prints every second (confirm system is working)
- **After 3 seconds**: Prints once every 30 seconds only
- **All data**: Still saved to CSV and published to Redis every second

**Why throttle printing:**
- Terminal I/O can add significant overhead (especially with multiple devices)
- Reduces CPU usage and improves processing performance
- Periodic updates (every 30s) still allow monitoring system health

**What's printed:**
- Timestamp, device ID
- All 5 SD metrics with values
- Processing time
- Valence and arousal predictions

**Example output:**
```
[14:30:45] MD-V5-0001019 edl_sd=0.123, temp_roc_sd=0.045, scr_freq_sd=0.234, hr_sd=2.567, ibi_sd=15.789, proc=3.2ms, Val=1.45, Aro=1.78
```

### Network Latency (Not Measured)

**Why not included:**
End-to-end network latency measurement was removed (May 28, 2026) because:
- EmotiBit uses device-relative timestamps (milliseconds since boot)
- Computer uses absolute Unix timestamps
- No reliable clock synchronization protocol between device and computer
- Clock offset calculations were producing unrealistic values (±5000ms)
- Without NTP/PTP sync, measurements are not meaningful

**If you need network latency monitoring:**
- Use external network monitoring tools
- Monitor `processing_time_ms` for system performance
- Check for dropped packets or connection issues in system logs

---

## Summary

### Script Versions
- **multiemotibit_UDP_SD.py**: Original version with RF models v1 (legacy Python/sklearn)
- **multiemotibit_UDP_SD_RFv2.py**: Updated version with RF models v2 (modern Python/sklearn) - **Recommended for new deployments**

Both scripts have identical functionality with only the RF model files differing.

### Key Features
**Previous Implementation**: Sent filtered HR/EDA values (what's happening right now)  
**Current Implementation**: Sends SD of 5 signals (how much things are changing)

**Why**: Variability metrics provide richer information about physiological and emotional states, are more robust for machine learning, and reduce network traffic while maintaining all raw data in CSV logs.

### Data Flow
- **CSV Storage**: All raw signals + SD metrics + performance metric + predictions (complete dataset)
- **Redis Publishing**: 5 SD metrics + predictions only (real-time streaming)
- **Performance**: Processing time typically 1-10ms per window

### For Deployment
- Use `multiemotibit_UDP_SD_RFv2.py` with updated RF models v2 for Python 3.9+
- Follow `INSTALLATION_GUIDE.md` for complete setup
- Refer to `requirements.txt` for package dependencies
- See `CLIENT_DEPLOYMENT_CHECKLIST.md` for deployment package

---

**Document Version**: May 28, 2026  
**Compatible Scripts**: multiemotibit_UDP_SD.py, multiemotibit_UDP_SD_RFv2.py  
**Last Updated**: Fixed CSV headers; removed unreliable latency measurement; added terminal output throttling for performance optimization
