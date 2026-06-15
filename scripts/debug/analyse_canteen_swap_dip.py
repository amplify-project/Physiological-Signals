"""Pinpoint the sensor-off window in Sowmya's CSV."""
import pandas as pd, numpy as np
p = r"C:\Users\Eoghan Hynes\Downloads\emotibit_MD-V5-0000334_20260605_145203.csv"
d = pd.read_csv(p)
d["t"] = pd.to_datetime(d["timestamp"])
d["t_rel"] = (d["t"] - d["t"].iloc[0]).dt.total_seconds()
eda = pd.to_numeric(d["EDA"], errors="coerce")
edl = pd.to_numeric(d["EDL"], errors="coerce")
ppg = pd.to_numeric(d["PPGGreen"], errors="coerce")
hr = pd.to_numeric(d["HeartRate"], errors="coerce")
ibi = pd.to_numeric(d["InterBeatInterval"], errors="coerce")

low_eda = eda < 0.1
low_ppg = ppg < 1000
combined = low_eda & low_ppg

runs = []
start = None
for i, b in enumerate(combined):
    if b and start is None:
        start = i
    elif not b and start is not None:
        if i - start >= 3:
            runs.append((start, i - 1))
        start = None
if start is not None and len(combined) - start >= 3:
    runs.append((start, len(combined) - 1))

print("EDA<0.1 AND PPGGreen<1000 runs (>=3 rows):")
for s, e in runs:
    s_t = d.loc[s, "t_rel"]
    e_t = d.loc[e, "t_rel"]
    print(f"  rows {s}-{e}  t_rel {s_t:.1f}s -> {e_t:.1f}s  ({e_t - s_t:.1f}s)")

print(f"\nGlobal EDA min/max: {eda.min():.4f} / {eda.max():.4f}")
print(f"Global PPG min/max: {ppg.min():.0f} / {ppg.max():.0f}")

# Sample around dip
if runs:
    s0 = runs[0][0]
    print("\nSample-by-sample across the sensor-off window:")
    for i in range(max(0, s0 - 5), min(len(d), runs[0][1] + 60), 3):
        print(
            f"  t_rel={d.loc[i,'t_rel']:6.1f}s  EDA={eda.iloc[i]:6.3f}  EDL={edl.iloc[i]:6.3f}  PPGGreen={ppg.iloc[i]:6.0f}  HR={hr.iloc[i]:5.1f}  IBI={ibi.iloc[i]:6.1f}"
        )
