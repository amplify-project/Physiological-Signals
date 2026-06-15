"""Quick one-shot diagnostic for Sowmya's 5 June canteen test.
Detects sensor-removal / re-application events from the CSV the publisher writes.
"""
import sys
import pandas as pd
import numpy as np

p = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Eoghan Hynes\Downloads\emotibit_MD-V5-0000334_20260605_145203.csv"
d = pd.read_csv(p)
d["t"] = pd.to_datetime(d["timestamp"])
d["t_rel"] = (d["t"] - d["t"].iloc[0]).dt.total_seconds()
dt = d["t_rel"].diff()
print(f"rows={len(d)}  span={d['t_rel'].iloc[-1]:.1f}s  median_gap={dt.median():.3f}s  max_gap={dt.max():.3f}s")

big = dt[dt > 5]
print(f"\ngaps > 5 s (candidate sensor-removal silences): {len(big)}")
for i, v in big.items():
    t_here = d.loc[i, "t_rel"]
    print(f"  row {i}: gap {v:5.1f}s, ends at t_rel={t_here:6.1f}s ({d.loc[i,'timestamp']})")

print("\nRow-by-row scan: zero/NaN runs on key channels")
for col, lo, hi in [("EDA", 0.01, 100), ("Temperature0", 25, 40), ("HeartRate", 40, 200), ("PPGGreen", 1, 1e9)]:
    s = pd.to_numeric(d[col], errors="coerce")
    bad_mask = s.isna() | (s == 0) | (s < lo) | (s > hi)
    # Find runs of >5 consecutive bad rows
    runs = []
    run_start = None
    for idx, bad in enumerate(bad_mask):
        if bad and run_start is None:
            run_start = idx
        elif not bad and run_start is not None:
            if idx - run_start >= 5:
                runs.append((run_start, idx - 1))
            run_start = None
    if run_start is not None and len(bad_mask) - run_start >= 5:
        runs.append((run_start, len(bad_mask) - 1))
    print(f"  {col}: bad-runs (>=5 rows): {len(runs)}")
    for s_i, e_i in runs[:10]:
        ts_s = d.loc[s_i, "t_rel"]; ts_e = d.loc[e_i, "t_rel"]
        print(f"    rows {s_i}-{e_i} : t_rel {ts_s:.1f}s -> {ts_e:.1f}s ({ts_e-ts_s:.1f}s)")

# Cross-channel simultaneous silence (the publisher's auto-swap criterion)
print("\nSimultaneous bad windows across EDA + PPG + Temperature (the publisher's auto-swap criterion):")
def bad(col, lo, hi):
    s = pd.to_numeric(d[col], errors="coerce")
    return s.isna() | (s == 0) | (s < lo) | (s > hi)
m = bad("EDA", 0.01, 100) & bad("PPGGreen", 1, 1e9) & bad("Temperature0", 25, 40)
print(f"  rows where all three look 'off': {int(m.sum())} / {len(m)}")
# Find contiguous runs >= 5 rows of all-off
runs = []
run_start = None
for idx, bb in enumerate(m):
    if bb and run_start is None:
        run_start = idx
    elif not bb and run_start is not None:
        if idx - run_start >= 5:
            runs.append((run_start, idx - 1))
        run_start = None
if run_start is not None and len(m) - run_start >= 5:
    runs.append((run_start, len(m) - 1))
print(f"  all-three-off runs of >=5 rows: {len(runs)}")
for s_i, e_i in runs[:10]:
    ts_s = d.loc[s_i, "t_rel"]; ts_e = d.loc[e_i, "t_rel"]
    print(f"    rows {s_i}-{e_i} : t_rel {ts_s:.1f}s -> {ts_e:.1f}s ({ts_e-ts_s:.1f}s)")

# Temperature step around any candidate silence
print("\nSkin temperature trajectory (Temperature0, 5 s rolling mean):")
t0 = pd.to_numeric(d["Temperature0"], errors="coerce")
roll = t0.rolling(5, min_periods=1).mean()
# Print every ~30 s
step = max(1, len(d) // 25)
for i in range(0, len(d), step):
    print(f"  t_rel={d.loc[i,'t_rel']:6.1f}s  T0={roll.iloc[i]:.2f}°C  EDA={pd.to_numeric(d['EDA'], errors='coerce').iloc[i]:.3f}  PPGGreen={pd.to_numeric(d['PPGGreen'], errors='coerce').iloc[i]:.0f}")
