# Known Issues & Fixes — Concert Engagement AR

## 1. SETUP.bat fails: `. was unexpected at this time.`

**Status:** Fixed  
**Cause:** The original Python-detection loop used nested `for`/`if`/`if` blocks with parentheses. `cmd.exe` expands `%PYEXE%` at parse time (not runtime), so the variable was always empty inside the loop body. This caused a syntax error.  
**Fix:** Replaced the `for %%V in (3.12 3.11 3.10)` loop with sequential `py -3.12` / `py -3.11` / `py -3.10` checks using `goto :found_python`. This is reliable across all `cmd.exe` versions and path configurations.

## 2. YOLO26n not detecting people — engagement always 0.0000

**Status:** Fixed  
**Cause:** `yolo26n.pt` (YOLO v26 nano) failed to detect any people in the camera feed — engagement scores stayed at `0.0000` throughout. Likely a model compatibility or weight issue.  
**Fix:** Swapped `yolo11n.pt` as the primary detector and `yolo26n.pt` as the fallback in `live_multiperson_binary_v2.py` (lines 228–234). YOLO11n detects people reliably; engagement scores immediately jumped to ~0.92.

## 3. Valence/Arousal values exceeding 0–2 range (e.g. 7.40, 8.39)

**Status:** Fixed  
**Cause:** `predict_proba()` on the Random Forest models (`rf_valence_full.pkl`, `rf_arousal_full.pkl`) returned raw leaf-node frequency counts that did **not** sum to 1.0 (e.g. `[1.84, 1.795, 2.515]`, sum = 6.15). The dot product `np.dot(proba, [0, 1, 2])` then produced values well above 2.  
**Fix:** Added normalization before the dot product in `multiemotibit_redis_MAC.py`:
```python
val_proba = val_proba / val_proba.sum()
aro_proba = aro_proba / aro_proba.sum()
```
Values now correctly stay in the 0–2 range.

## 4. Batch files don't open separate windows in VS Code terminal

**Status:** Known / by design  
**Cause:** When double-clicked from Explorer, each `.bat` opens its own `cmd.exe` window. Inside VS Code's integrated terminal, they all run in the current terminal and block each other.  
**Workaround:** In VS Code, open separate terminal tabs manually and run each script in its own tab. Or just double-click the `.bat` files from Explorer.

## 5. EmotiBit firewall rules require Administrator

**Status:** Known / expected  
**Cause:** The EmotiBit publisher needs UDP 3131/3132 and TCP 3133 open. The script tries to add rules automatically but fails without admin privileges.  
**Workaround:** Run these once in an Administrator PowerShell:
```
netsh advfirewall firewall add rule name="AMPLIFY EmotiBit UDP 3131" dir=in action=allow protocol=UDP localport=3131
netsh advfirewall firewall add rule name="AMPLIFY EmotiBit UDP 3132" dir=in action=allow protocol=UDP localport=3132
netsh advfirewall firewall add rule name="AMPLIFY EmotiBit TCP 3133" dir=in action=allow protocol=TCP localport=3133
```
