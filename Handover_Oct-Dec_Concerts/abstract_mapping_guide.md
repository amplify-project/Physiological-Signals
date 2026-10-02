# Abstract Mapping Guide — Engagement GUI Edition

Physiological Signals GUI · abstract particle-mesh panel · September 2026

This document explains what the abstract particle-mesh visualisation is showing and how to
operate it in the Engagement GUI (`live_multiperson_binary_v2.py`). The panel is a faithful
port of the AR developers' Unity `ParticleMeshVisualizer` (RayNeo X3 Pro, XR Score Viewer
v010alpha9) and is driven by the same five per-wearer streams the glasses use, one panel per
EmotiBit wearer in the sidebar.

## Overall Reading

- **Slow global shape** shows tonic electrodermal activity.
- **Mesh colour and directional trails** show temperature rate of change.
- **Local sparks, rings, and short deformations** show skin conductance responses.
- **Global particle size, synchronized pulse, and four corner halos** show heart rate.
- **Spatial dispersion and mesh coherence** show engagement.

Visual style matches the Unity build: a 5×5 grid of yellow-green particles, the cells
between the mesh wires shaded green, blue semicircular heart-rate arches on the four
corners, and SCR explosions in the same yellow-green shade as the particles.

## Behavioural Signal

### Engagement

Engagement controls how dispersed or coherent the mesh is.

- **Low engagement:** particles disperse and drift, and the surface loses spatial coherence.
- **High engagement:** particles settle into a more coherent, stable mesh.

Engagement does not resize the main particles or remove their minimum visibility. This
preserves the independent heart-rate mapping.

*This represents how strongly the audience member is held by the performance.*

## Physiological Standard Deviations

All four channels are per-wearer session-baseline z-scores (deviation from that wearer's own
mean, in their own SD units) — the same values plotted on the ±3 SD spline panel above.

### Tonic Electrodermal Activity Standard Deviation

This controls the slow, overall pressure of the mesh.

- **Higher positive values** compress the mesh inward and lift its centre upward.
- **Lower negative values** expand the mesh outward and let its centre sink or relax downward.

*This represents slow bodily tension or release.*

### Temperature Rate-of-Change Standard Deviation

This controls particle hue, travelling colour waves, and directional trails.

- **Positive values** produce warm red/orange activity travelling from left to right, with
  trails offset above the particles.
- **Negative values** produce cool blue activity travelling from right to left, with trails
  offset below the particles.
- **Larger magnitude** makes the trails stronger and wider.
- **Strong changes** can trigger short bursts of travelling colour waves.

Temperature does not change the slow global form, the global particle-size baseline, or the
particles' minimum opacity.

*This represents thermal change becoming a visible wave or wash through the audience body.*

### Skin Conductance Response Frequency Standard Deviation

This controls sudden local events: rings, sparks, and short shocks in the mesh.

- **Higher magnitude** creates more frequent and stronger events.
- **Positive events** form expanding rings and push the local surface upward and outward.
- **Negative events** contract inward and pull the local surface downward.
- **Strong events** brighten nearby particles and create spark activity in the particle shade.

SCR events do not change the global size of the main particles. Events fire when the SCR
frequency z-score swings past roughly ±1.65 SD (or rises sharply), so they are rare while a
wearer is calm — use the mock keys below to preview them.

*This represents moments of reaction, surprise, stress, or heightened attention.*

### Heart-Rate Standard Deviation

This controls the global size and synchronized pulse of every main particle, together with an
upper semicircular blue halo above each of the four corner particles.

- **Positive values** enlarge every particle, quicken the pulse, and place each active halo
  outside its neutral reference ring.
- **Negative values** reduce every particle, slow the pulse, and place each active halo inside
  its neutral reference ring.
- **Positive pulses** expand outward; **negative pulses** contract inward.
- The active halos never cross their neutral reference rings, so the sign remains unambiguous
  during a pulse.
- All main particles and all four active halos share one pulse clock and remain in unison.
- Each halo follows its rendered corner particle with slight centre smoothing, while the pulse
  itself remains synchronized.
- If heart-rate data becomes stale, particles return to neutral and the active halos fade
  while the reference rings remain.

*This represents the visible tempo of audience physiology.*

## GUI Controls

The main window must have keyboard focus for key presses to register.

### Mouse

| Input | Where | Action |
| --- | --- | --- |
| Drag left border of the sidebar | ridge at the video/sidebar seam | Resize the whole sidebar (left = wider plots) |
| Drag **bottom border** of an abstract panel | grip dots on the panel's bottom edge | Drag **down to enlarge** the abstract plot (its row grows; the ±SD spline keeps its size), drag up to shrink |
| Left-drag inside an abstract panel | anywhere in the panel | Orbit / tilt that wearer's camera (independent per wearer) |
| Mouse wheel over an abstract panel | anywhere in the panel | Zoom that panel (all panels if none is hovered) |
| Right-click an abstract panel | anywhere in the panel | Reset that panel's view to the default oblique angle |
| Click a wearer row | radio button / row | Focus that wearer (click again for all) |
| Click **RECONNECT** | bottom of the sidebar | Re-run EmotiBit discovery without restarting |

### Keyboard — view

| Key | Action |
| --- | --- |
| `↑` / `+` / `=` | Zoom in on the hovered abstract panel (all panels if none hovered) |
| `↓` / `-` | Zoom out |

### Keyboard — mock signals (demo / verification)

Each key pushes one channel of **every** abstract panel to **+2.5 SD**; hold `Shift` for
**−2.5 SD**. The push decays back to neutral over a few seconds — hold or re-tap the key to
sustain it. Mocks drive only the abstract visual: the ±SD splines, Redis payloads and logged
data are never affected.

| Key | Channel | What you should see (`Shift` = the negative reading) |
| --- | --- | --- |
| `T` | Tonic EDA | Mesh compresses inward and centre lifts (Shift: expands and sinks) |
| `W` | Temperature RoC | Warm waves sweep left→right, trails above (Shift: cool blue right→left, trails below) |
| `S` | SCR frequency | Frequent strong ring events pushing up/outward (Shift: implosions pulling down) |
| `H` | Heart rate | Particles enlarge and pulse faster, halos move outside their reference rings (Shift: shrink, slow, inside) |
| `G` | Engagement | Mesh snaps coherent and stable (Shift: particles disperse and drift) |
| `E` | — | Inject one full-strength SCR ring event directly (bypasses thresholds; Shift: negative implosion). Targets the hovered panel, or all |
| `X` | — | Clear all active mocks immediately |

### Keyboard — session

| Key | Action |
| --- | --- |
| `R` (lowercase) | Register a face for a detected EmotiBit |
| `F` | Cycle focus: all → none → each wearer |
| `V` | Swap primary/secondary camera view (dual-camera mode) |
| `C` | Clear face enrollments |
| `Space` / `Esc` | Advance / cancel an enrollment session |
| `Q` | Quit |
