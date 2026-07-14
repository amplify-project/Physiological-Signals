# Roadmap

Outstanding development work for the Concert Engagement + Physiological Signals
system. Completed items are removed as they land — this file only tracks what is
still to do.

## Performance & scaling

### Crowd-load FPS floor — parallel MediaPipe
The adaptive round-robin MediaPipe budget (`TARGET_FPS_FLOOR = 12`) is in place.
If it cannot hold 12 FPS on the largest expected audiences, the next lever is
true parallel extraction:

- [ ] Thread pool of per-worker MediaPipe Holistic instances (`threading.local` + `ThreadPoolExecutor`) — currently blocked by MediaPipe Holistic not being thread-safe.
- [ ] Remove remaining blocking synchronisation points; pre-allocate tensors/buffers per frame.
- [ ] Timeline tracing of queue-wait vs compute time; deadline-based scheduling.

### GPU-native keypoint extractor (move off MediaPipe)
MediaPipe Holistic's Python binding is CPU-only TFLite, so the GPU sits near idle
while one CPU core saturates (~30–50 ms/person hard ceiling).

- [ ] Benchmark post-throttle FPS at 10 / 20 / 30 people (2D + 360°) before committing to a swap.
- [ ] DWPose / RTMPose via ONNXRuntime-CUDA or TensorRT — batched 133-keypoint whole-body; needs an adapter to the 543-keypoint schema or a retrain (scaffolding under `src/dwpose_engagement/`).
- [ ] Alternatives if retraining: MMPose (PyTorch); MediaPipe Tasks GPU C++ with a pybind shim (preserves the 543 topology); Sapiens (Meta, 2024) for maximum accuracy.
- [ ] Benchmark script: batched DWPose vs MediaPipe on a recorded clip — FPS at N = {1, 10, 20, 30} plus per-keypoint agreement; decide adapter-vs-retrain.

## Engagement model behaviour & fairness

- [ ] Retrain toward action-based engagement signals.
- [ ] Reduce dependence on distance-from-camera and direct gaze.
- [ ] Don't penalise seated-but-engaged audiences.
- [ ] Better separate arousal from engagement (e.g. separate heads with distinct metrics).
- [ ] Rolling local-max normalisation per audience context without introducing bias.
- [ ] Ensure applause/cheering away from the camera centre still contributes appropriately.
- [ ] Rebalance training across distance bands, viewing angles, seated/standing, and off-centre subjects.
- [ ] Per-session and post-hoc score calibration for venue-specific behaviour.
- [ ] (SOTA) Multi-task / domain-adaptive training with fairness slicing across audience styles.

## Retraining the engagement model
Work tracked on the `Retrain_Engagement_Model` branch. Scope, datasets, label
schema, and training/evaluation plan to be defined.

**Context — young-families concerts need a different notion of "engagement".**
The current model was trained on webcam-facing data and rewards movement, so it
mis-scores this audience. For young-families concerts:

1. Direct gaze at the camera is over-weighted as positive engagement.
2. We only score registered adults — they will *not* be gazing at the camera; if they do, they are usually *not* engaged in the performance.
3. They may be gazing at mobile performers or at their own children.
4. Singing along is likely pro-engagement.
5. High arousal / large amounts of movement means *low* engagement for this audience (inverted vs the standard concert model).

**Scoring stance for this profile:** *presume high engagement by default and only
subtract when an anti-engagement cue is detected.* Scores are computed for the
**registered adults only** (never the infants), per the existing pipeline.
Because parents are normally **seated**, any **standing adult can be treated as a
performer** and excluded from audience scoring (a useful gate, not a label).

**Target action taxonomy (young-families).** Coverage legend against the model's
current Kinetics-700 subset:
`✅ in current subset` · `⚠️ exists in Kinetics-700 but not downloaded / weak in pose-only` · `❌ not a discrete action — must be a pose/gaze/arousal feature, not a Kinetics class`

*Pro-engagement:*
- [ ] Clapping hands — ✅ `applause` (applauding/clapping), already trained.
- [ ] Clicking/snapping fingers — ⚠️ `snapping fingers` exists in K700, not in subset; subtle, weak in pose-only.
- [ ] Tapping hands on lap — ❌ no clean class; nearest `drumming fingers` is currently *dis*engaged. Better as a rhythmic-motion feature.
- [ ] Singing along — ✅ `singing`, but the pose-only proxy is weak; mouth/jaw landmarks or an audio channel would help.
- [ ] Smiling — ⚠️ `smiling` is a K700 class, not in subset; needs **face**, not body pose.
- [ ] Mimicking performers' motions (dancing) — ✅ `dancing` family, already trained.
- [ ] Still head / long gaze fixation on stage — ❌ gaze-stability feature (head-pose variance over time), not an action class.
- [ ] Seated — ❌ posture feature (hip/knee geometry); also the standing→performer gate above.

*Anti-engagement:*
- [ ] Moving around a lot / agitation — ❌ arousal/motion-energy feature (inverted sign vs standard model).
- [ ] Touching face — ⚠️ no exact K700 class (closest `scratching head`); mostly a hand-to-face-proximity feature.
- [ ] Talking — ⚠️ no plain "talking" class (`arguing` exists, disengaged); needs face/audio for reliability.
- [ ] Drinking — ✅ `eating_drinking` (sipping cup / drinking shots), already trained.
- [ ] Eating — ✅ `eating_drinking`, already trained.
- [ ] Distracted by child (looks like holding their arm) — ❌/⚠️ contextual; `carrying baby` exists in K700 (not in subset). Likely a body-orientation + arm-pose feature.
- [ ] Turned away from performance (often toward their child) — ❌ torso/head orientation feature.
- [ ] Sudden synchronous gaze shift of multiple parents (child noise/fall) — ❌ multi-person temporal feature (shared, sudden fixation change); not learnable from single-clip Kinetics actions.
- [ ] Holding objects — phone: ✅ `phone_distraction`; bottle/other: ❌ object-in-hand feature (would need an object detector, not the action head).

**Key takeaway:** roughly half of the listed cues (clapping, dancing, singing,
eating, drinking, phone) map to actions the current model already covers; the
other half (gaze stillness/shift, orientation, agitation, seated/standing,
touching face, holding a bottle, child-tending) are **not discrete Kinetics
actions** — they are pose/gaze/orientation/arousal/object features. This argues
for a **hybrid** design: keep the pose-sequence action head for the "✅" cues and
add a lightweight **feature/rules layer** (orientation, gaze variance, motion
energy, hand-to-face, object-in-hand) for the "❌" cues, rather than trying to
force everything through a Kinetics-style action classifier.

**Approach:** keep the existing model as a "Standard/Concert" profile and add a
"Young Families" profile, selectable in the GUI. A profile = (model weights +
label semantics + optional post-processing). Because every engagement model
shares the same input contract (543-keypoint MediaPipe sequence → binary head),
swapping is just loading a different `state_dict`; both startup selection and
runtime hot-swap are viable.

**Open questions / decisions (to resolve before building):**
- [ ] **Data** — do we have labelled young-families footage, or annotate from saved sessions (`data/sessions/*` JSONL + keypoints NPZ)?
- [ ] **Definition** — is "engaged" here ≈ *calm attention oriented to performers + singing along*, penalised by agitation/distraction? Pin the label rubric.
- [ ] **Singing** — stay pose-only (mouth/jaw landmarks) or add an audio channel?
- [ ] **Model shape** — keep binary, or move to two heads (engagement + arousal) so a profile can express "engaged = attentive AND low-arousal"?
- [ ] **Switching** — startup-only selection, or hot-swappable mid-session?

**Labelling heuristic (from reviewing the footage):** parents' gaze is mostly on
their *children*, not the stage or camera. Default assumption = **Engaged**,
flipped to **Disengaged** only on detectable negatives: technology/phone
distraction, high parent movement/agitation, or a child crying. Check whether the
YouTube action dataset behind the current model already contains
parent-with-baby / childcare actions we can reuse for these classes.

**Interim (no retrain):** ship the profile/model-selector mechanism + a
rule-based "Young Families" post-processor over the existing model (down-weight
motion/arousal, strip camera-gaze bias) so there's something usable while data
for a full retrain is assembled.

**Validation caveat — `staring` semantics (verify on footage).** The v0
young-families model flips the single class `staring` from *disengagement* →
*engagement* (its only dataset change). This assumes `staring` in Kinetics-700
means **gaze held steady / low head movement** (an attentive parent watching the
stage). If, instead, the Kinetics `staring` clips actually depict staring
*toward the camera*, the flip is wrong for our setup — the audience is **not**
looking at the camera, so a camera-facing "staring" cue would be a false
engagement signal. Confirm which meaning holds when reviewing the model's
predictions on the family-concert videos.

**Gaze concern (camera-relative vs stage-relative).** More generally, any
gaze/head-orientation cue is only meaningful **relative to the stage**, not the
camera. Because the camera can sit **behind, to the side of, or in front of** the
audience (see below), "looking forward" in the frame does not reliably mean
"looking at the performers." Gaze/orientation features must therefore be
interpreted relative to the (unknown, per-setup) stage direction, or they will
misfire when the camera angle changes. Flag any gaze-based signal as
camera-angle-dependent until we can calibrate the stage direction per recording.

**Camera angle — must cater for multiple viewpoints.** The audience will **not**
necessarily be filmed from directly behind; a **side** angle is likely, and a
**front** angle is also possible. The pipeline must tolerate all three
(rear / side / front) rather than assuming backs-of-heads. Practically: pose,
posture and motion features are fairly view-robust, but face/gaze features and
any "facing forward = engaged" logic are **not** and need the stage-relative
calibration noted above.

**Evaluation plan — two viewpoints.** We will evaluate the v0 model on **two
videos: one filmed from the rear and one from the front** of the audience, to
measure how sensitive accuracy is to camera angle (and to sanity-check the
`staring`/gaze behaviour under each viewpoint).

## Gaze-first retrain (observations from "4th lab video.mp4")

Validation of the current model on `Family Lab Videos/4th lab video.mp4`
suggests the tiny/basic model should be retrained to be **primarily about gaze
rather than actions**. Key scene structure observed:

- **Audience adults are typically seated**; musicians and dancers are typically
  standing / walking / mobile.
- **The mean of the audience gaze converges on a single point** — the active
  performer. This shared focal point is the core engagement signal.

### Core gaze mechanics
- [ ] Model gaze as a **straight vector out from the front of the face** (no eye tracking needed initially).
- [ ] Compute per-person gaze vectors each frame and estimate the **common focal point** (e.g. least-squares intersection / density peak of vector crossings).
- [ ] **Departures from the common focal point = distraction = disengagement** for that individual.
- [ ] **Staring at own feet / down at the floor = disengaged.**
- [ ] **Sudden synchronized deviation** of multiple gazes to a new rapid common direction = an off-stage event (e.g. a child fell) — a distraction, so **disengaged from the concert** (even though gazes still agree).
- [ ] **No common gaze focal point at all → the concert likely hasn't started yet** (pre-show state; suppress scoring or mark session as not-started).
- [ ] **Audience movement is NOT penalised** as disengagement if the person's gaze stays on the crowd-average focal point — people shift position to get a better view.

### Performer identification & attribution
- [ ] **Standing adults are identified as performers** (audience adults are seated).
- [ ] Performer status is **sticky**: once detected standing as a performer, they remain a performer even if they later sit.
- [ ] **Performers currently being gazed at get an orange bounding box** in the overlay.
- [ ] Record each **performer's engagement contribution to file** — especially when multiple performers are standing at once (they usually take turns), attribute audience gaze/engagement to whichever performer holds the focal point.
- [ ] (Stretch) Use YOLO to **identify the instrument being played** (saxophone, guitar, accordion — each musician plays exactly one), to help distinguish/track individual musicians.

### Overlay
- [ ] Draw the per-person **gaze vectors** on the overlay.
- [ ] Mark the estimated common focal point.
- [ ] Orange bounding box on the performer(s) being gazed at (above).

### Fallback if gaze-focal-point doesn't improve accuracy — parent–child linking
- [ ] Use YOLO to identify **infants as roughly 1/3 the size of adults**.
- [ ] The **adult sitting closest to a child is likely its parent**; an **adult touching a child is almost certainly its parent**.
- [ ] With parent–child links established: **adult gaze on their own child = definitively acceptable** (not disengagement).
- [ ] **Adult fixation on another audience adult = disengaged.**

### Staged model plan
1. **v1 — gaze-based model:** retrain the tiny model around gaze-vector /
   focal-point features (above) as the primary engagement signal.
2. **v2 — fortify with actions:** reintroduce the action head (existing
   Kinetics-subset classes) on top of the gaze core.
3. **v3 — rhythmic/participation features:** add sing-along, clapping, and
   tapping detection as pro-engagement features.

**If accuracy is poor — revisit the dataset.** If validation on the family
footage shows the v0 model is not accurate enough, the next step is to **revisit
the Kinetics-700 dataset and consider downloading/adding new action classes** we
previously excluded (e.g. `snapping fingers`, `smiling`, `carrying baby`,
`scratching head`) and/or re-deriving the `staring` label, then re-extract
features and retrain. Treat the current 86-class subset as a starting point, not
a fixed ceiling.

**Data assets:**
- `D:\Wspc\Python\Amplify\TUS Evaluations\Data Anal\Timestamped Audio\2nd FAMILY LABORATORY - 21 MARCH\2nd Family Laboratory.mp4`
  - Broadly annotatable as **Engaged**.
  - Filmed **from behind** the audience (backs of heads — limits face/gaze features; good for movement/posture).
  - Concert starts **~15:48** into the video.
  - File is **corrupted at the half-way mark** — usable footage ≈ 15:48 → mid-point.
- [ ] Obtain a companion video filmed **from in front** of the audience (faces/gaze visible).
- [ ] **Evaluation set = one rear video + one front video** (camera-angle sensitivity check; expect side angles in real deployments too).

**Plan:**
- [ ] Define objective and label schema (engagement vs arousal; binary vs graded).
- [ ] Assemble / curate training and evaluation datasets.
- [ ] Set the training pipeline, metrics, and success criteria.
- [ ] Validate against held-out data and current-model baselines before release.

## Facial identification operating range
Current stack (facenet-pytorch, MTCNN `min_face_size=40`, InceptionResnetV1 on
160×160 crops) gives a detection floor ≈ 40 px, usable ≈ 80 px, reliable ≈ 120 px
→ ≈ 7.3 / 3.7 / 2.4 m on a 1080p / ~60° FOV webcam (halve for the Insta360 5.7K
equirectangular feed). Enrolment quality caps the effective runtime range.

- [ ] Keep face ID optional and non-blocking so engagement stays stable at long range.
- [ ] Use higher-resolution crops; match only on stable frontal frames.
- [ ] Maintain identity between sparse matches via tracker continuity.
- [ ] Multi-angle / multi-shot enrolment (3–5 crops per person, per-person gallery, quality-weighted average embedding).
- [ ] Enrolment quality gate (min face-pixel size, frontal pose, sharpness) with a re-enrol prompt.
- [ ] (SOTA) Fuse face + body re-identification embeddings for longer-range identity persistence.

## 360° validation

- [ ] Validate the Insta360 ONE RS 360° workflow end-to-end.
