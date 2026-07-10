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

**Data assets:**
- `D:\Wspc\Python\Amplify\TUS Evaluations\Data Anal\Timestamped Audio\2nd FAMILY LABORATORY - 21 MARCH\2nd Family Laboratory.mp4`
  - Broadly annotatable as **Engaged**.
  - Filmed **from behind** the audience (backs of heads — limits face/gaze features; good for movement/posture).
  - Concert starts **~15:48** into the video.
  - File is **corrupted at the half-way mark** — usable footage ≈ 15:48 → mid-point.
- [ ] Obtain a companion video filmed **from in front** of the audience (faces/gaze visible).

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
