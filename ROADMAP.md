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
