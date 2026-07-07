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
