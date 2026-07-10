# Server mirror (`~/concert_engagement` on `sri-gpu-12t4`, 192.168.200.206)

This folder version-controls the **server-only** source that produces the
engagement models. The training server directory `~/concert_engagement` is **not**
a git repository, so these are mirrored here to preserve the work.

> Not mirrored (intentionally): datasets, extracted `.npz` features, model
> weights (`*.pth`), logs, and the Python virtualenv. Those are large / derived
> and stay on the server.

| File | Purpose |
|------|---------|
| `scripts/ddp_precomputed_dataset.py` | `PrecomputedActionDataset` + sampler. Loads MediaPipe feature `.npz`, applies fine→binary/macro label conversion. |
| `scripts/generate_dataset_indices.py` | Pre-computes shuffled train/val indices (avoids the DDP dataset-scan hang). |
| `scripts/train_action_transformer_ddp_v2.py` | Base DDP transformer trainer (the standard model). |
| `scripts/extract_kinetics_features.py` | MediaPipe Holistic → 543-keypoint feature extraction from raw video. |
| `setup_server_venv.sh` | Recreates the server Python 3.12 venv. |
| `labels/hierarchical_labels.json` | Standard fine→binary label map. |
| `labels/hierarchical_labels_young_families.json` | Young-families label map (staring→engagement). |
| `labels/label_mapping.json` | 86 fine action classes → indices. |

The **Young Families** trainer is generated from the base trainer by
`scripts/training/patch_young_families_trainer.py` (in the main repo) and launched
detached with `scripts/training/launch_young_families.sh`.
