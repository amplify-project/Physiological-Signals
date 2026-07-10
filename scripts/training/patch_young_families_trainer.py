#!/usr/bin/env python3
"""
Generate `train_young_families.py` from the proven DDP trainer.

Rather than fork ~550 lines, we take `train_action_transformer_ddp_v2.py` and
apply three surgical patches:

  1. New CLI args: --hierarchical-file and --engagement-weight.
  2. After the datasets load, override each dataset's `label_conversion` from the
     young-families label JSON (so the same pre-computed feature indices are
     reused, only the fine->binary semantics change, e.g. staring -> engagement).
  3. Replace the plain CrossEntropyLoss with a class-weighted loss that
     up-weights the engagement class ("presume engaged" tilt).

Run this ON THE SERVER from ~/concert_engagement:
    python3 scripts/patch_young_families_trainer.py
"""

from pathlib import Path

SRC = Path("scripts/train_action_transformer_ddp_v2.py")
DST = Path("scripts/train_young_families.py")

ARGS_ANCHOR = """    parser.add_argument('--sequence-length', type=int, default=300)
    args = parser.parse_args()"""

ARGS_PATCH = """    parser.add_argument('--sequence-length', type=int, default=300)
    parser.add_argument('--hierarchical-file', type=str,
                        default='models/action_transformer_kinetics700/hierarchical_labels_young_families.json',
                        help='Young-families fine->binary label file (overrides label_conversion)')
    parser.add_argument('--engagement-weight', type=float, default=1.5,
                        help='CrossEntropy class weight for the engagement class (disengagement stays 1.0)')
    args = parser.parse_args()"""

LOAD_ANCHOR = "    total_load_time = time.time() - start_time"

LOAD_PATCH = """    total_load_time = time.time() - start_time

    # === Young Families label override ===============================
    # Swap the standard binary label_conversion for the young-families one.
    # Reuses the same feature files/indices; only fine->binary semantics change.
    import json as _yf_json
    with open(args.hierarchical_file) as _yf_f:
        _yf = _yf_json.load(_yf_f)
    _fine_to_binary = _yf['conversions']['fine_to_binary']
    _binary_to_idx = _yf['binary']['action_to_idx']
    args._engagement_idx = int(_binary_to_idx.get('engagement', 0))

    def _apply_yf(ds):
        conv = {}
        for fine_name, fine_idx in ds.action_to_idx.items():
            b = _fine_to_binary.get(fine_name)
            if b is not None:
                conv[fine_idx] = _binary_to_idx[b]
        ds.label_conversion = conv
        ds.num_classes = int(_yf['binary']['num_classes'])
        ds.idx_to_action = {v: k for k, v in _binary_to_idx.items()}

    _apply_yf(train_dataset)
    _apply_yf(val_dataset)
    if rank == 0:
        log(f"\\u2705 Young Families labels applied from {args.hierarchical_file}")
        log(f"   num_classes={train_dataset.num_classes}, engagement_idx={args._engagement_idx}")
        _ov = _yf.get('overrides_vs_standard', [])
        log(f"   overrides vs standard: {_ov}")
    # ================================================================"""

CRIT_ANCHOR = "    criterion = nn.CrossEntropyLoss()"

CRIT_PATCH = """    if args.label_level == 'binary' and getattr(args, 'engagement_weight', 1.0) != 1.0:
        _w = torch.ones(num_classes, device=device)
        _w[getattr(args, '_engagement_idx', 0)] = args.engagement_weight
        criterion = nn.CrossEntropyLoss(weight=_w)
        log(f"[Rank {rank}] \\u2696\\ufe0f  Class-weighted loss: engagement={args.engagement_weight}, disengagement=1.0", rank)
    else:
        criterion = nn.CrossEntropyLoss()"""


def main():
    text = SRC.read_text()
    for anchor, patch in (
        (ARGS_ANCHOR, ARGS_PATCH),
        (LOAD_ANCHOR, LOAD_PATCH),
        (CRIT_ANCHOR, CRIT_PATCH),
    ):
        count = text.count(anchor)
        if count != 1:
            raise SystemExit(f"Anchor found {count} times (expected 1):\n{anchor}")
        text = text.replace(anchor, patch)

    header = (
        '"""AUTO-GENERATED from train_action_transformer_ddp_v2.py by '
        'patch_young_families_trainer.py.\nDo not edit directly; edit the patch '
        'script and regenerate.\n"""\n'
    )
    DST.write_text(header + text)
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
