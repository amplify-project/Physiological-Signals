#!/usr/bin/env python3
"""
Generate `train_family_concert.py` from the proven DDP trainer.

Family-Concert v1 is the attention-first, young-families-concert model. It reuses
the exact same feature files and pre-computed indices as the 86-class models, but:

  1. New CLI args: --hierarchical-file, --engagement-weight, --num-workers
     (identical to the young-families trainer).
  2. After the datasets load, RESTRICT each dataset to the `train_classes`
     allow-list in the family-concert label JSON (drops the ~74 neutral/irrelevant
     classes), then override fine->binary from that JSON. The allow-list filter
     mirrors the proven `quarantined_classes` filter already in
     PrecomputedActionDataset, so it is low-risk.
  3. Replace CrossEntropyLoss with a class-weighted loss that up-weights the
     engagement class ("presume engaged" tilt).

Run this ON THE SERVER from ~/concert_engagement:
    python3 scripts/patch_family_concert_trainer.py
"""

from pathlib import Path

SRC = Path("scripts/train_action_transformer_ddp_v2.py")
DST = Path("scripts/train_family_concert.py")

ARGS_ANCHOR = """    parser.add_argument('--sequence-length', type=int, default=300)
    args = parser.parse_args()"""

ARGS_PATCH = """    parser.add_argument('--sequence-length', type=int, default=300)
    parser.add_argument('--hierarchical-file', type=str,
                        default='models/action_transformer_kinetics700/hierarchical_labels_family_concert.json',
                        help='Family-concert fine->binary label file with a train_classes allow-list')
    parser.add_argument('--engagement-weight', type=float, default=1.5,
                        help='CrossEntropy class weight for the engagement class (disengagement stays 1.0)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader worker processes per rank (0 = main-thread loading).')
    args = parser.parse_args()"""

LOAD_ANCHOR = "    total_load_time = time.time() - start_time"

LOAD_PATCH = """    total_load_time = time.time() - start_time

    # === Family-Concert allow-list filter + label override ==========
    # Reuses the same feature files/indices as the 86-class models. First keep
    # only the classes in `train_classes` (drops the neutral/irrelevant ones),
    # then remap the survivors fine->binary. Absent classes in the allow-list
    # (e.g. not-yet-extracted karaoke/laughing) simply contribute no samples.
    import json as _fc_json
    import numpy as _fc_np
    with open(args.hierarchical_file) as _fc_f:
        _fc = _fc_json.load(_fc_f)
    _fine_to_binary = _fc['conversions']['fine_to_binary']
    _binary_to_idx = _fc['binary']['action_to_idx']
    args._engagement_idx = int(_binary_to_idx.get('engagement', 0))
    _allow = set(_fc.get('train_classes', []))  # empty => keep all classes

    def _apply_fc(ds):
        # 1) restrict samples to the allow-list (uses the ORIGINAL fine idx_to_action)
        if _allow:
            kept = [(p, l) for (p, l) in ds.samples if ds.idx_to_action[l] in _allow]
            ds.samples = _fc_np.array(kept, dtype=object)
        # 2) fine -> binary conversion for surviving classes
        conv = {}
        for fine_name, fine_idx in ds.action_to_idx.items():
            b = _fine_to_binary.get(fine_name)
            if b is not None:
                conv[fine_idx] = _binary_to_idx[b]
        ds.label_conversion = conv
        ds.num_classes = int(_fc['binary']['num_classes'])
        ds.idx_to_action = {v: k for k, v in _binary_to_idx.items()}

    _apply_fc(train_dataset)
    _apply_fc(val_dataset)
    # Filtering shrinks the datasets, so the precomputed sampler (indexed against
    # the full 86-class sample list) would overrun. Force the resize-safe
    # DistributedSampler branch below, exactly as the cleaned-labels path does.
    args.use_cleaned_labels = True
    if rank == 0:
        log(f"Family-Concert labels applied from {args.hierarchical_file}")
        log(f"   kept classes: {sorted(_allow) if _allow else 'ALL'}")
        log(f"   train={len(train_dataset.samples):,} val={len(val_dataset.samples):,} "
            f"num_classes={train_dataset.num_classes} engagement_idx={args._engagement_idx}")
    # ================================================================"""

CRIT_ANCHOR = "    criterion = nn.CrossEntropyLoss()"

CRIT_PATCH = """    if args.label_level == 'binary' and getattr(args, 'engagement_weight', 1.0) != 1.0:
        _w = torch.ones(num_classes, device=device)
        _w[getattr(args, '_engagement_idx', 0)] = args.engagement_weight
        criterion = nn.CrossEntropyLoss(weight=_w)
        log(f"[Rank {rank}] Class-weighted loss: engagement={args.engagement_weight}, disengagement=1.0", rank)
    else:
        criterion = nn.CrossEntropyLoss()"""

TRAIN_LOADER_ANCHOR = """    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              sampler=train_sampler, num_workers=0, pin_memory=True)"""

TRAIN_LOADER_PATCH = """    _dl_kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        _dl_kwargs.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, **_dl_kwargs)"""

VAL_LOADER_ANCHOR = """    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=0, pin_memory=True)"""

VAL_LOADER_PATCH = """    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            sampler=val_sampler, **_dl_kwargs)"""


def main():
    text = SRC.read_text()
    for anchor, patch in (
        (ARGS_ANCHOR, ARGS_PATCH),
        (LOAD_ANCHOR, LOAD_PATCH),
        (CRIT_ANCHOR, CRIT_PATCH),
        (TRAIN_LOADER_ANCHOR, TRAIN_LOADER_PATCH),
        (VAL_LOADER_ANCHOR, VAL_LOADER_PATCH),
    ):
        count = text.count(anchor)
        if count != 1:
            raise SystemExit(f"Anchor found {count} times (expected 1):\n{anchor}")
        text = text.replace(anchor, patch)

    header = (
        '"""AUTO-GENERATED from train_action_transformer_ddp_v2.py by '
        'patch_family_concert_trainer.py.\nDo not edit directly; edit the patch '
        'script and regenerate.\n"""\n'
    )
    DST.write_text(header + text)
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
