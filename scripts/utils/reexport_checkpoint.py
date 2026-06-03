"""Re-export a training checkpoint as a slim state_dict.

Loads the full training-time checkpoint (which contains optimiser state,
schedulers, argparse Namespaces, numpy scalars, etc. and therefore needs
`weights_only=False` to deserialise), strips it down to just the model's
`state_dict`, normalises any DDP `module.` prefix, and saves the result.

The output file loads cleanly under `torch.load(..., weights_only=True)`
on PyTorch 2.6+ with no security trade-off.

Usage:
    python scripts/utils/reexport_checkpoint.py \
        models/action_transformer_12gpus_binary_v2_cleaned/best_model.pth \
        --output models/action_transformer_12gpus_binary_v2_cleaned/best_model.pth
"""

import argparse
import sys
from pathlib import Path

import torch


def reexport(src: Path, dst: Path) -> None:
    print(f"Loading {src} (weights_only=False)...")
    ckpt = torch.load(src, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd = ckpt
    else:
        print(f"ERROR: unexpected checkpoint structure: {type(ckpt)}", file=sys.stderr)
        sys.exit(1)

    cleaned = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    print(f"Extracted {len(cleaned)} tensors.")

    tmp = dst.with_suffix(dst.suffix + '.tmp')
    torch.save(cleaned, tmp)
    print(f"Wrote {tmp}, verifying weights_only=True load...")
    verify = torch.load(tmp, map_location='cpu', weights_only=True)
    assert set(verify.keys()) == set(cleaned.keys()), "round-trip mismatch"
    tmp.replace(dst)
    print(f"OK -> {dst}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('src', type=Path, help='input .pth')
    p.add_argument('--output', '-o', type=Path, required=True, help='output .pth')
    args = p.parse_args()
    reexport(args.src, args.output)


if __name__ == '__main__':
    main()
