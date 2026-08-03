#!/usr/bin/env python3
"""Strip training state from a checkpoint, keeping only what inference needs.

A training checkpoint is about two thirds optimizer state — Adam keeps two
moments per trainable parameter, so the file is roughly 3x the size of the
weights themselves. `PrismRunner` reads exactly three keys:

    config             the architecture (hook layer, model id, window size, ...)
    lora_state         the LoRA adapters
    projection_state   the activation -> embedding projection

Everything else (optimizer_state, scheduler_state, reward_tracker_state,
encoder_state, wandb_run_id) is training bookkeeping that no one downloading
the release can use. Dropping it takes the published set from ~4.2 GB to
~1.4 GB with bit-identical weights.

    python scripts/strip_checkpoint.py checkpoints/*.pt --out-dir dist/
    python scripts/strip_checkpoint.py checkpoints/prism-qwen3.5-9b-grpo.pt --in-place

Verification is not optional here: the script reloads what it wrote and
compares every retained tensor against the source before reporting success.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Read by PrismRunner. `opt_step` / `val_reward` / `best_reward` are scalars
# kept for provenance — they cost nothing and say which step produced this.
KEEP = ("config", "lora_state", "projection_state", "opt_step", "val_reward", "best_reward")


def _tensors(obj, prefix=""):
    """Flatten a nested state dict into {path: tensor}."""
    out = {}
    if torch.is_tensor(obj):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_tensors(v, f"{prefix}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_tensors(v, f"{prefix}[{i}]"))
    return out


def _identical(a, b) -> tuple[bool, str]:
    ta, tb = _tensors(a), _tensors(b)
    if ta.keys() != tb.keys():
        return False, f"tensor set differs ({len(ta)} vs {len(tb)})"
    for k in ta:
        if ta[k].dtype != tb[k].dtype or ta[k].shape != tb[k].shape:
            return False, f"{k}: dtype/shape changed"
        if not torch.equal(ta[k], tb[k]):
            return False, f"{k}: values changed"
    return True, f"{len(ta)} tensors bit-identical"


def strip_one(src: Path, dst: Path) -> bool:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    missing = [k for k in ("config", "lora_state") if k not in ck]
    if missing:
        print(f"  ✗ {src.name}: not a PRISM checkpoint (missing {missing})", file=sys.stderr)
        return False

    slim = {k: ck[k] for k in KEEP if k in ck}
    dropped = sorted(k for k in ck if k not in slim)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".pt.tmp")
    torch.save(slim, tmp)

    # Reload and compare before committing to the destination.
    ok, detail = _identical({k: ck[k] for k in slim}, torch.load(tmp, map_location="cpu", weights_only=False))
    if not ok:
        tmp.unlink(missing_ok=True)
        print(f"  ✗ {src.name}: verification FAILED — {detail}", file=sys.stderr)
        return False
    tmp.replace(dst)

    before, after = src.stat().st_size, dst.stat().st_size
    print(f"  ✓ {src.name}: {before / 1e6:.0f} MB → {after / 1e6:.0f} MB "
          f"({(1 - after / before) * 100:.0f}% smaller, {detail})")
    if dropped:
        print(f"      dropped: {', '.join(dropped)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, help="Write stripped copies here (default alongside, .slim.pt)")
    ap.add_argument("--in-place", action="store_true", help="Overwrite the source files")
    args = ap.parse_args()

    if args.in_place and args.out_dir:
        ap.error("--in-place and --out-dir are mutually exclusive")

    ok = True
    for src in args.checkpoints:
        if not src.exists():
            print(f"  ✗ {src}: not found", file=sys.stderr)
            ok = False
            continue
        src = src.resolve()  # follow symlinks so --in-place never rewrites a link
        if args.in_place:
            dst = src
        elif args.out_dir:
            dst = args.out_dir / src.name
        else:
            dst = src.with_suffix(".slim.pt")
        ok &= strip_one(src, dst)

    if not ok:
        print("\nSome checkpoints failed — nothing was overwritten for those.", file=sys.stderr)
        return 1
    print("\nRe-run scripts/download_weights.py --verify-only after updating the "
          "SHA256 manifest for the new files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
