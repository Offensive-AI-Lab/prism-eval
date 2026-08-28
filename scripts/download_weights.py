#!/usr/bin/env python3
"""Download the published checkpoints from the Hugging Face Hub.

Configs refer to checkpoints as ``${PRISM_EVAL_CHECKPOINT_DIR}/<name>.pt``, so
this only needs to run once:

    export PRISM_EVAL_CHECKPOINT_DIR=./checkpoints
    python scripts/download_weights.py

Downloads are verified against the SHA256 manifest below. A mismatch is a hard
error — a truncated or substituted checkpoint would silently produce wrong
numbers rather than failing loudly.

These are inference-only checkpoints: optimizer and scheduler state have been
stripped (see scripts/strip_checkpoint.py), which is why they are about a third
the size of the training artifacts. The weights themselves are bit-identical.

    python scripts/download_weights.py --only prism-qwen3.5-9b-grpo   # just the main model
    python scripts/download_weights.py --verify-only      # re-check local files
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

# Each checkpoint is its own Hugging Face model repo, so it carries a
# `base_model` tag linking it to the target model it reads and shows up in
# search on its own. $PRISM_EVAL_WEIGHTS_ORG overrides the org (useful for a
# fork or a mirror).
ORG = os.environ.get("PRISM_EVAL_WEIGHTS_ORG", "Offensive-AI-Lab")

# filename -> (sha256, approx size, what it is). The repo id is the filename
# without its extension, under $ORG.
MANIFEST: dict[str, tuple[str, str, str]] = {
    # ── Qwen3.5-9B target model ─────────────────────────────────────────
    "prism-qwen3.5-9b-grpo.pt": (
        "5bde25517e11ff26130c2d842dd01ebbf7b7ed5c997ce89b949ff05aa1d7d2d1",
        "266M",
        "Main model: PRISM, GRPO-tuned. Target model: Qwen3.5-9B, hook layer 16.",
    ),
    "prism-qwen3.5-9b-sft.pt": (
        "347cc6c6674a839dd995633a057f9ddb6cb45e153f912671445eed66c6a56002",
        "266M",
        "PRISM, SFT only — GRPO's initialisation. Target model: Qwen3.5-9B, hook layer 16.",
    ),
    # ── Other target models: the generalisation rows ────────────────────
    "prism-gemma-2-9b-it-grpo.pt": (
        "45e475030a31286e4b25a3404f04bc17b7624156fa7852a6362fbdf59ce09a4a",
        "458M",
        "PRISM, GRPO-tuned. Target model: Gemma-2-9B, hook layer 21.",
    ),
    "prism-ministral-3-8b-grpo.pt": (
        "1ad068c4928e503b9526510d1790b839d16dc0de534efa0ff148b9ee221ef128",
        "390M",
        "PRISM, GRPO-tuned. Target model: Ministral-3-8B, hook layer 17.",
    ),
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, expected: str) -> bool:
    if expected == "PENDING_UPLOAD":
        print(f"  ! {path.name}: no checksum recorded yet — skipping verification")
        return True
    actual = sha256_of(path)
    if actual != expected:
        print(f"  ✗ {path.name}: SHA256 MISMATCH")
        print(f"      expected {expected}")
        print(f"      actual   {actual}")
        return False
    print(f"  ✓ {path.name}: checksum OK")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--org", default=ORG,
                    help=f"Hugging Face org holding the checkpoint repos (default: {ORG})")
    ap.add_argument("--dest", type=Path,
                    default=Path(os.environ.get("PRISM_EVAL_CHECKPOINT_DIR", "./checkpoints")),
                    help="Destination dir (default: $PRISM_EVAL_CHECKPOINT_DIR or ./checkpoints)")
    ap.add_argument("--only", action="append", metavar="NAME",
                    help="Download only this checkpoint (repeatable). "
                         f"Choices: {', '.join(n[:-3] for n in MANIFEST)}")
    ap.add_argument("--verify-only", action="store_true",
                    help="Don't download; just checksum what's already in --dest.")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token, if the repo is gated (default: $HF_TOKEN)")
    args = ap.parse_args()

    wanted = dict(MANIFEST)
    if args.only:
        keys = {f"{n}.pt" if not n.endswith(".pt") else n for n in args.only}
        unknown = keys - set(MANIFEST)
        if unknown:
            ap.error(f"unknown checkpoint(s): {', '.join(sorted(unknown))}")
        wanted = {k: v for k, v in MANIFEST.items() if k in keys}

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {args.dest.resolve()}")

    if not args.verify_only:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("ERROR: pip install huggingface_hub", file=sys.stderr)
            return 2
        for name, (_, size, desc) in wanted.items():
            print(f"→ {args.org}/{name[:-3]} ({size}) — {desc}")
            hf_hub_download(
                repo_id=f"{args.org}/{name[:-3]}",
                filename=name,
                local_dir=args.dest,
                token=args.token,
            )

    print("\nVerifying checksums...")
    ok = True
    for name, (expected, _, _) in wanted.items():
        path = args.dest / name
        if not path.exists():
            print(f"  ✗ {name}: missing")
            ok = False
            continue
        ok &= verify(path, expected)

    if not ok:
        print("\nVerification FAILED — do not use these files.", file=sys.stderr)
        return 1

    print(
        f"\nDone. Point your config at them with:\n"
        f"  export PRISM_EVAL_CHECKPOINT_DIR={args.dest.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
