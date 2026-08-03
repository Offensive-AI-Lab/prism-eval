"""Prefill the base-response cache from an already-finished run's Weave traces.

Reconstructs the exact cache keys the runner would compute (same messages
construction as _run_batch_single_turn) and writes {key, response} JSONL
entries for every itm_annotate root trace in the given time window.

Usage:
  python scripts/prefill_response_cache.py \
      --checkpoint <ckpt.pt> --max-new-tokens 2048 \
      --since "2026-07-10T04:01:00Z" \
      --cache results/window_ablation/base_responses.jsonl

Safe by construction: a wrong key is just a cache miss and the next run
regenerates as usual.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prism_eval.runners.prism import (  # noqa: E402
    _base_response_key,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--since", required=True, help="ISO timestamp (UTC) of run start")
    ap.add_argument("--until", default=None, help="optional ISO timestamp (UTC) of run end")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--project", required=True,
                    help="Weave project as <entity>/<project>")
    args = ap.parse_args()

    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_id = cfg["model_id"]
    system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")
    del ckpt

    import weave
    from weave.trace_server.trace_server_interface import CallsFilter

    client = weave.init(args.project)
    lo = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
    hi = datetime.fromisoformat(args.until.replace("Z", "+00:00")) if args.until else None

    calls = client.get_calls(
        filter=CallsFilter(trace_roots_only=True),
        columns=["inputs", "output", "started_at", "op_name"],
        sort_by=[{"field": "started_at", "direction": "desc"}],
    )

    entries: dict[str, str] = {}
    for c in calls:
        t = c.started_at
        if t is None:
            continue
        if t < lo:
            break  # sorted desc — everything after is older
        if hi and t > hi:
            continue
        if "itm_annotate" not in (c.op_name or ""):
            continue
        out = c.output or {}
        if not isinstance(out, dict) or "model_response" not in out:
            continue
        prompt = (c.inputs or {}).get("prompt") or ""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        key = _base_response_key(model_id, messages, args.max_new_tokens)
        entries[key] = out["model_response"]

    cache_path = Path(args.cache)
    existing: set[str] = set()
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    existing.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    continue

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    new = {k: v for k, v in entries.items() if k not in existing}
    with cache_path.open("a", encoding="utf-8") as fh:
        for key, response in new.items():
            fh.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")

    print(f"traces harvested: {len(entries)}, new cache entries written: {len(new)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
