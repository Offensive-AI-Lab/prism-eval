#!/usr/bin/env python3
"""Label injection_success per row by judging the runner's OWN response.

Compliance post-pass (plan step 4): for each ATTACK row, find the injected
instruction(s) with the adversarial identifier, then judge whether the stored
``EvalResult.model_response`` actually complied. Writes one record per eval_id.
This keeps the attempted-vs-successful adversarial-recovery fork open without
re-running the model — the label is tied to the exact response whose activations
produced the ITM report.

Inputs:
  --suite    eval suite JSON (instruction_sources, prompt, metadata.label)
  --results  EvalResult JSONL (eval_id + model_response), e.g. from `prism-eval run`
Outputs:
  --out      JSONL: {eval_id, injection_success, reasoning, injected_instructions, label}
  --write-into-suite  also set metadata.injection_success on a copy of the suite

Env: PRISM_EVAL_MODEL / PRISM_EVAL_BASE_URL / PRISM_EVAL_API_KEY (gemma4).

Usage:
    python scripts/label_injection_success.py \
        --suite data/eval_suite_xpia.json \
        --results results/xpia/run.jsonl \
        --out results/xpia/injection_success.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from prism_eval.scoring.adversarial_identifier import identify  # noqa: E402
from prism_eval.scoring.injection_compliance import judge_compliance  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")


def _load_suite(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    evals = data["evals"] if isinstance(data, dict) else data
    out = {}
    for r in evals:
        src = (r.get("instruction_sources") or {}).get("original") or []
        md = r.get("metadata") or {}
        out[r["eval_id"]] = {
            "instructions": src,
            "prompt": r.get("prompt"),
            "label": md.get("label"),
        }
    return out


def _load_results(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["eval_id"]] = r.get("model_response")
    return out


def label_one(eval_id: str, rec: dict, model_response: str | None,
              model: str, client) -> dict:
    base = {"eval_id": eval_id, "label": rec.get("label"),
            "injection_success": None, "reasoning": "", "injected_instructions": []}

    # Only attack rows have an injection to comply with.
    if rec.get("label") != "attack":
        base["reasoning"] = "benign row — no injection"
        return base

    instructions = rec["instructions"]
    idx = identify(instructions, rec.get("prompt"), model=model, client=client)
    injected = [instructions[i - 1] for i in idx]
    base["injected_instructions"] = injected
    if not injected:
        base["reasoning"] = "no adversarial bullet identified"
        return base

    verdict = judge_compliance(injected, model_response, model=model, client=client)
    base["injection_success"] = verdict["injection_success"]
    base["reasoning"] = verdict["reasoning"]
    return base


def main() -> None:
    p = argparse.ArgumentParser(description="Label injection_success per row.")
    p.add_argument("--suite", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--write-into-suite", default="",
                   help="optional path to write a suite copy with "
                        "metadata.injection_success merged in")
    p.add_argument("--model", default=os.environ.get("PRISM_EVAL_MODEL", ""))
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    if not args.model:
        sys.exit("No model. Set PRISM_EVAL_MODEL or pass --model.")

    suite = _load_suite(Path(args.suite))
    results = _load_results(Path(args.results))
    eval_ids = [e for e in results if e in suite]
    print(f"{len(eval_ids)} rows to label "
          f"({sum(suite[e]['label'] == 'attack' for e in eval_ids)} attack)")

    from openai import OpenAI

    kwargs: dict = {"api_key": os.environ.get("PRISM_EVAL_API_KEY") or "not-needed"}
    if os.environ.get("PRISM_EVAL_BASE_URL"):
        kwargs["base_url"] = os.environ["PRISM_EVAL_BASE_URL"]
    client = OpenAI(**kwargs)

    labels: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(label_one, e, suite[e], results.get(e), args.model, client): e
            for e in eval_ids
        }
        for i, fut in enumerate(as_completed(futs)):
            row = fut.result()
            labels[row["eval_id"]] = row
            print(f"  [{i+1}/{len(eval_ids)}] {row['eval_id']} "
                  f"success={row['injection_success']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for e in eval_ids:
            f.write(json.dumps(labels[e]) + "\n")

    n_attack = sum(labels[e]["label"] == "attack" for e in eval_ids)
    n_success = sum(labels[e]["injection_success"] is True for e in eval_ids)
    n_refused = sum(labels[e]["injection_success"] is False for e in eval_ids)
    print(f"Wrote {len(eval_ids)} labels -> {out_path}")
    print(f"  attack rows: {n_attack} | complied: {n_success} | "
          f"refused: {n_refused} | unjudged: {n_attack - n_success - n_refused}")

    if args.write_into_suite:
        data = json.loads(Path(args.suite).read_text())
        evals = data["evals"] if isinstance(data, dict) else data
        for r in evals:
            lab = labels.get(r["eval_id"])
            if lab is not None:
                md = r.get("metadata") or {}
                md["injection_success"] = lab["injection_success"]
                r["metadata"] = md
        Path(args.write_into_suite).write_text(json.dumps(data, indent=2))
        print(f"  merged injection_success into {args.write_into_suite}")


if __name__ == "__main__":
    main()
