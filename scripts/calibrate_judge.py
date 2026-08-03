#!/usr/bin/env python3
"""Measure how well a judge agrees with the human gold labels on instruction coverage.

`data/calibration/coverage_gold_v1.jsonl` holds 93 ITM reports scored claim by
claim by two annotators who then sat together and reconciled every disagreement
into a single gold label. Each record also carries the prompt, the target
model's response, and the scores the published judge gave. That is enough to
calibrate *any* judge:

    # 1. Reproduce the shipped numbers — no endpoint, no GPU, instant.
    python scripts/calibrate_judge.py

    # 2. Score your own judge against the same labels.
    export PRISM_EVAL_MODEL=... PRISM_EVAL_BASE_URL=... PRISM_EVAL_API_KEY=...
    python scripts/calibrate_judge.py --rescore --round pilot

**judge vs gold** is the headline number. The **human-vs-human** row is the
ceiling for context: two trained annotators working from the same rubric agree
only to that level *before* reconciling, so a judge approaching it is doing as
well as independent human labelling does.

Nothing here needs Weave, a W&B account, or a network connection — the labels
ship as files.

Kappa is reported with quadratic weights on the ordinal scale
(0.0 missed, 0.5 partial, 1.0 covered), which is what the paper reports, and
unweighted alongside it. On these labels the two differ a lot (0.82 vs 0.63),
so a number without its convention attached is meaningless.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Importable without installing the package (scripts/ lands on sys.path, not the root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
SNAPSHOT = REPO / "data" / "calibration" / "coverage_gold_v1.jsonl"
FROZEN = REPO / "data" / "calibration" / "coverage_calibration_v1.json"

ROUNDS = ("pilot", "hard", "hard2")
ANNOTATORS = ("annotator_a", "annotator_b")


def agreement(a: list[float], b: list[float]) -> dict:
    """Quadratic-weighted and unweighted Cohen's kappa, plus Gwet's AC2."""
    n = len(a)
    labels = sorted(set(a) | set(b))
    q = len(labels)
    dis = lambda x, y: (abs(x - y)) ** 2  # noqa: E731

    p_a = {l: a.count(l) / n for l in labels}
    p_b = {l: b.count(l) / n for l in labels}

    d_obs = sum(dis(x, y) for x, y in zip(a, b)) / n
    d_exp = sum(dis(i, j) * p_a[i] * p_b[j] for i in labels for j in labels)
    kappa_w = 1 - d_obs / d_exp if d_exp else 1.0

    p_obs = sum(x == y for x, y in zip(a, b)) / n
    p_exp = sum(p_a[l] * p_b[l] for l in labels)
    kappa_u = (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else 1.0

    weights = [[1 - dis(i, j) for j in labels] for i in labels]
    p_agree = sum(weights[labels.index(x)][labels.index(y)] for x, y in zip(a, b)) / n
    pi = {l: (p_a[l] + p_b[l]) / 2 for l in labels}
    total_w = sum(sum(row) for row in weights)
    ac2_exp = (total_w / (q * (q - 1))) * sum(pi[l] * (1 - pi[l]) for l in labels) if q > 1 else 0.0
    ac2 = (p_agree - ac2_exp) / (1 - ac2_exp) if ac2_exp < 1 else 1.0

    return {
        "n_claims": n,
        "cohens_kappa_quadratic": round(kappa_w, 4),
        "cohens_kappa_unweighted": round(kappa_u, 4),
        "gwets_ac2": round(ac2, 4),
        "agreement_pct": round(p_obs * 100, 1),
    }


def _paired(rows: list[dict], left, right) -> dict | None:
    """Pool claim scores from two label sources. Vectors zip to the shorter."""
    a: list[float] = []
    b: list[float] = []
    n_evals = 0
    for row in rows:
        la, lb = left(row), right(row)
        if not la or not lb:
            continue
        for x, y in zip(la, lb):
            a.append(x)
            b.append(y)
        n_evals += 1
    if not a:
        return None
    return {"n_evals": n_evals, **agreement(a, b)}


def human(name: str):
    return lambda row: ((row.get("human") or {}).get(name) or {}).get("instruction_scores")


def judge(key: str = "judge"):
    return lambda row: (row.get(key) or {}).get("instruction_scores")


def gold(row):
    return row["gold"]["instruction_scores"]


def rescore(rows: list[dict], model: str | None, workers: int) -> None:
    """Re-run a judge over every row, writing scores into row['rescored']."""
    from prism_eval.scoring import judge_llm

    def one(row: dict) -> None:
        out = judge_llm.score(
            instructions=row["gt_instructions"],
            report=row["itm_report"],
            prompt=row["prompt"],
            model_response=row["model_response"],
            model=model,
        )
        row["rescored"] = {
            "instruction_scores": [c.score for c in out.instruction_scores],
            "hallucination_scores": list(out.hallucination_scores),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(one, rows), 1):
            if i % 25 == 0 or i == len(rows):
                print(f"  scored {i}/{len(rows)}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--round", choices=[*ROUNDS, "all"], default="pilot",
                    help="Gold round to score (default: pilot — the one the paper reports)")
    ap.add_argument("--rescore", action="store_true",
                    help="Re-run a judge over the records instead of using the shipped scores. "
                         "Needs PRISM_EVAL_MODEL / _BASE_URL / _API_KEY.")
    ap.add_argument("--model", default=os.environ.get("PRISM_EVAL_MODEL"),
                    help="Judge model name (default: $PRISM_EVAL_MODEL)")
    ap.add_argument("--workers", type=int, default=16, help="Concurrent judge calls")
    ap.add_argument("--limit", type=int, help="Only score the first N records (smoke test)")
    ap.add_argument("-o", "--output", type=Path, help="Write the full report as JSON here")
    args = ap.parse_args()

    rows = [json.loads(l) for l in SNAPSHOT.read_text().splitlines() if l.strip()]
    if args.round != "all":
        rows = [r for r in rows if r["round"] == args.round]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("No records selected.", file=sys.stderr)
        return 1

    label = "shipped judge"
    key = "judge"
    if args.rescore:
        if not args.model:
            ap.error("--rescore needs --model or $PRISM_EVAL_MODEL")
        print(f"Re-scoring {len(rows)} records with {args.model}...", file=sys.stderr)
        rescore(rows, args.model, args.workers)
        label, key = args.model, "rescored"

    doubly = [r for r in rows if len(r.get("human") or {}) >= 2]
    report = {
        "round": args.round,
        "n_records": len(rows),
        "n_double_annotated": len(doubly),
        "judge": label,
        "judge_vs_gold": _paired(rows, judge(key), gold),
        "human_vs_human": _paired(doubly, human("annotator_a"), human("annotator_b")),
        "judge_vs_human": {a: _paired(rows, human(a), judge(key)) for a in ANNOTATORS},
    }

    hdr = f"{'comparison':28}{'n':>6}{'kappa_w':>10}{'kappa_u':>10}{'AC2':>9}{'agree%':>9}"
    print(f"\nRound: {args.round}   records: {len(rows)}   judge: {label}\n")
    print(hdr)
    print("-" * len(hdr))

    def line(name: str, st: dict | None) -> None:
        if not st:
            return
        print(f"{name:28}{st['n_claims']:>6}{st['cohens_kappa_quadratic']:>10.4f}"
              f"{st['cohens_kappa_unweighted']:>10.4f}{st['gwets_ac2']:>9.4f}"
              f"{st['agreement_pct']:>9.1f}")

    line("judge vs GOLD", report["judge_vs_gold"])
    print("-" * len(hdr))
    line("human A vs human B", report["human_vs_human"])
    for a in ANNOTATORS:
        line(f"judge vs {a}", report["judge_vs_human"][a])

    ceiling = (report["human_vs_human"] or {}).get("cohens_kappa_quadratic")
    if ceiling is not None:
        print(f"\nPre-reconciliation human ceiling (weighted kappa): {ceiling:.4f}. "
              f"Read the judge rows against it, not against 1.0.")
    print("kappa_w = quadratic-weighted (what the paper reports); "
          "kappa_u = unweighted.")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
