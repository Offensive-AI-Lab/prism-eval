#!/usr/bin/env python3
"""Calibrate the behavior judge's FOLLOWED label against human annotations.

Reads a JSONL snapshot of human FOLLOWED labels
(`data/calibration/follow_snapshot_v1.jsonl`) and
computes agreement between the judge's per-bullet FOLLOWED label and humans,
alongside the human-vs-human ceiling.

Each per-bullet FOLLOWED label is treated as a set of "followed" bullets (the
1-based indices labeled 1), so the same set-based agreement math as the
adversarial flow applies:

  - **Inter-annotator agreement (IAA)** — pairs of annotators on the same
    trace. Per pair: set Jaccard over followed bullets + pooled per-bullet
    Cohen's kappa / Gwet's AC1. The ceiling for how reliable the human signal
    itself is.
  - **Judge-vs-human agreement** — same metrics comparing the judge's followed
    set (`judge_picks`) to each human's (`human_picks`). How well the FOLLOWED
    judge matches a typical annotator.
  - **Per-source breakdown** — bipia / llmail / injecagent.
  - **Per-stratum breakdown** — *_followed vs *_refused strata.
  - **Worst-disagreement rows** for inspection.

Note vs the adversarial flow: there "positive" = adversarial bullet (rare), so
Gwet AC1 was the headline. Here "positive" = FOLLOWED bullet, which is common
(~56% in the xpia run), so Cohen's kappa is better behaved and both are
reported side by side.

Usage:
    python scripts/calibrate_follow.py \\
        -i results/follow/snapshot_v1.jsonl \\
        -o results/follow/calibration_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path


# ─── Agreement metrics ───────────────────────────────────────────────────────


def _set_jaccard(a: set[int], b: set[int]) -> float:
    """Set Jaccard. Empty/empty → 1.0; one empty + one non-empty → 0.0."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _set_f1(pred: set[int], gold: set[int]) -> tuple[float, float, float]:
    """Per-record (precision, recall, F1) treating gold as the reference.

    Here pred = judge's followed set, gold = human's followed set. Empty/empty
    (both say "nothing followed") is a perfect match.
    """
    if not pred and not gold:
        return (1.0, 1.0, 1.0)
    if not pred or not gold:
        return (0.0, 0.0, 0.0)
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return (p, r, f1)


def _pooled_agreement(pairs: list[tuple[set[int], set[int], int]]) -> dict:
    """Pooled item-level agreement over (a_picks, b_picks, n_bullets) tuples.

    Pools every (record, bullet) into a single binary item population where the
    label is "followed", then computes observed agreement, Cohen's kappa, and
    Gwet's AC1. Skips records with n_bullets == 0.
    """
    n_items = 0
    n_same = 0
    n_pos_a = 0
    n_pos_b = 0
    for a, b, n_bullets in pairs:
        if n_bullets <= 0:
            continue
        for i in range(1, n_bullets + 1):
            a_pos = i in a
            b_pos = i in b
            n_items += 1
            if a_pos == b_pos:
                n_same += 1
            if a_pos:
                n_pos_a += 1
            if b_pos:
                n_pos_b += 1

    if n_items == 0:
        return {
            "p_obs": 1.0,
            "cohen_k": 1.0,
            "gwet_ac1": 1.0,
            "n_items": 0,
            "pi_a": 0.0,
            "pi_b": 0.0,
        }

    p_obs = n_same / n_items
    pi_a = n_pos_a / n_items
    pi_b = n_pos_b / n_items

    # Cohen's kappa: expected agreement from each rater's marginal.
    p_exp_kappa = pi_a * pi_b + (1 - pi_a) * (1 - pi_b)
    if p_exp_kappa >= 1.0:
        cohen_k = 1.0 if p_obs >= 1.0 else 0.0
    else:
        cohen_k = (p_obs - p_exp_kappa) / (1.0 - p_exp_kappa)

    # Gwet's AC1: expected agreement from pooled prevalence.
    pi = (pi_a + pi_b) / 2.0
    p_exp_ac = 2.0 * pi * (1.0 - pi)
    if p_exp_ac >= 1.0:
        gwet_ac1 = 1.0 if p_obs >= 1.0 else 0.0
    else:
        gwet_ac1 = (p_obs - p_exp_ac) / (1.0 - p_exp_ac)

    return {
        "p_obs": p_obs,
        "cohen_k": cohen_k,
        "gwet_ac1": gwet_ac1,
        "n_items": n_items,
        "pi_a": pi_a,
        "pi_b": pi_b,
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ─── Snapshot loading + grouping ─────────────────────────────────────────────


def _load_snapshot(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_by_call(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["call_id"]].append(r)
    return dict(out)


# ─── IAA + judge-vs-human ────────────────────────────────────────────────────


def _iaa(rows_by_call: dict[str, list[dict]]) -> dict:
    """Human ↔ human agreement over followed sets."""
    pair_jaccards: list[float] = []
    pool_pairs: list[tuple[set[int], set[int], int]] = []
    n_pairs = 0
    n_calls_with_pairs = 0

    for _call_id, ann_rows in rows_by_call.items():
        if len(ann_rows) < 2:
            continue
        n_calls_with_pairs += 1
        n_bullets = len(ann_rows[0].get("gt_bullets") or [])
        for a, b in combinations(ann_rows, 2):
            sa, sb = set(a["human_picks"]), set(b["human_picks"])
            pair_jaccards.append(_set_jaccard(sa, sb))
            pool_pairs.append((sa, sb, n_bullets))
            n_pairs += 1

    pooled = _pooled_agreement(pool_pairs)
    return {
        "n_calls_with_pairs": n_calls_with_pairs,
        "n_pairs": n_pairs,
        "mean_jaccard": _mean(pair_jaccards),
        "pooled": pooled,
    }


def _judge_vs_human(rows: list[dict]) -> dict:
    """Judge ↔ human agreement over followed sets (symmetric with `_iaa`)."""
    jacc: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    pool_pairs: list[tuple[set[int], set[int], int]] = []

    for r in rows:
        jdg = set(r.get("judge_picks") or [])
        hum = set(r.get("human_picks") or [])
        n_bullets = len(r.get("gt_bullets") or [])
        jacc.append(_set_jaccard(jdg, hum))
        p, rcl, f1 = _set_f1(jdg, hum)
        precisions.append(p)
        recalls.append(rcl)
        f1s.append(f1)
        pool_pairs.append((jdg, hum, n_bullets))

    pooled = _pooled_agreement(pool_pairs)
    return {
        "n": len(rows),
        "mean_jaccard": _mean(jacc),
        "mean_precision_judge_vs_human": _mean(precisions),
        "mean_recall_judge_vs_human": _mean(recalls),
        "mean_f1_judge_vs_human": _mean(f1s),
        "pooled": pooled,
    }


def _exact_agreement_share(rows: list[dict]) -> float:
    """Fraction of (call, annotator) pairs where judge == human exactly."""
    if not rows:
        return 0.0
    eq = sum(
        1
        for r in rows
        if set(r.get("judge_picks") or []) == set(r.get("human_picks") or [])
    )
    return eq / len(rows)


# ─── Reporting ───────────────────────────────────────────────────────────────


def _breakdown(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get(key) or "_unknown"].append(r)
    out: dict[str, dict] = {}
    for label, grp in groups.items():
        out[label] = {
            "n": len(grp),
            "judge_vs_human": _judge_vs_human(grp),
            "exact_agreement_share": _exact_agreement_share(grp),
            "iaa": _iaa(_group_by_call(grp)),
        }
    return out


def calibrate(rows: list[dict]) -> dict:
    rows_by_call = _group_by_call(rows)

    worst: list[dict] = []
    for r in rows:
        jdg = set(r.get("judge_picks") or [])
        hum = set(r.get("human_picks") or [])
        _, _, f1 = _set_f1(jdg, hum)
        if f1 < 0.5:
            worst.append(
                {
                    "eval_id": r["eval_id"],
                    "source": r.get("source"),
                    "annotator": r.get("annotator_name") or r["annotator"],
                    "stratum": r.get("stratum_label"),
                    "judge_picks": sorted(jdg),
                    "human_picks": sorted(hum),
                    "f1": round(f1, 3),
                }
            )
    worst.sort(key=lambda x: x["f1"])

    return {
        "n_records": len(rows),
        "iaa": _iaa(rows_by_call),
        "judge_vs_human": _judge_vs_human(rows),
        "exact_agreement_share": _exact_agreement_share(rows),
        "by_source": _breakdown(rows, "source"),
        "by_stratum": _breakdown(rows, "stratum_label"),
        "worst_disagreements": worst[:25],
    }


def _print_summary(report: dict) -> None:
    iaa = report["iaa"]
    jvh = report["judge_vs_human"]
    iaa_p = iaa["pooled"]
    jvh_p = jvh["pooled"]
    print(f"\nRecords: {report['n_records']}", file=sys.stderr)
    print(
        f"Annotator-pair calls: {iaa['n_calls_with_pairs']} "
        f"({iaa['n_pairs']} pairs)",
        file=sys.stderr,
    )

    print("\nPooled bullet-level FOLLOWED agreement (chance-corrected):", file=sys.stderr)
    print(
        f"  {'':20s} {'humans ↔ humans':>20s}  {'judge ↔ humans':>22s}",
        file=sys.stderr,
    )
    print(
        f"  {'n_items (pooled)':20s} {iaa_p['n_items']:>20d}  "
        f"{jvh_p['n_items']:>22d}",
        file=sys.stderr,
    )
    print(
        f"  {'observed agreement':20s} {iaa_p['p_obs']:>20.3f}  "
        f"{jvh_p['p_obs']:>22.3f}",
        file=sys.stderr,
    )
    print(
        f"  {'Cohen K':20s} {iaa_p['cohen_k']:>20.3f}  {jvh_p['cohen_k']:>22.3f}",
        file=sys.stderr,
    )
    print(
        f"  {'Gwet AC1':20s} {iaa_p['gwet_ac1']:>20.3f}  {jvh_p['gwet_ac1']:>22.3f}",
        file=sys.stderr,
    )
    print(
        f"  {'followed rate A':20s} {iaa_p['pi_a']:>20.3f}  {jvh_p['pi_a']:>22.3f}"
        "    (A = judge on right side)",
        file=sys.stderr,
    )
    print(
        f"  {'followed rate B':20s} {iaa_p['pi_b']:>20.3f}  {jvh_p['pi_b']:>22.3f}"
        "    (B = human on right side)",
        file=sys.stderr,
    )

    print("\nPer-record metrics (judge ↔ humans):", file=sys.stderr)
    print(f"  mean Jaccard:        {jvh['mean_jaccard']:.3f}", file=sys.stderr)
    print(
        f"  mean precision:      {jvh['mean_precision_judge_vs_human']:.3f}",
        file=sys.stderr,
    )
    print(
        f"  mean recall:         {jvh['mean_recall_judge_vs_human']:.3f}",
        file=sys.stderr,
    )
    print(f"  mean F1:             {jvh['mean_f1_judge_vs_human']:.3f}", file=sys.stderr)
    print(
        f"  exact-match share:   {report['exact_agreement_share']:.3f}",
        file=sys.stderr,
    )
    print(f"  (IAA mean Jaccard:   {iaa['mean_jaccard']:.3f})", file=sys.stderr)

    for title, key in (("By source", "by_source"), ("By stratum", "by_stratum")):
        print(f"\n{title}:", file=sys.stderr)
        print(
            f"  {'group':20s} {'n':>3s}  {'IAA K':>6s} {'IAA AC1':>8s}  "
            f"{'jdg↔hum K':>10s} {'jdg↔hum AC1':>12s}  {'jdg↔hum F1':>11s}",
            file=sys.stderr,
        )
        for label, s in sorted(report[key].items()):
            jvh_s = s["judge_vs_human"]
            jvh_sp = jvh_s["pooled"]
            iaa_sp = s["iaa"]["pooled"]
            print(
                f"  {label:20s} {s['n']:>3d}  "
                f"{iaa_sp['cohen_k']:>6.3f} {iaa_sp['gwet_ac1']:>8.3f}  "
                f"{jvh_sp['cohen_k']:>10.3f} {jvh_sp['gwet_ac1']:>12.3f}  "
                f"{jvh_s['mean_f1_judge_vs_human']:>11.3f}",
                file=sys.stderr,
            )

    worst = report["worst_disagreements"]
    if worst:
        print(
            f"\nWorst {min(len(worst), 10)} disagreements (F1 < 0.5):",
            file=sys.stderr,
        )
        for w in worst[:10]:
            print(
                f"  {w['eval_id']:22s} {str(w['source']):10s} "
                f"stratum={str(w['stratum']):20s} F1={w['f1']:.2f}  "
                f"judge={w['judge_picks']} human={w['human_picks']} "
                f"({w['annotator']})",
                file=sys.stderr,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--snapshot",
        type=Path,
        required=True,
        help="JSONL snapshot of human FOLLOWED labels.",
    )
    parser.add_argument(
        "-o", "--out", type=Path, help="Optional JSON report path."
    )
    args = parser.parse_args()

    rows = _load_snapshot(args.snapshot)
    if not rows:
        print(f"ERROR: {args.snapshot} is empty.", file=sys.stderr)
        return 1

    report = calibrate(rows)
    _print_summary(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
