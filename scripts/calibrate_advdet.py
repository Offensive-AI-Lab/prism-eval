#!/usr/bin/env python3
"""Calibrate the adversarial-bullet identifier against human annotations.

Reads a JSONL snapshot of human adversarial-detection labels
and computes:

  - **Inter-annotator agreement (IAA)** — pairs of annotators on the same
    trace. For each pair: set-level Jaccard between their picked indices,
    and per-bullet Cohen's kappa over the GT bullets. Averaged across
    pairs; a baseline for how reliable the human signal itself is.
  - **Identifier-vs-human agreement** — same metrics, comparing the
    identifier's picks (`identifier_picks`) to each human's picks
    (`human_picks`). Tells us how well the identifier matches a typical
    annotator.
  - **Per-stratum breakdown** — agreement scores grouped by the
    `stratum_label` carried in the trace attributes. Shows whether the
    identifier fails worse on benign-control rows, K=1, K=2+, etc.
  - **Per-setting breakdown** — AP vs HO vs BC vs BN.

Writes a JSON report and prints a human-readable summary.

Usage:
    python scripts/calibrate_advdet.py \\
        -i results/advdet/snapshot_v1.jsonl \\
        -o results/advdet/calibration_v1.json
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
    """Per-record (precision, recall, F1) treating gold as the reference."""
    if not pred and not gold:
        return (1.0, 1.0, 1.0)
    if not pred or not gold:
        return (0.0, 0.0, 0.0)
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return (p, r, f1)


def _pooled_agreement(
    pairs: list[tuple[set[int], set[int], int]],
) -> dict:
    """Pooled item-level agreement stats over a list of (a_picks, b_picks,
    n_bullets) tuples — each tuple is one record / pair of raters.

    Pools every (record, bullet) into a single item population, then
    computes:
      - p_obs:     observed binary agreement (% of items where both raters
                   gave the same label).
      - cohen_k:   Cohen's kappa with per-rater marginals from the pool.
      - gwet_ac1:  Gwet's AC1 (binary unordered = AC2 with identity
                   weights). Less prone to the "kappa paradox" when most
                   items are negative (typical here: few bullets per
                   record are tagged adversarial).
      - n_items:   total (record × bullet) items in the pool.
      - pi_a, pi_b: positive-class marginals per rater (in the pool).

    Skips records with n_bullets == 0 (vacuous).
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
        return {"p_obs": 1.0, "cohen_k": 1.0, "gwet_ac1": 1.0,
                "n_items": 0, "pi_a": 0.0, "pi_b": 0.0}

    p_obs = n_same / n_items
    pi_a = n_pos_a / n_items
    pi_b = n_pos_b / n_items

    # Cohen's kappa: expected agreement uses each rater's marginal.
    p_exp_kappa = pi_a * pi_b + (1 - pi_a) * (1 - pi_b)
    if p_exp_kappa >= 1.0:
        cohen_k = 1.0 if p_obs >= 1.0 else 0.0
    else:
        cohen_k = (p_obs - p_exp_kappa) / (1.0 - p_exp_kappa)

    # Gwet's AC1: expected agreement uses the pooled prevalence.
    # For binary: p_e = 2 * pi * (1 - pi) where pi = (pi_a + pi_b) / 2.
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
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _group_by_call(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["call_id"]].append(r)
    return dict(out)


# ─── IAA + identifier-vs-human ───────────────────────────────────────────────


def _iaa(rows_by_call: dict[str, list[dict]]) -> dict:
    """Human ↔ human agreement.

    Two layers:
      - **Per-record Jaccard**: mean over annotator pairs of set-Jaccard
        between their pick sets.
      - **Pooled bullet-level**: every (call, pair, bullet) becomes one
        binary item; computes Cohen's K + Gwet's AC1 on the pool.
    """
    pair_jaccards: list[float] = []
    pool_pairs: list[tuple[set[int], set[int], int]] = []
    n_pairs = 0
    n_calls_with_pairs = 0

    for call_id, ann_rows in rows_by_call.items():
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


def _identifier_vs_human(rows: list[dict]) -> dict:
    """Identifier ↔ human agreement.

    Two layers (symmetric with `_iaa` so the numbers compare apples to
    apples):
      - **Per-record**: mean Jaccard + mean precision/recall/F1 (treating
        the human's picks as gold).
      - **Pooled bullet-level**: every (row, bullet) becomes one binary
        item where rater_A = identifier and rater_B = that human;
        computes Cohen's K + Gwet's AC1 on the pool.
    """
    jacc: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    pool_pairs: list[tuple[set[int], set[int], int]] = []

    for r in rows:
        idf = set(r.get("identifier_picks") or [])
        hum = set(r.get("human_picks") or [])
        n_bullets = len(r.get("gt_bullets") or [])
        jacc.append(_set_jaccard(idf, hum))
        p, rcl, f1 = _set_f1(idf, hum)
        precisions.append(p)
        recalls.append(rcl)
        f1s.append(f1)
        pool_pairs.append((idf, hum, n_bullets))

    pooled = _pooled_agreement(pool_pairs)
    return {
        "n": len(rows),
        "mean_jaccard": _mean(jacc),
        "mean_precision_identifier_vs_human": _mean(precisions),
        "mean_recall_identifier_vs_human": _mean(recalls),
        "mean_f1_identifier_vs_human": _mean(f1s),
        "pooled": pooled,
    }


def _exact_agreement_share(rows: list[dict]) -> float:
    """Fraction of (call, annotator) pairs where identifier == human exactly."""
    if not rows:
        return 0.0
    eq = sum(
        1 for r in rows
        if set(r.get("identifier_picks") or []) == set(r.get("human_picks") or [])
    )
    return eq / len(rows)


# ─── Reporting ───────────────────────────────────────────────────────────────


def _is_gold_schema(rows: list[dict]) -> bool:
    return bool(rows) and "gold_indices" in rows[0]


def calibrate_gold(rows: list[dict]) -> dict:
    """Reconciled-gold snapshot (advdet_gold_v1.jsonl): identifier vs gold.

    Gold rows carry one reconciled ``gold_indices`` set per record (no
    per-annotator rows), so there is no IAA to compute — the report covers
    the identifier against gold only.
    """
    def _stats(srows: list[dict]) -> dict:
        jacc, f1s = [], []
        exact = 0
        for r in srows:
            idf = set(r.get("judge_indices") or [])
            gold = set(r.get("gold_indices") or [])
            jacc.append(_set_jaccard(idf, gold))
            _, _, f1 = _set_f1(idf, gold)
            f1s.append(f1)
            exact += int(idf == gold)
        return {
            "n": len(srows),
            "exact_match": exact,
            "exact_match_share": exact / len(srows) if srows else 0.0,
            "mean_jaccard": _mean(jacc),
            "mean_f1": _mean(f1s),
        }

    by_stratum = {label: _stats(srows) for label, srows in _group_field(rows, "stratum_label").items()}
    by_setting = {s: _stats(srows) for s, srows in _group_field(rows, "setting").items()}
    worst = [
        {"eval_id": r["eval_id"], "setting": r.get("setting"),
         "stratum": r.get("stratum_label"),
         "judge_indices": r.get("judge_indices"), "gold_indices": r.get("gold_indices")}
        for r in rows if set(r.get("judge_indices") or []) != set(r.get("gold_indices") or [])
    ]
    return {
        "schema": "reconciled_gold",
        "n_records": len(rows),
        "identifier_vs_gold": _stats(rows),
        "by_stratum": by_stratum,
        "by_setting": by_setting,
        "disagreements": worst,
    }


def _group_field(rows: list[dict], field: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r.get(field) or "_unknown"].append(r)
    return dict(out)


def _print_gold_summary(report: dict) -> None:
    s = report["identifier_vs_gold"]
    print(f"\nRecords: {report['n_records']} (reconciled gold)", file=sys.stderr)
    print(f"Identifier matches gold exactly on {s['exact_match']}/{s['n']} "
          f"({s['exact_match_share']:.3f})", file=sys.stderr)
    print(f"mean Jaccard {s['mean_jaccard']:.3f} | mean F1 {s['mean_f1']:.3f}", file=sys.stderr)
    for r in report["disagreements"]:
        print(f"  disagreement: {r['eval_id']} judge={r['judge_indices']} gold={r['gold_indices']}", file=sys.stderr)


def calibrate(rows: list[dict]) -> dict:
    rows_by_call = _group_by_call(rows)

    overall_iaa = _iaa(rows_by_call)
    overall_ivh = _identifier_vs_human(rows)
    overall_exact = _exact_agreement_share(rows)

    # Per-stratum
    rows_by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rows_by_stratum[r.get("stratum_label") or "_unknown"].append(r)
    by_stratum: dict[str, dict] = {}
    for label, srows in rows_by_stratum.items():
        by_stratum[label] = {
            "n": len(srows),
            "identifier_vs_human": _identifier_vs_human(srows),
            "exact_agreement_share": _exact_agreement_share(srows),
            "iaa": _iaa(_group_by_call(srows)),
        }

    # Per-setting
    rows_by_setting: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rows_by_setting[r.get("setting") or "_unknown"].append(r)
    by_setting: dict[str, dict] = {}
    for s, srows in rows_by_setting.items():
        by_setting[s] = {
            "n": len(srows),
            "identifier_vs_human": _identifier_vs_human(srows),
            "exact_agreement_share": _exact_agreement_share(srows),
            "iaa": _iaa(_group_by_call(srows)),
        }

    # Worst-disagreement rows (identifier vs human F1 < 0.5)
    worst: list[dict] = []
    for r in rows:
        idf = set(r.get("identifier_picks") or [])
        hum = set(r.get("human_picks") or [])
        _, _, f1 = _set_f1(idf, hum)
        if f1 < 0.5:
            worst.append({
                "eval_id": r["eval_id"],
                "setting": r["setting"],
                "annotator": r.get("annotator_name") or r["annotator"],
                "stratum": r.get("stratum_label"),
                "identifier_picks": r["identifier_picks"],
                "human_picks": r["human_picks"],
                "f1": round(f1, 3),
            })
    worst.sort(key=lambda x: x["f1"])

    return {
        "n_records": len(rows),
        "iaa": overall_iaa,
        "identifier_vs_human": overall_ivh,
        "exact_agreement_share": overall_exact,
        "by_stratum": by_stratum,
        "by_setting": by_setting,
        "worst_disagreements": worst[:25],
    }


def _print_summary(report: dict) -> None:
    iaa = report["iaa"]
    ivh = report["identifier_vs_human"]
    iaa_p = iaa["pooled"]
    ivh_p = ivh["pooled"]
    print(f"\nRecords: {report['n_records']}", file=sys.stderr)
    print(f"Annotator-pair calls: {iaa['n_calls_with_pairs']} "
          f"({iaa['n_pairs']} pairs)", file=sys.stderr)

    # Side-by-side bullet-level pooled agreement: this is the apples-to-
    # apples comparison the user reads first. "Is the identifier as good
    # as a human?" → compare these two columns.
    print("\nPooled bullet-level agreement (chance-corrected):", file=sys.stderr)
    print(f"  {'':20s} {'humans ↔ humans':>20s}  {'identifier ↔ humans':>22s}",
          file=sys.stderr)
    print(f"  {'n_items (pooled)':20s} {iaa_p['n_items']:>20d}  "
          f"{ivh_p['n_items']:>22d}", file=sys.stderr)
    print(f"  {'observed agreement':20s} {iaa_p['p_obs']:>20.3f}  "
          f"{ivh_p['p_obs']:>22.3f}", file=sys.stderr)
    print(f"  {'Cohen K':20s} {iaa_p['cohen_k']:>20.3f}  "
          f"{ivh_p['cohen_k']:>22.3f}", file=sys.stderr)
    print(f"  {'Gwet AC1':20s} {iaa_p['gwet_ac1']:>20.3f}  "
          f"{ivh_p['gwet_ac1']:>22.3f}", file=sys.stderr)
    print(f"  {'rater A pos rate':20s} {iaa_p['pi_a']:>20.3f}  "
          f"{ivh_p['pi_a']:>22.3f}    (A = identifier on right side)",
          file=sys.stderr)
    print(f"  {'rater B pos rate':20s} {iaa_p['pi_b']:>20.3f}  "
          f"{ivh_p['pi_b']:>22.3f}    (B = human on right side)",
          file=sys.stderr)

    print("\nPer-record metrics (identifier ↔ humans):", file=sys.stderr)
    print(f"  mean Jaccard:        {ivh['mean_jaccard']:.3f}", file=sys.stderr)
    print(f"  mean precision:      {ivh['mean_precision_identifier_vs_human']:.3f}",
          file=sys.stderr)
    print(f"  mean recall:         {ivh['mean_recall_identifier_vs_human']:.3f}",
          file=sys.stderr)
    print(f"  mean F1:             {ivh['mean_f1_identifier_vs_human']:.3f}",
          file=sys.stderr)
    print(f"  exact-match share:   {report['exact_agreement_share']:.3f}",
          file=sys.stderr)
    print(f"  (IAA mean Jaccard:   {iaa['mean_jaccard']:.3f})", file=sys.stderr)

    print("\nBy stratum:", file=sys.stderr)
    print(f"  {'stratum':14s} {'n':>3s}  {'IAA K':>6s} {'IAA AC1':>8s}  "
          f"{'id↔hum K':>9s} {'id↔hum AC1':>11s}  {'id↔hum F1':>10s}",
          file=sys.stderr)
    for label, s in sorted(report["by_stratum"].items()):
        ivh_s = s["identifier_vs_human"]
        ivh_sp = ivh_s["pooled"]
        iaa_s = s["iaa"]
        iaa_sp = iaa_s["pooled"]
        print(f"  {label:14s} {s['n']:>3d}  "
              f"{iaa_sp['cohen_k']:>6.3f} {iaa_sp['gwet_ac1']:>8.3f}  "
              f"{ivh_sp['cohen_k']:>9.3f} {ivh_sp['gwet_ac1']:>11.3f}  "
              f"{ivh_s['mean_f1_identifier_vs_human']:>10.3f}",
              file=sys.stderr)

    print("\nBy setting:", file=sys.stderr)
    print(f"  {'setting':9s} {'n':>3s}  {'IAA K':>6s} {'IAA AC1':>8s}  "
          f"{'id↔hum K':>9s} {'id↔hum AC1':>11s}  {'id↔hum F1':>10s}",
          file=sys.stderr)
    for s, v in sorted(report["by_setting"].items()):
        ivh_v = v["identifier_vs_human"]
        ivh_vp = ivh_v["pooled"]
        iaa_v = v["iaa"]
        iaa_vp = iaa_v["pooled"]
        print(f"  [{s}]      {v['n']:>3d}  "
              f"{iaa_vp['cohen_k']:>6.3f} {iaa_vp['gwet_ac1']:>8.3f}  "
              f"{ivh_vp['cohen_k']:>9.3f} {ivh_vp['gwet_ac1']:>11.3f}  "
              f"{ivh_v['mean_f1_identifier_vs_human']:>10.3f}",
              file=sys.stderr)

    worst = report["worst_disagreements"]
    if worst:
        print(f"\nWorst {min(len(worst), 10)} disagreements (F1 < 0.5):",
              file=sys.stderr)
        for w in worst[:10]:
            print(f"  {w['eval_id']:15s} {w['setting']:3s} "
                  f"stratum={w['stratum']:11s} F1={w['f1']:.2f}  "
                  f"id={w['identifier_picks']} human={w['human_picks']} "
                  f"({w['annotator']})", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--snapshot", type=Path, required=True,
                        help="JSONL snapshot: per-annotator rows, or the reconciled "
                             "gold set (data/calibration/advdet_gold_v1.jsonl).")
    parser.add_argument("-o", "--out", type=Path,
                        help="Optional JSON report path.")
    args = parser.parse_args()

    rows = _load_snapshot(args.snapshot)
    if not rows:
        print(f"ERROR: {args.snapshot} is empty.", file=sys.stderr)
        return 1

    if _is_gold_schema(rows):
        report = calibrate_gold(rows)
        _print_gold_summary(report)
    else:
        report = calibrate(rows)
        _print_summary(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
