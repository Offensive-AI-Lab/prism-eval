#!/usr/bin/env python
"""Build donor-prompt pairs for the swapped act-context ablation.

For each single-turn record in the suite, pick a donor record from the same
setting (seeded shuffle + rotate-by-one, so nobody is their own donor). At
extraction time (PRISM_EVAL_ACT_CONTEXT=swapped) the donor's prompt is placed
behind the *fixed* base response; whichever instruction the ITM report then
names tells us whether the decoder reads response-state content (original)
or attention-routed prompt content (donor).

Outputs:
  --pairs-out        {"pairs": {eval_id: {donor_eval_id, donor_prompt, setting}}}
                     — consumed by the runner via PRISM_EVAL_ACT_SWAP_FILE.
  --donor-suite-out  (optional) suite JSON where each record keeps its own
                     eval_id/prompt but carries the donor's ground-truth
                     annotations. Scoring a swapped run against this suite
                     (prism-eval score --suite <donor suite>) measures how much
                     of the report matches the *donor* instruction — the
                     directional readout.

Usage:
  python scripts/make_act_swap_pairs.py \
      --suite data/eval_suite_v2_final.json \
      --pairs-out results/act_context_ablation/swap_pairs.json \
      --donor-suite-out results/act_context_ablation/donor_suite.json
"""

from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import click

# Ground-truth fields transplanted from the donor record into the donor suite.
DONOR_ANNOTATION_FIELDS = ("instruction_sources", "probe_points", "constraints", "goals")


def build_pairs(records: list[dict], seed: int) -> dict[str, dict]:
    """eval_id -> donor mapping; within-setting derangement via rotate-by-one."""
    by_setting: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_setting[rec["setting"]].append(rec)

    pairs: dict[str, dict] = {}
    rng = random.Random(seed)
    for setting in sorted(by_setting):
        group = sorted(by_setting[setting], key=lambda r: r["eval_id"])
        if len(group) < 2:
            click.echo(f"WARN: setting {setting} has {len(group)} record(s) — skipped")
            continue
        rng.shuffle(group)
        for i, rec in enumerate(group):
            donor = group[(i + 1) % len(group)]
            pairs[rec["eval_id"]] = {
                "donor_eval_id": donor["eval_id"],
                "donor_prompt": donor["prompt"],
                "setting": setting,
            }
    return pairs


@click.command()
@click.option("--suite", "suite_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--pairs-out", type=click.Path(path_type=Path), required=True)
@click.option("--donor-suite-out", type=click.Path(path_type=Path), default=None)
@click.option("--seed", type=int, default=0, show_default=True)
def main(suite_path: Path, pairs_out: Path, donor_suite_out: Path | None, seed: int) -> None:
    raw = json.loads(suite_path.read_text())
    evals = raw["evals"]

    single_turn = [r for r in evals if r.get("prompt")]
    skipped = len(evals) - len(single_turn)
    if skipped:
        click.echo(f"WARN: {skipped} multi-turn/promptless record(s) excluded from pairing")

    pairs = build_pairs(single_turn, seed)
    assert all(eid != p["donor_eval_id"] for eid, p in pairs.items()), "self-pair found"

    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    pairs_out.write_text(json.dumps(
        {"_suite": str(suite_path), "_seed": seed, "pairs": pairs},
        indent=2, ensure_ascii=False,
    ))
    click.echo(f"Wrote {len(pairs)} pairs -> {pairs_out}")

    if donor_suite_out is not None:
        by_id = {r["eval_id"]: r for r in evals}
        donor_evals = []
        for rec in single_turn:
            pair = pairs.get(rec["eval_id"])
            if pair is None:
                continue
            donor = by_id[pair["donor_eval_id"]]
            swapped = copy.deepcopy(rec)
            for field in DONOR_ANNOTATION_FIELDS:
                if field in swapped or field in donor:
                    swapped[field] = copy.deepcopy(donor.get(field))
            donor_evals.append(swapped)

        donor_suite = {k: v for k, v in raw.items() if k != "evals"}
        donor_suite["_derived_from"] = str(suite_path)
        donor_suite["_swap_seed"] = seed
        donor_suite["_note"] = (
            "Donor-annotation suite for the swapped act-context ablation: each "
            "record keeps its own eval_id but carries the donor record's "
            "ground-truth annotations. Use with `prism-eval score` on a swapped "
            "run to measure report-vs-donor-instruction match."
        )
        donor_suite["evals"] = donor_evals
        donor_suite_out.parent.mkdir(parents=True, exist_ok=True)
        donor_suite_out.write_text(json.dumps(donor_suite, indent=2, ensure_ascii=False))
        click.echo(f"Wrote donor suite ({len(donor_evals)} records) -> {donor_suite_out}")


if __name__ == "__main__":
    main()
