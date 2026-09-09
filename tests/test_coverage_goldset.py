"""Structural checks for the scoring-judge annotation set.

`data/calibration/coverage_gold.jsonl` is the calibration behind the
*coverage* and *hallucination* metrics. Two annotators scored every claim
independently, then sat together and reconciled each disagreement into a single
**gold** label. Both the independent labels and the reconciled gold ship.

The tests validate the annotation schema, paper sample size, agreement-metric
implementation, and adversarial-identification labels without network access.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
CALIB = REPO / "data" / "calibration"
GOLD = CALIB / "coverage_gold.jsonl"
ADVDET = CALIB / "advdet_gold.jsonl"
ANNOTATORS = ("annotator_a", "annotator_b")
SCALE = {0.0, 0.5, 1.0}


def _load_tool():
    """Import scripts/calibrate_judge.py so tests exercise the shipped code."""
    spec = importlib.util.spec_from_file_location(
        "calibrate_judge", REPO / "scripts" / "calibrate_judge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _round(name: str) -> list[dict]:
    return [r for r in _rows() if r["round"] == name]


def test_gold_set_shape():
    rows = _rows()
    assert len(rows) == 93
    assert {r["round"] for r in rows} == set(TOOL.ROUNDS)
    assert {name: len(_round(name)) for name in TOOL.ROUNDS} == {
        "pilot": 49,
        "hard": 25,
        "hard2": 19,
    }


def test_every_record_has_gold_and_is_re_runnable():
    """Gold is the point; the rest is what a judge needs to be re-scored."""
    for row in _rows():
        assert row["gold"]["instruction_scores"], row["record_id"]
        for field in ("prompt", "model_response", "itm_report"):
            assert row[field], f"{row['record_id']} missing {field}"
        assert len(row["gold"]["instruction_scores"]) == len(row["gt_instructions"])


def test_labels_are_on_the_documented_scale():
    for row in _rows():
        for axis, scores in row["gold"].items():
            assert set(scores or []) <= SCALE, f"{row['record_id']}/gold/{axis}"
        for name, axes in (row.get("human") or {}).items():
            for axis, scores in axes.items():
                assert set(scores or []) <= SCALE, f"{row['record_id']}/{name}/{axis}"


def test_only_the_two_valid_annotators_appear():
    """No real names, and no third rater — they had 16 labels over 8 reports."""
    for row in _rows():
        for name in (row.get("human") or {}):
            assert name in ANNOTATORS, f"unexpected annotator {name!r}"


def test_nothing_requires_weave():
    """Calibration is file-based end to end — no pull, no account, no network."""
    src = (REPO / "scripts" / "calibrate_judge.py").read_text(encoding="utf-8")
    assert "import weave" not in src and "weave.init" not in src
    for row in _rows():
        assert "call_id" not in row and "weave_ref" not in row


def test_pilot_reproduces_the_paper_counts():
    """Paper Table 4 is the pilot round: 49 reports, 170 gold labels."""
    pilot = _round("pilot")
    assert len(pilot) == 49
    assert sum(len(r["gold"]["instruction_scores"]) for r in pilot) == 170


def test_pilot_inter_annotator_agreement_is_near_paper_value():
    doubly = [r for r in _round("pilot") if len(r.get("human") or {}) >= 2]
    got = TOOL._paired(doubly, TOOL.human("annotator_a"), TOOL.human("annotator_b"))
    assert got["n_claims"] == 170
    assert got["cohens_kappa_quadratic"] == pytest.approx(0.823, abs=0.002)


def test_judge_agrees_with_gold_above_the_calibration_gate():
    """The runbook gate was weighted kappa >= 0.60 before GRPO was allowed to run."""
    got = TOOL._paired(_round("pilot"), TOOL.judge(), TOOL.gold)
    assert got["n_claims"] == 170
    assert got["cohens_kappa_quadratic"] >= 0.60


def test_unweighted_kappa_is_reported_and_differs():
    """The weighted/unweighted gap is large enough that omitting it would mislead."""
    stats = TOOL._paired(_round("pilot"), TOOL.judge(), TOOL.gold)
    assert "cohens_kappa_unweighted" in stats
    assert stats["cohens_kappa_quadratic"] - stats["cohens_kappa_unweighted"] > 0.15


def test_advdet_gold_matches_paper_counts():
    rows = [
        json.loads(line)
        for line in ADVDET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 50
    assert sum(r["judge_matches_gold"] for r in rows) == 49
    for row in rows:
        assert set(row["gold_indices"]) <= set(range(1, (row["n_bullets"] or 0) + 1)), row["eval_id"]


def test_advdet_match_flag_is_consistent_with_the_indices():
    """judge_matches_gold must be exact-set equality, not a stored opinion."""
    for row in [
        json.loads(line)
        for line in ADVDET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]:
        expected = set(row["gold_indices"]) == set(row["judge_indices"])
        assert row["judge_matches_gold"] is expected, row["eval_id"]
