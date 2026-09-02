"""The shipped judge-calibration gold set must reproduce its published numbers.

`data/calibration/` is a contribution in its own right: human labels for the
FOLLOWED judge, and the agreement statistics computed from them. Those numbers
are what justify trusting an LLM judge to score the benchmark at all, so they
have to be checkable rather than asserted.

These tests recompute the headline statistics straight from the raw labels and
compare against the frozen report. If either file is edited without the other,
they fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CALIB = Path(__file__).parent.parent / "data" / "calibration"
SNAPSHOT = CALIB / "follow_snapshot.jsonl"
FROZEN = CALIB / "follow_calibration.json"


def _rows() -> list[dict]:
    return [json.loads(line) for line in SNAPSHOT.read_text().splitlines() if line.strip()]


def _frozen() -> dict:
    return json.loads(FROZEN.read_text())


def _cohens_kappa(a: list[int], b: list[int]) -> tuple[float, float]:
    """Pooled two-rater kappa over binary labels. Returns (kappa, p_observed)."""
    n = len(a)
    p_obs = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    p_exp = pa * pb + (1 - pa) * (1 - pb)
    return (p_obs - p_exp) / (1 - p_exp), p_obs


def test_gold_set_shape():
    rows = _rows()
    frozen = _frozen()
    assert len(rows) == frozen["n_records"] == 184
    assert frozen["iaa"]["n_pairs"] == 92, "half the calls are double-annotated"


ANNOTATOR_NAMES = {"annotator_a", "annotator_b", "annotator_c", "annotator_excluded"}


def test_annotators_are_pseudonymous():
    """No real names and no account ids in a published human-label set."""
    for r in _rows():
        for field in ("annotator", "annotator_name", "wb_user_id"):
            v = r.get(field)
            if v is None:
                continue
            assert v in ANNOTATOR_NAMES, f"{field}={v!r} is not a pseudonym"


def test_judge_vs_human_kappa_matches_frozen_report():
    """Recompute the headline judge-human agreement from the raw labels."""
    a: list[int] = []
    b: list[int] = []
    for r in _rows():
        h, j = r.get("human_binary"), r.get("judge_followed_binary")
        if isinstance(h, list) and isinstance(j, list) and len(h) == len(j):
            a += [int(x) for x in h]
            b += [int(x) for x in j]

    kappa, p_obs = _cohens_kappa(a, b)
    pooled = _frozen()["judge_vs_human"]["pooled"]

    assert len(a) == pooled["n_items"] == 414
    assert kappa == pytest.approx(pooled["cohen_k"], abs=1e-9)
    assert p_obs == pytest.approx(pooled["p_obs"], abs=1e-9)


def test_published_agreement_is_substantial():
    """Guard the claim the benchmark rests on: the judge tracks humans about as
    well as humans track each other."""
    f = _frozen()
    human_human = f["iaa"]["pooled"]["cohen_k"]
    judge_human = f["judge_vs_human"]["pooled"]["cohen_k"]

    assert human_human == pytest.approx(0.7858, abs=5e-4)
    assert judge_human == pytest.approx(0.7341, abs=5e-4)
    # The judge should not be wildly better than the human ceiling — that would
    # mean the labels, not the judge, are the thing to distrust.
    assert judge_human <= human_human + 0.05


def test_strata_cover_all_three_sources():
    sources = {r.get("source") for r in _rows()}
    assert sources == {"bipia", "injecagent", "llmail"}
