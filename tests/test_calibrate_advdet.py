"""Tests for the agreement math in scripts/calibrate_advdet.py.

Covers:
  - _set_jaccard / _set_f1 / _set_jaccard edge cases (empty/empty etc).
  - _pooled_agreement on known-answer fixtures:
      perfect agreement; perfect disagreement; balanced random;
      the "kappa paradox" case (high p_obs, low Cohen K, high AC1).
  - _iaa and _identifier_vs_human surfacing pooled + per-record metrics.
  - calibrate() end-to-end on a tiny synthetic snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# scripts/ isn't a package; add the repo root and import via the module path.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_advdet import (  # noqa: E402
    _identifier_vs_human,
    _iaa,
    _pooled_agreement,
    _set_f1,
    _set_jaccard,
    calibrate,
)


# ─── Set helpers ─────────────────────────────────────────────────────────────


def test_jaccard_empty_empty_is_one() -> None:
    assert _set_jaccard(set(), set()) == 1.0


def test_jaccard_empty_vs_nonempty_is_zero() -> None:
    assert _set_jaccard(set(), {1}) == 0.0
    assert _set_jaccard({1}, set()) == 0.0


def test_jaccard_partial_overlap() -> None:
    assert _set_jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_set_f1_perfect() -> None:
    p, r, f = _set_f1({1, 2}, {1, 2})
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_set_f1_human_gold_superset() -> None:
    # Identifier picked {1,2}, gold (human) was {1,2,3}.
    # Identifier-as-pred: precision 1.0 (all picks were right), recall 2/3.
    p, r, f = _set_f1({1, 2}, {1, 2, 3})
    assert p == 1.0
    assert r == pytest.approx(2 / 3)
    assert f == pytest.approx(0.8)


# ─── _pooled_agreement: known-answer cases ───────────────────────────────────


def test_pooled_perfect_agreement() -> None:
    """Both raters pick {1, 3} on a 5-bullet record."""
    out = _pooled_agreement([({1, 3}, {1, 3}, 5)])
    assert out["n_items"] == 5
    assert out["p_obs"] == 1.0
    assert out["cohen_k"] == 1.0
    assert out["gwet_ac1"] == 1.0


def test_pooled_no_bullets_skipped() -> None:
    """Empty record (no GT bullets) should be skipped, not crash."""
    out = _pooled_agreement([(set(), set(), 0)])
    assert out["n_items"] == 0


def test_pooled_complementary_disagreement() -> None:
    """A picks {1,2}, B picks {3,4,5} on a 5-bullet record. Zero overlap."""
    out = _pooled_agreement([({1, 2}, {3, 4, 5}, 5)])
    # Items: 1,2 → (1,0), 3,4,5 → (0,1). Same label = none.
    assert out["n_items"] == 5
    assert out["p_obs"] == 0.0
    # pi_a = 2/5, pi_b = 3/5 → p_exp_kappa = 0.4*0.6 + 0.6*0.4 = 0.48
    # Cohen K = (0 - 0.48) / (1 - 0.48) = -12/13 ≈ -0.923
    assert out["cohen_k"] == pytest.approx(-0.48 / 0.52, abs=1e-6)
    # pi = 0.5 → p_exp_ac = 2*0.5*0.5 = 0.5
    # AC1 = (0 - 0.5) / 0.5 = -1.0
    assert out["gwet_ac1"] == pytest.approx(-1.0)


def test_pooled_kappa_paradox() -> None:
    """Classic kappa-paradox setup.

    20 records × 5 bullets = 100 items. Both raters pick 1 bullet per
    record. They agree on every item except disagree on 5 records out of 20
    (where one rater swaps which single bullet they tag).

    p_obs ≈ 0.94 (94/100 same), but pi_a ≈ pi_b ≈ 0.2 so Cohen K paradoxically
    drops despite very high observed agreement. Gwet AC1 is more stable.
    """
    pairs: list[tuple[set[int], set[int], int]] = []
    # 15 agree-records: both raters pick bullet 1.
    for _ in range(15):
        pairs.append(({1}, {1}, 5))
    # 5 disagree-records: rater A picks {1}, rater B picks {2}.
    # Per record: items 1,2 disagree (2 items differ), 3,4,5 match.
    for _ in range(5):
        pairs.append(({1}, {2}, 5))

    out = _pooled_agreement(pairs)
    # 100 items total. Same: 15*5 + 5*3 = 90. p_obs = 0.9.
    assert out["n_items"] == 100
    assert out["p_obs"] == pytest.approx(0.9)
    # pi_a = 20/100 = 0.2; pi_b = (15+5)/100 = 0.2.
    # p_exp_kappa = 0.2*0.2 + 0.8*0.8 = 0.68
    # Cohen K = (0.9 - 0.68) / (1 - 0.68) = 0.22/0.32 = 0.6875
    assert out["cohen_k"] == pytest.approx(0.6875)
    # pi = 0.2; p_exp_ac = 2*0.2*0.8 = 0.32
    # AC1 = (0.9 - 0.32) / 0.68 = 0.58/0.68 ≈ 0.853
    assert out["gwet_ac1"] == pytest.approx(0.58 / 0.68)
    # Both > 0 (agreement above chance under both formulations) and AC1 > K,
    # which is the textbook signature of the kappa paradox.
    assert out["gwet_ac1"] > out["cohen_k"]


def test_pooled_handles_all_zero_marginals() -> None:
    """Both raters say "nothing is adversarial" on every record."""
    pairs = [(set(), set(), 5)] * 10
    out = _pooled_agreement(pairs)
    assert out["p_obs"] == 1.0
    # pi_a = pi_b = 0. p_exp_kappa = 1, p_exp_ac = 0 → AC1 = 1.0.
    # cohen_k branches: p_exp >= 1, returns 1.0 since p_obs == 1.
    assert out["cohen_k"] == 1.0
    assert out["gwet_ac1"] == 1.0


# ─── _iaa + _identifier_vs_human integration ─────────────────────────────────


def _make_row(call_id: str, eval_id: str, setting: str,
              annotator: str, identifier: list[int], human: list[int],
              n_bullets: int = 5, stratum: str = "X") -> dict:
    return {
        "call_id": call_id,
        "eval_id": eval_id,
        "setting": setting,
        "annotator": annotator,
        "annotator_name": annotator,
        "stratum_label": stratum,
        "gt_bullets": [f"b{i}" for i in range(n_bullets)],
        "identifier_picks": identifier,
        "human_picks": human,
    }


def test_iaa_perfect_humans() -> None:
    """Two humans, perfect agreement on 3 traces."""
    rows = []
    for i, eid in enumerate(["e1", "e2", "e3"]):
        rows.append(_make_row(f"c{i}", eid, "AP", "u1",
                              identifier=[1, 2], human=[1, 2]))
        rows.append(_make_row(f"c{i}", eid, "AP", "u2",
                              identifier=[1, 2], human=[1, 2]))
    rows_by_call = {}
    for r in rows:
        rows_by_call.setdefault(r["call_id"], []).append(r)
    out = _iaa(rows_by_call)
    assert out["n_pairs"] == 3
    assert out["mean_jaccard"] == 1.0
    assert out["pooled"]["cohen_k"] == 1.0
    assert out["pooled"]["gwet_ac1"] == 1.0


def test_identifier_vs_human_matches_iaa_when_judge_is_a_human() -> None:
    """If the identifier's picks are identical to one of the humans'
    picks on every record, identifier↔human agreement should match
    that human's agreement with the others."""
    rows = []
    for i, eid in enumerate(["e1", "e2", "e3"]):
        rows.append(_make_row(f"c{i}", eid, "AP", "u1",
                              identifier=[1, 2], human=[1, 2]))
        rows.append(_make_row(f"c{i}", eid, "AP", "u2",
                              identifier=[1, 2], human=[1, 3]))
    ivh = _identifier_vs_human(rows)
    # Identifier picks == u1's picks → 3 perfect rows. Identifier vs u2:
    # {1,2} vs {1,3} on 3 records. Per-record F1 = 2*0.5*0.5/(0.5+0.5) = 0.5.
    # Mean F1 over 6 rows: (1+1+1+0.5+0.5+0.5)/6 = 0.75.
    assert ivh["mean_f1_identifier_vs_human"] == pytest.approx(0.75)


# ─── calibrate() end-to-end ──────────────────────────────────────────────────


def test_calibrate_produces_expected_top_level_shape() -> None:
    rows = [
        _make_row("c1", "AP-1", "AP", "u1",
                  identifier=[3, 4, 5], human=[3, 4], stratum="AP_K=2+"),
        _make_row("c1", "AP-1", "AP", "u2",
                  identifier=[3, 4, 5], human=[3, 4], stratum="AP_K=2+"),
        _make_row("c2", "AP-2", "AP", "u1",
                  identifier=[1], human=[1], stratum="AP_K=1"),
        _make_row("c2", "AP-2", "AP", "u2",
                  identifier=[1], human=[1], stratum="AP_K=1"),
        _make_row("c3", "BN-1", "BN", "u1",
                  identifier=[], human=[], stratum="AP_K=0"),
        _make_row("c3", "BN-1", "BN", "u2",
                  identifier=[], human=[], stratum="AP_K=0"),
    ]
    report = calibrate(rows)
    assert report["n_records"] == 6
    assert report["iaa"]["n_pairs"] == 3
    # Per-stratum keys present
    for k in ("AP_K=2+", "AP_K=1", "AP_K=0"):
        assert k in report["by_stratum"]
    # Per-setting keys present
    assert "AP" in report["by_setting"]
    assert "BN" in report["by_setting"]
    # Pooled stats are nested under iaa.pooled / identifier_vs_human.pooled
    assert "cohen_k" in report["iaa"]["pooled"]
    assert "gwet_ac1" in report["iaa"]["pooled"]
    assert "cohen_k" in report["identifier_vs_human"]["pooled"]
    assert "gwet_ac1" in report["identifier_vs_human"]["pooled"]
