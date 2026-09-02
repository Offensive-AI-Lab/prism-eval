"""Tests for prism_eval.loader."""

from pathlib import Path

from prism_eval.loader import filter_evals, load_suite
from prism_eval.schema import EvalRecord

CANONICAL_SUITE = Path(__file__).parent.parent / "data" / "eval_suite.json"


def _make_record(**overrides) -> EvalRecord:
    base = dict(
        eval_id="T-001",
        setting="BC",
        category="Test",
        structural_tags=["single-turn"],
        difficulty="easy",
        source="hand_crafted",
        prompt="test",
        instruction_sources={"original": ["x"]},
    )
    base.update(overrides)
    return EvalRecord(**base)


class TestCanonicalSuite:
    """Integrity of the shipped 1000-record suite.

    Every published number is computed against this exact file, so a silent
    edit would invalidate the paper's results. These assertions are the
    tripwire.
    """

    def test_loads(self):
        suite = load_suite(CANONICAL_SUITE)
        assert len(suite.evals) == 1000
        assert suite.schema_version == "2.0"

    def test_balanced_across_settings(self):
        suite = load_suite(CANONICAL_SUITE)
        counts: dict[str, int] = {}
        for r in suite.evals:
            counts[r.setting] = counts.get(r.setting, 0) + 1
        assert counts == {"AP": 250, "HO": 250, "BC": 250, "BN": 250}

    def test_eval_ids_unique(self):
        suite = load_suite(CANONICAL_SUITE)
        ids = [r.eval_id for r in suite.evals]
        assert len(set(ids)) == len(ids)

    def test_every_record_has_ground_truth(self):
        suite = load_suite(CANONICAL_SUITE)
        for r in suite.evals:
            assert r.instruction_sources is not None, r.eval_id
            assert r.instruction_sources.get("original"), r.eval_id


class TestFilterEvals:
    def test_filter_by_setting(self):
        records = [
            _make_record(eval_id="A", setting="BC"),
            _make_record(eval_id="B", setting="S1"),
            _make_record(eval_id="C", setting="BN"),
        ]
        result = filter_evals(records, settings=["BC", "BN"])
        assert len(result) == 2
        assert {r.eval_id for r in result} == {"A", "C"}

    def test_filter_by_tags(self):
        records = [
            _make_record(eval_id="A", structural_tags=["single-turn", "explicit"]),
            _make_record(eval_id="B", structural_tags=["multi-turn:persistence"]),
        ]
        result = filter_evals(records, tags=["single-turn"])
        assert len(result) == 1
        assert result[0].eval_id == "A"

    def test_filter_by_source(self):
        records = [
            _make_record(eval_id="A", source="hand_crafted"),
            _make_record(eval_id="B", source="generated"),
        ]
        result = filter_evals(records, sources=["generated"])
        assert len(result) == 1
        assert result[0].eval_id == "B"

    def test_filter_combined(self):
        records = [
            _make_record(eval_id="A", setting="BC", difficulty="easy"),
            _make_record(eval_id="B", setting="BC", difficulty="hard"),
            _make_record(eval_id="C", setting="S1", difficulty="easy"),
        ]
        result = filter_evals(records, settings=["BC"], difficulties=["easy"])
        assert len(result) == 1
        assert result[0].eval_id == "A"

    def test_no_filters_returns_all(self):
        records = [_make_record(eval_id="A"), _make_record(eval_id="B")]
        assert len(filter_evals(records)) == 2
