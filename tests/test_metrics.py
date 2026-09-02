"""Tests for prism_eval.metrics (current hallucination spec)."""

from prism_eval.metrics import (
    aggregate_metrics,
    decay_analysis,
    instruction_recall,
    mean_hallucination_score,
    per_eval_metrics,
    scorer_comparison,
)
from prism_eval.schema import (
    ClaimScore,
    EvalRecord,
    GenerationParams,
    ScoredResult,
    ScorerOutput,
)


def _so(scores: list[float], halluc_scores: list[float] | None = None) -> ScorerOutput:
    """Build a current-spec ScorerOutput with instruction_scores + hallucination_scores."""
    return ScorerOutput(
        scorer="test",
        instruction_scores=[ClaimScore(claim=f"i{i}", score=s) for i, s in enumerate(scores)],
        hallucination_scores=halluc_scores or [],
    )


def _so_legacy(constraint_scores: list[float], goal_scores: list[float], hall: int = 0) -> ScorerOutput:
    """Build a legacy-style ScorerOutput for backward compat tests."""
    return ScorerOutput(
        scorer="test",
        constraint_scores=[ClaimScore(claim=f"c{i}", score=s) for i, s in enumerate(constraint_scores)],
        goal_scores=[ClaimScore(claim=f"g{i}", score=s) for i, s in enumerate(goal_scores)],
        hallucination_count=hall,
    )


class TestBasicMetrics:
    def test_instruction_recall(self):
        so = _so([1.0, 0.5, 0.0])
        assert instruction_recall(so) == 0.5

    def test_instruction_recall_perfect(self):
        so = _so([1.0, 1.0])
        assert instruction_recall(so) == 1.0

    def test_mean_hallucination_score(self):
        # 4 ITM bullets, mix of grounded / partial / hallucinated → mean = 0.5
        so = _so([1.0, 1.0], halluc_scores=[0.0, 0.5, 0.5, 1.0])
        assert mean_hallucination_score(so) == 0.5

    def test_mean_hallucination_score_empty(self):
        """Empty report ⇒ no halluc signal ⇒ 0.0 (no penalty)."""
        so = _so([1.0, 1.0])
        assert mean_hallucination_score(so) == 0.0

    def test_empty_scores(self):
        so = _so([])
        assert instruction_recall(so) == 0.0

    def test_v1_fallback(self):
        """instruction_recall falls back to legacy fields when instruction_scores is empty."""
        so = _so_legacy([1.0, 0.0], [0.5])
        assert instruction_recall(so) == 0.5  # mean of [1.0, 0.0, 0.5]


class TestPerEvalMetrics:
    def test_returns_all_scorers(self):
        sr = ScoredResult(
            eval_id="T-001",
            runner="test",
            scores={
                "exact_match": _so([1.0, 0.0]),
                "token_f1": _so([0.5, 0.5]),
            },
        )
        m = per_eval_metrics(sr)
        assert "exact_match" in m
        assert "token_f1" in m
        assert m["exact_match"]["instruction_recall"] == 0.5
        assert m["token_f1"]["instruction_recall"] == 0.5


class TestAggregateMetrics:
    def test_aggregation(self):
        results = [
            ScoredResult(eval_id="A", runner="t", scores={"s": _so([1.0])}),
            ScoredResult(eval_id="B", runner="t", scores={"s": _so([0.0])}),
        ]
        agg = aggregate_metrics(results, "s")
        assert agg["instruction_recall"] == 0.5
        assert agg["n"] == 2

    def test_missing_scorer(self):
        results = [
            ScoredResult(eval_id="A", runner="t", scores={"other": _so([1.0])}),
        ]
        agg = aggregate_metrics(results, "missing")
        assert agg["n"] == 0


class TestDecayAnalysis:
    def test_basic_decay(self):
        gen_params = GenerationParams(
            n_turns=8, n_instructions=2, k_distribution="fixed", k_param=1, flavor="benign"
        )
        record = EvalRecord(
            eval_id="G-001",
            setting="BC",
            category="test",
            structural_tags=["multi-turn:stacking"],
            difficulty="medium",
            source="generated",
            prompt_turns=["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"],
            instruction_sources={"original": ["instr at turn 0", "instr at turn 4"]},
            generation_params=gen_params,
            instruction_positions=[0, 4],
        )
        scored = ScoredResult(
            eval_id="G-001",
            runner="test",
            scores={
                "test_scorer": ScorerOutput(
                    scorer="test_scorer",
                    instruction_scores=[
                        ClaimScore(claim="instr at turn 0", score=0.5),  # distance 8
                        ClaimScore(claim="instr at turn 4", score=1.0),  # distance 4
                    ],
                ),
            },
        )

        decay = decay_analysis({"G-001": record}, [scored], "test_scorer")
        assert 8 in decay  # distance from turn 0
        assert 4 in decay  # distance from turn 4
        assert decay[8] == 0.5
        assert decay[4] == 1.0


class TestScorerComparison:
    def test_comparison(self):
        results = [
            ScoredResult(
                eval_id="A",
                runner="t",
                scores={
                    "exact_match": _so([1.0, 0.5]),
                    "token_f1": _so([0.5, 1.0]),
                },
            ),
        ]
        comp = scorer_comparison(results)
        assert "exact_match" in comp
        assert "token_f1" in comp
        assert comp["exact_match"]["mean_instruction_recall"] == 0.75
        assert comp["token_f1"]["mean_instruction_recall"] == 0.75
