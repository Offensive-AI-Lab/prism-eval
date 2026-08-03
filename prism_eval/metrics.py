"""Metrics computation and decay analysis.

v2 per-eval metrics: ``instruction_recall``, ``mean_hallucination_score``
(replaces v1 ``hallucination_rate``, which assumed an int count).
Decay analysis: group instructions by distance from end, plot recall vs distance.
Scorer comparison: agreement across methods.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from prism_eval.schema import EvalRecord, ScoredResult, ScorerOutput


def instruction_recall(scorer_output: ScorerOutput) -> float:
    """Mean score across instruction claims."""
    scores = scorer_output.instruction_scores
    if not scores:
        # Fallback to deprecated v1 fields for old JSONL results
        v1_scores = scorer_output.constraint_scores + scorer_output.goal_scores
        if not v1_scores:
            return 0.0
        return sum(cs.score for cs in v1_scores) / len(v1_scores)
    return sum(cs.score for cs in scores) / len(scores)


def mean_hallucination_score(scorer_output: ScorerOutput) -> float:
    """Mean per-ITM-bullet hallucination score in [0, 1]. Only meaningful for judge_llm.

    v1 ``hallucination_rate`` used ``count / (gt_count + count)``, but with v2
    per-bullet 0/0.5/1 scores the analogous quantity is just the mean. Empty
    score list ⇒ 0.0 (no hallucination signal available).
    """
    if not scorer_output.hallucination_scores:
        return 0.0
    return sum(scorer_output.hallucination_scores) / len(scorer_output.hallucination_scores)


def per_eval_metrics(scored: ScoredResult) -> dict[str, dict[str, float]]:
    """Compute per-scorer metrics for a single eval.

    Returns {scorer_name: {metric_name: value}}.
    """
    result: dict[str, dict[str, float]] = {}
    for name, so in scored.scores.items():
        result[name] = {
            "instruction_recall": round(instruction_recall(so), 4),
            "mean_hallucination_score": round(mean_hallucination_score(so), 4),
        }
    return result


def aggregate_metrics(
    scored_results: Sequence[ScoredResult],
    scorer_name: str,
) -> dict[str, float]:
    """Aggregate metrics across all evals for a given scorer.

    Returns mean instruction_recall, mean_hallucination_score, n.
    """
    i_recalls: list[float] = []
    halluc_means: list[float] = []

    for sr in scored_results:
        so = sr.scores.get(scorer_name)
        if so is None:
            continue
        i_recalls.append(instruction_recall(so))
        halluc_means.append(mean_hallucination_score(so))

    n = len(i_recalls)
    if n == 0:
        return {"instruction_recall": 0.0, "mean_hallucination_score": 0.0, "n": 0}

    return {
        "instruction_recall": round(sum(i_recalls) / n, 4),
        "mean_hallucination_score": round(sum(halluc_means) / n, 4),
        "n": n,
    }


def decay_analysis(
    records: dict[str, EvalRecord],
    scored_results: Sequence[ScoredResult],
    scorer_name: str,
) -> dict[int, float]:
    """Compute instruction recall as a function of distance from conversation end.

    For generated evals with instruction_positions, we know which turn introduced
    each instruction. The "distance" is (n_turns - instruction_turn_index).

    Returns {distance: mean_recall_at_that_distance}.
    """
    distance_scores: defaultdict[int, list[float]] = defaultdict(list)

    for sr in scored_results:
        record = records.get(sr.eval_id)
        if record is None:
            continue
        if record.instruction_positions is None or record.generation_params is None:
            continue

        so = sr.scores.get(scorer_name)
        if so is None:
            continue

        n_turns = record.generation_params.n_turns
        positions = record.instruction_positions

        # Use instruction_scores (v2), fall back to constraint_scores (v1)
        claim_scores = so.instruction_scores or so.constraint_scores

        for i, pos in enumerate(positions):
            if i < len(claim_scores):
                distance = n_turns - pos
                distance_scores[distance].append(claim_scores[i].score)

    return {
        dist: round(sum(scores) / len(scores), 4)
        for dist, scores in sorted(distance_scores.items())
    }


def scorer_comparison(
    scored_results: Sequence[ScoredResult],
) -> dict[str, dict[str, float]]:
    """Compare scorers head-to-head on the same evals.

    Returns {scorer_name: {mean_instruction_recall, mean_hallucination_score, n_evals}}.
    """
    # Collect all scorer names
    scorer_names: set[str] = set()
    for sr in scored_results:
        scorer_names.update(sr.scores.keys())

    comparison: dict[str, dict[str, float]] = {}
    for name in sorted(scorer_names):
        agg = aggregate_metrics(scored_results, name)
        comparison[name] = {
            "mean_instruction_recall": agg["instruction_recall"],
            "mean_hallucination_score": agg["mean_hallucination_score"],
            "n_evals": agg["n"],
        }

    return comparison
