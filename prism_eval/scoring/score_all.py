"""Orchestrator that runs all scoring methods on an EvalResult."""

from __future__ import annotations

from collections.abc import Sequence

from prism_eval.schema import EvalRecord, EvalResult, ProbePoint, ScoredResult, ScorerOutput
from prism_eval.scoring import exact_match, token_f1


def resolve_instructions(record: EvalRecord, probe: ProbePoint | None = None) -> list[str]:
    """Resolve the instruction list to score against.

    For v2 records with probe_points, resolves source keys from the probe's
    expected_instructions against instruction_sources.

    For v1 records (or v2 without probe_points), returns instruction_sources["original"]
    (or falls back to constraints for un-migrated records).
    """
    if record.instruction_sources is not None:
        if probe is not None:
            instructions: list[str] = []
            for key in probe.expected_instructions:
                if key == "TBD_post_hoc":
                    continue
                instructions.extend(record.instruction_sources.get(key, []))
            return instructions
        # No probe specified — return all original instructions
        return list(record.instruction_sources.get("original", []))

    # v1 fallback: concatenate constraints + goals
    result: list[str] = []
    if record.constraints:
        result.extend(record.constraints)
    if record.goals:
        result.extend(record.goals)
    return result


def score_eval(
    record: EvalRecord,
    result: EvalResult,
    scorers: Sequence[str] | None = None,
    use_bertscore: bool = False,
    use_judge_llm: bool = False,
) -> ScoredResult:
    """Run specified scorers on a single eval result.

    Args:
        record: The eval record with ground-truth claims.
        result: The ITM runner output.
        scorers: Explicit list of scorer names, or None for auto-selection.
        use_bertscore: Enable BERTScore (requires model download on first run).
        use_judge_llm: Enable LLM judge (requires PRISM_EVAL_MODEL + API credentials).
    """
    active_scorers = scorers or _default_scorers(use_bertscore, use_judge_llm)

    # For single-probe evals (majority of Round 1), resolve instructions once.
    # Multi-probe scoring will be handled per-probe when runners support it.
    probe = record.probe_points[0] if record.probe_points else None

    # Skip entirely if probe is TBD_post_hoc
    if probe and all(k == "TBD_post_hoc" for k in probe.expected_instructions):
        return ScoredResult(eval_id=record.eval_id, runner=result.runner, scores={})

    instructions = resolve_instructions(record, probe)
    scores: dict[str, ScorerOutput] = {}

    if "exact_match" in active_scorers:
        scores["exact_match"] = exact_match.score(instructions, result.itm_report)

    if "token_f1" in active_scorers:
        scores["token_f1"] = token_f1.score(instructions, result.itm_report)

    if "bertscore" in active_scorers:
        from prism_eval.scoring import bertscore
        scores["bertscore"] = bertscore.score(instructions, result.itm_report)

    if "judge_llm" in active_scorers:
        from prism_eval.scoring import judge_llm
        scores["judge_llm"] = judge_llm.score(instructions, result.itm_report)

    return ScoredResult(
        eval_id=record.eval_id,
        runner=result.runner,
        scores=scores,
    )


def _default_scorers(use_bertscore: bool, use_judge_llm: bool) -> list[str]:
    """Determine which scorers to run by default."""
    s = ["exact_match", "token_f1"]
    if use_bertscore:
        s.append("bertscore")
    if use_judge_llm:
        s.append("judge_llm")
    return s
