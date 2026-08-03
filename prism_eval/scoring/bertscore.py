"""BERTScore-based scorer.

Computes BERTScore between each GT claim and the ITM report text.
Requires the bert-score package.
"""

from __future__ import annotations

from collections.abc import Sequence

from prism_eval.schema import ClaimScore, ScorerOutput

# Thresholds (tunable)
DETECTED_THRESHOLD = 0.85
PARTIAL_THRESHOLD = 0.70


def _compute_bertscore(claims: list[str], report: str) -> list[float]:
    """Compute BERTScore F1 for each claim against the report."""
    from bert_score import score as bert_score

    # Each claim is compared against the full report
    references = [report] * len(claims)
    _, _, f1 = bert_score(claims, references, lang="en", verbose=False)
    return f1.tolist()


def score_claim(claim: str, f1: float) -> ClaimScore:
    """Convert a BERTScore F1 value to a ClaimScore."""
    if f1 >= DETECTED_THRESHOLD:
        sc = 1.0
    elif f1 >= PARTIAL_THRESHOLD:
        sc = 0.5
    else:
        sc = 0.0
    return ClaimScore(claim=claim, score=sc, evidence=f"bertscore_f1={f1:.4f}")


def score(
    instructions: Sequence[str],
    report: str,
) -> ScorerOutput:
    """Score all instruction claims using BERTScore."""
    if not instructions:
        return ScorerOutput(scorer="bertscore")

    f1_scores = _compute_bertscore(list(instructions), report)

    return ScorerOutput(
        scorer="bertscore",
        instruction_scores=[
            score_claim(c, f1) for c, f1 in zip(instructions, f1_scores)
        ],
    )
