"""Token-level F1 scorer.

Computes token-level precision/recall/F1 between GT claim tokens and ITM report tokens.
Simple bag-of-words baseline.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from prism_eval.schema import ClaimScore, ScorerOutput


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenization."""
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(reference: str, hypothesis: str) -> tuple[float, float, float]:
    """Compute token-level precision, recall, F1.

    Returns (precision, recall, f1).
    """
    ref_tokens = Counter(_tokenize(reference))
    hyp_tokens = Counter(_tokenize(hypothesis))

    if not ref_tokens or not hyp_tokens:
        return (0.0, 0.0, 0.0)

    # Intersection count (min overlap per token)
    overlap = sum((ref_tokens & hyp_tokens).values())

    precision = overlap / sum(hyp_tokens.values()) if hyp_tokens else 0.0
    recall = overlap / sum(ref_tokens.values()) if ref_tokens else 0.0

    if precision + recall == 0:
        return (0.0, 0.0, 0.0)

    f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


def score_claim(claim: str, report: str) -> ClaimScore:
    """Score a single claim using token F1."""
    _, recall, f1 = token_f1(claim, report)

    # Use recall as primary signal (did the report cover the claim?)
    if recall >= 0.6:
        score = 1.0
    elif recall >= 0.3:
        score = 0.5
    else:
        score = 0.0

    evidence = f"recall={recall:.2f}, f1={f1:.2f}"
    return ClaimScore(claim=claim, score=score, evidence=evidence)


def score(
    instructions: Sequence[str],
    report: str,
) -> ScorerOutput:
    """Score all instruction claims using token F1."""
    return ScorerOutput(
        scorer="token_f1",
        instruction_scores=[score_claim(c, report) for c in instructions],
    )
