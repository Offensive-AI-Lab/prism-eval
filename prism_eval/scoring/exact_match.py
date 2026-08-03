"""Exact-match / keyword scorer.

Checks if key terms from each GT claim appear in the ITM report.
Cheapest baseline scorer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from prism_eval.schema import ClaimScore, ScorerOutput

# Words to ignore when extracting key terms
STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could must need to of in for "
    "on at by with from as into through during before after above below "
    "between and or but not no nor so yet both either neither each every "
    "all any few more most other some such that this these those it its "
    "he she they them their his her we our you your i me my".split()
)


def _extract_key_terms(text: str) -> set[str]:
    """Extract meaningful lowercased terms from text, filtering stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def score_claim(claim: str, report: str) -> ClaimScore:
    """Score a single claim against the report using keyword overlap."""
    claim_terms = _extract_key_terms(claim)
    if not claim_terms:
        return ClaimScore(claim=claim, score=0.0, evidence="no key terms extracted from claim")

    report_lower = report.lower()
    matched = {t for t in claim_terms if t in report_lower}
    ratio = len(matched) / len(claim_terms)

    if ratio >= 0.7:
        score = 1.0
    elif ratio >= 0.4:
        score = 0.5
    else:
        score = 0.0

    evidence = f"matched {len(matched)}/{len(claim_terms)} terms: {sorted(matched)}" if matched else None
    return ClaimScore(claim=claim, score=score, evidence=evidence)


def score(
    instructions: Sequence[str],
    report: str,
) -> ScorerOutput:
    """Score all instruction claims using exact match."""
    return ScorerOutput(
        scorer="exact_match",
        instruction_scores=[score_claim(c, report) for c in instructions],
    )
