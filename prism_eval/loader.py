"""Load and filter eval suites."""

from __future__ import annotations

import json
from pathlib import Path

from prism_eval.schema import EvalRecord, EvalSuite


def load_suite(path: Path) -> EvalSuite:
    """Load an eval suite from a JSON file.

    Accepts both v1 and v2 JSON files. v1 records are auto-upgraded to v2
    in memory (the file on disk is not modified).
    """
    raw = json.loads(path.read_text())
    version = raw.get("schema_version", "1.0")

    if version.startswith("1"):
        raise ValueError(
            f"{path} is a schema v1 suite, which this release no longer supports. "
            f"The shipped suites are v2 — see DATA_CARD.md."
        )

    return EvalSuite.model_validate(raw)


def filter_evals(
    evals: list[EvalRecord],
    settings: list[str] | None = None,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    difficulties: list[str] | None = None,
) -> list[EvalRecord]:
    """Filter eval records by setting, structural tags, source, or difficulty.

    All filters are AND-combined. Tag filtering uses intersection (any match).
    """
    result = evals

    if settings:
        settings_set = set(settings)
        result = [r for r in result if r.setting in settings_set]

    if tags:
        tags_set = set(tags)
        result = [r for r in result if tags_set & set(r.structural_tags)]

    if sources:
        sources_set = set(sources)
        result = [r for r in result if r.source in sources_set]

    if difficulties:
        diff_set = set(difficulties)
        result = [r for r in result if r.difficulty in diff_set]

    return result


def load_results(path: Path) -> list[dict]:
    """Load JSONL results file."""
    results = []
    for line in path.read_text().strip().splitlines():
        if line.strip():
            results.append(json.loads(line))
    return results
