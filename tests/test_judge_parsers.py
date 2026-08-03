"""Unit tests for the v2 judge response parsers.

Covers the three helpers introduced in phase 1/5 of the v2 hallucination
spec migration: ``_parse_text_response`` (two-line format → dict),
``_split_report_bullets`` (ITM report → numbered list), and
``_parse_csv_scores`` (length-padded CSV → list[float]).
"""

from __future__ import annotations

import pytest

from prism_eval.scoring.judge_llm import (
    _parse_csv_scores,
    _parse_text_response,
    _split_report_bullets,
)


# ─── _parse_text_response ────────────────────────────────────────────────────


def test_parse_text_response_happy_path() -> None:
    out = _parse_text_response("INSTRUCTIONS: 1.0,0.5,0.0\nHALLUCINATIONS: 0.0,0.5,1.0")
    assert out == {
        "instruction_scores": "1.0,0.5,0.0",
        "hallucination_scores": "0.0,0.5,1.0",
    }


def test_parse_text_response_case_insensitive() -> None:
    out = _parse_text_response("instructions: 1.0\nhallucinations: 0.0")
    assert out["instruction_scores"] == "1.0"
    assert out["hallucination_scores"] == "0.0"


def test_parse_text_response_tolerates_surrounding_text() -> None:
    """Judge sometimes prepends 'Sure! Here are the scores:' or similar."""
    raw = "Sure! Here are the scores:\nINSTRUCTIONS: 1.0,0.0\nHALLUCINATIONS: 0.5\nDone."
    out = _parse_text_response(raw)
    assert out["instruction_scores"] == "1.0,0.0"
    assert out["hallucination_scores"] == "0.5"


def test_parse_text_response_empty_hallucinations_line() -> None:
    """Spec says: empty ITM report ⇒ 'HALLUCINATIONS:' with nothing after."""
    out = _parse_text_response("INSTRUCTIONS: 1.0\nHALLUCINATIONS:")
    assert out["instruction_scores"] == "1.0"
    assert out["hallucination_scores"] == ""


def test_parse_text_response_only_one_line_present() -> None:
    """Tolerates partial responses — missing keys are simply absent."""
    out = _parse_text_response("INSTRUCTIONS: 1.0,1.0")
    assert "instruction_scores" in out
    assert "hallucination_scores" not in out


def test_parse_text_response_raises_when_neither_line_present() -> None:
    with pytest.raises(ValueError, match="Could not parse"):
        _parse_text_response("This response has no recognised fields.")


# ─── _split_report_bullets ───────────────────────────────────────────────────


def test_split_bullets_dash() -> None:
    assert _split_report_bullets("- one\n- two\n- three") == ["one", "two", "three"]


def test_split_bullets_numbered() -> None:
    assert _split_report_bullets("1. alpha\n2. beta\n3. gamma") == ["alpha", "beta", "gamma"]


def test_split_bullets_asterisk() -> None:
    assert _split_report_bullets("* foo\n* bar") == ["foo", "bar"]


def test_split_bullets_paragraph_fallback() -> None:
    """No bullets → fall back to paragraph split on blank-line."""
    text = "First paragraph.\n\nSecond paragraph."
    assert _split_report_bullets(text) == ["First paragraph.", "Second paragraph."]


def test_split_bullets_single_unstructured() -> None:
    assert _split_report_bullets("just one line of text") == ["just one line of text"]


def test_split_bullets_empty_report() -> None:
    assert _split_report_bullets("") == []
    assert _split_report_bullets("   \n  \n") == []


# ─── _parse_csv_scores ───────────────────────────────────────────────────────


def test_parse_csv_exact_length() -> None:
    assert _parse_csv_scores("1.0,0.5,0.0", 3) == [1.0, 0.5, 0.0]


def test_parse_csv_short_pads_right() -> None:
    """Judge returned fewer scores than expected → pad with default."""
    assert _parse_csv_scores("1.0,0.5", 4, default=0.0) == [1.0, 0.5, 0.0, 0.0]


def test_parse_csv_long_truncates() -> None:
    """Judge returned more scores than expected → truncate."""
    assert _parse_csv_scores("1.0,0.5,0.0,1.0,0.5", 3) == [1.0, 0.5, 0.0]


def test_parse_csv_handles_whitespace() -> None:
    assert _parse_csv_scores(" 1.0 , 0.5 , 0.0 ", 3) == [1.0, 0.5, 0.0]


def test_parse_csv_non_numeric_token_defaults() -> None:
    """A garbage token in the CSV becomes `default`, doesn't crash the row."""
    assert _parse_csv_scores("1.0,oops,0.0", 3) == [1.0, 0.0, 0.0]


def test_parse_csv_none_input() -> None:
    assert _parse_csv_scores(None, 3) == [0.0, 0.0, 0.0]


def test_parse_csv_empty_string() -> None:
    """Empty CSV (e.g. 'HALLUCINATIONS:' with nothing after) → all defaults."""
    assert _parse_csv_scores("", 2) == [0.0, 0.0]


def test_parse_csv_custom_default() -> None:
    assert _parse_csv_scores(None, 2, default=1.0) == [1.0, 1.0]
