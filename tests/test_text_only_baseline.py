"""Tests for the text_only_baseline runner plumbing.

Covers what we can exercise without GPU + OpenAI:
  - Config validation (text_only_baseline requires baseline fields and forbids
    a checkpoint; other runners still require a checkpoint).
  - identity() round-trips through parse_identity().
  - Tail extraction respects tail_tokens and short-response edge cases.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prism_eval.config import RunnerConfig
from prism_eval.runners.text_only_baseline import (
    TextOnlyBaselineRunner,
    parse_identity,
)


# ── config validation ──────────────────────────────────────────────────────


def test_text_only_baseline_requires_response_model_id():
    with pytest.raises(ValidationError, match="response_model_id"):
        RunnerConfig(type="text_only_baseline", baseline_llm="gpt-5")


def test_text_only_baseline_requires_baseline_llm():
    with pytest.raises(ValidationError, match="baseline_llm"):
        RunnerConfig(
            type="text_only_baseline",
            response_model_id="Qwen/Qwen3-8B",
        )


def test_text_only_baseline_forbids_checkpoint():
    with pytest.raises(ValidationError, match="no checkpoint"):
        RunnerConfig(
            type="text_only_baseline",
            checkpoint="/some/path.pt",
            response_model_id="Qwen/Qwen3-8B",
            baseline_llm="gpt-5",
        )


def test_other_runners_still_require_checkpoint():
    with pytest.raises(ValidationError, match="checkpoint is required"):
        RunnerConfig(type="prism")


def test_text_only_baseline_defaults():
    cfg = RunnerConfig(
        type="text_only_baseline",
        response_model_id="Qwen/Qwen3-8B",
        baseline_llm="gpt-5",
    )
    assert cfg.tail_tokens == 128
    assert cfg.baseline_base_url is None
    assert cfg.checkpoint is None


# ── identity round-trip ───────────────────────────────────────────────────


def test_identity_round_trip_default_base_url():
    cfg = RunnerConfig(
        type="text_only_baseline",
        response_model_id="Qwen/Qwen3-8B",
        baseline_llm="gpt-5",
        tail_tokens=64,
    )
    parsed = parse_identity(cfg.identity())
    assert parsed == {
        "response_model_id": "Qwen/Qwen3-8B",
        "baseline_llm": "gpt-5",
        "tail_tokens": 64,
        "baseline_base_url": None,
    }


def test_identity_round_trip_custom_base_url():
    cfg = RunnerConfig(
        type="text_only_baseline",
        response_model_id="Qwen/Qwen3-8B",
        baseline_llm="gpt-5",
        tail_tokens=128,
        baseline_base_url="https://api.example.com/v1",
    )
    parsed = parse_identity(cfg.identity())
    assert parsed["baseline_base_url"] == "https://api.example.com/v1"


def test_identity_for_other_runner_is_checkpoint_path():
    cfg = RunnerConfig(
        type="prism",
        checkpoint="/tmp/best.pt",
    )
    assert cfg.identity() == "/tmp/best.pt"


def test_parse_identity_rejects_malformed():
    with pytest.raises(ValueError, match="Not a text_only_baseline identity"):
        parse_identity("/some/checkpoint.pt")
    with pytest.raises(ValueError, match="Malformed"):
        parse_identity("text_only_baseline://only|two")


# ── tail extraction ────────────────────────────────────────────────────────


class _FakeTokenizer:
    """Reverse map ints to strings — enough to test _tail_text slicing.

    Each token decodes to ``f"<{id}>"`` so we can assert which ids made it
    into the tail without booting a real tokenizer.
    """

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"<{i}>" for i in ids)


def _runner_with_fake_tokenizer(tail_tokens: int) -> TextOnlyBaselineRunner:
    r = TextOnlyBaselineRunner()
    r._tokenizer = _FakeTokenizer()
    r._tail_tokens = tail_tokens
    return r


def test_tail_text_takes_last_k_tokens():
    r = _runner_with_fake_tokenizer(tail_tokens=3)
    text, n = r._tail_text([10, 11, 12, 13, 14])
    assert n == 3
    assert text == "<12> <13> <14>"


def test_tail_text_short_response_returns_whole():
    r = _runner_with_fake_tokenizer(tail_tokens=10)
    text, n = r._tail_text([7, 8])
    assert n == 2
    assert text == "<7> <8>"


def test_tail_text_empty_response():
    r = _runner_with_fake_tokenizer(tail_tokens=5)
    text, n = r._tail_text([])
    assert text == ""
    assert n == 0
