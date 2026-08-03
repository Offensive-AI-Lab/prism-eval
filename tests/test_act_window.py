"""Tests for the activation-window ablation helpers in the prism runner."""

import pytest

from prism_eval.runners.prism import (
    _act_window_start,
    _env_int,
    _resolve_act_window,
)


def test_end_window_is_last_n_act_tokens():
    # prompt=[0..99], response=[100..295] (196 tokens), window=128
    assert _act_window_start(100, 196, 128, "end") == 100 + 196 - 128


def test_start_window_begins_at_response_start():
    assert _act_window_start(100, 196, 128, "start") == 100


def test_middle_window_is_centered():
    # slack = 196 - 128 = 68 -> offset 34
    assert _act_window_start(100, 196, 128, "middle") == 134


def test_positions_coincide_when_response_shorter_than_window():
    # n_act = min(resp_len, k) = resp_len -> all positions identical
    for pos in ("start", "middle", "end"):
        assert _act_window_start(100, 96, 96, pos) == 100


def test_window_stays_within_response_bounds():
    for pos in ("start", "middle", "end"):
        start = _act_window_start(50, 300, 128, pos)
        assert 50 <= start
        assert start + 128 <= 50 + 300


def test_unknown_position_raises():
    with pytest.raises(ValueError):
        _act_window_start(100, 196, 128, "quarter")


def test_resolve_delegates_named_positions():
    # end: same as _act_window_start with n_act = min(resp_len, max_act)
    assert _resolve_act_window(100, 196, 128, "end") == (100 + 196 - 128, 128)
    assert _resolve_act_window(100, 196, 128, "start") == (100, 128)
    assert _resolve_act_window(100, 96, 128, "end") == (100, 96)


def test_chunk_windows_tile_the_response():
    # 300-token response, 128-token chunks: [0,128), [128,256), [256,300)
    assert _resolve_act_window(50, 300, 128, "chunk0") == (50, 128)
    assert _resolve_act_window(50, 300, 128, "chunk1") == (50 + 128, 128)
    assert _resolve_act_window(50, 300, 128, "chunk2") == (50 + 256, 44)


def test_chunk_beyond_response_is_empty():
    start, n_act = _resolve_act_window(50, 300, 128, "chunk3")
    assert n_act == 0
    # short response: chunk1 does not exist at all
    assert _resolve_act_window(50, 100, 128, "chunk1")[1] == 0
    # zero-length response: even chunk0 is empty
    assert _resolve_act_window(50, 0, 128, "chunk0")[1] == 0


def test_env_int_fallback_and_override(monkeypatch):
    monkeypatch.delenv("PRISM_EVAL_MAX_ACT_TOKENS", raising=False)
    assert _env_int("PRISM_EVAL_MAX_ACT_TOKENS", 128) == 128
    monkeypatch.setenv("PRISM_EVAL_MAX_ACT_TOKENS", "32")
    assert _env_int("PRISM_EVAL_MAX_ACT_TOKENS", 128) == 32
    monkeypatch.setenv("PRISM_EVAL_MAX_ACT_TOKENS", "")
    assert _env_int("PRISM_EVAL_MAX_ACT_TOKENS", 128) == 128
