"""Tests for the extraction-context ablation helpers (PRISM_EVAL_ACT_CONTEXT)."""

import json

import pytest

from prism_eval.runners import prism as ed


def test_default_mode_is_full(monkeypatch):
    monkeypatch.delenv("PRISM_EVAL_ACT_CONTEXT", raising=False)
    assert ed._act_context_mode() == "full"
    monkeypatch.setenv("PRISM_EVAL_ACT_CONTEXT", "")
    assert ed._act_context_mode() == "full"


def test_known_modes_parse(monkeypatch):
    for mode in ed._ACT_CONTEXT_MODES:
        monkeypatch.setenv("PRISM_EVAL_ACT_CONTEXT", mode)
        assert ed._act_context_mode() == mode


def test_unknown_mode_raises(monkeypatch):
    monkeypatch.setenv("PRISM_EVAL_ACT_CONTEXT", "hidden")
    with pytest.raises(ValueError):
        ed._act_context_mode()


def test_mask_span_only_for_masked_modes():
    for mode in ("full", "evicted", "swapped"):
        assert ed._act_context_mask_span(mode, 100, 20) is None
    assert ed._act_context_mask_span("masked_prompt", 100, 20) == (0, 100)
    assert ed._act_context_mask_span("masked_user", 100, 20) == (20, 100)


def test_masked_user_span_clamped_when_system_exceeds_prompt():
    # After mid-truncation prompt_len can shrink below the untruncated sys_len
    assert ed._act_context_mask_span("masked_user", 10, 20) == (10, 10)


def test_build_extraction_messages_single_turn():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "secret instruction"},
        {"role": "assistant", "content": "resp"},
    ]
    assert ed._build_extraction_messages(msgs, "...") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "resp"},
    ]


def test_build_extraction_messages_collapses_history():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    out = ed._build_extraction_messages(msgs, "donor prompt")
    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    assert out[1]["content"] == "donor prompt"
    assert out[2]["content"] == "a2"


def test_build_extraction_messages_without_system():
    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    out = ed._build_extraction_messages(msgs, "x")
    assert [m["role"] for m in out] == ["user", "assistant"]


def test_build_extraction_messages_requires_assistant_last():
    with pytest.raises(ValueError):
        ed._build_extraction_messages([{"role": "user", "content": "u"}], "x")
    with pytest.raises(ValueError):
        ed._build_extraction_messages([], "x")


def test_longest_common_prefix_len():
    assert ed._longest_common_prefix_len([1, 2, 3], [1, 2, 4]) == 2
    assert ed._longest_common_prefix_len([1, 2], [1, 2, 3]) == 2
    assert ed._longest_common_prefix_len([], [1]) == 0
    assert ed._longest_common_prefix_len([5], [6]) == 0


def test_swap_prompt_requires_file(monkeypatch):
    monkeypatch.setattr(ed, "_SWAP_PROMPTS", None)
    monkeypatch.delenv("PRISM_EVAL_ACT_SWAP_FILE", raising=False)
    with pytest.raises(RuntimeError):
        ed._swap_prompt_for("AP_001")


def test_swap_prompt_lookup_and_missing_id_fails_loud(tmp_path, monkeypatch):
    pairs_file = tmp_path / "pairs.json"
    pairs_file.write_text(json.dumps({
        "pairs": {"AP_001": {"donor_eval_id": "AP_002", "donor_prompt": "p2"}},
    }))
    monkeypatch.setattr(ed, "_SWAP_PROMPTS", None)
    monkeypatch.setenv("PRISM_EVAL_ACT_SWAP_FILE", str(pairs_file))
    assert ed._swap_prompt_for("AP_001") == "p2"
    with pytest.raises(KeyError):
        ed._swap_prompt_for("AP_999")
