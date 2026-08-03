"""Tests for the opt-in base-response cache in the prism runner."""

import json

import prism_eval.runners.prism as ed


def _reset_cache(monkeypatch, path):
    monkeypatch.setenv("PRISM_EVAL_BASE_RESPONSE_CACHE", str(path))
    monkeypatch.setattr(ed, "_BASE_RESPONSE_CACHE", None)


MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]


def test_key_is_stable_and_sensitive():
    k1 = ed._base_response_key("qwen", MESSAGES, 196)
    assert k1 == ed._base_response_key("qwen", MESSAGES, 196)
    assert k1 != ed._base_response_key("qwen", MESSAGES, 2048)
    assert k1 != ed._base_response_key("other-model", MESSAGES, 196)
    other = [dict(MESSAGES[0]), {"role": "user", "content": "What is 3+3?"}]
    assert k1 != ed._base_response_key("qwen", other, 196)


def test_cache_off_by_default(monkeypatch):
    monkeypatch.delenv("PRISM_EVAL_BASE_RESPONSE_CACHE", raising=False)
    assert ed._base_response_cache_path() is None


def test_append_then_load_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "cache.jsonl"
    _reset_cache(monkeypatch, path)
    key = ed._base_response_key("qwen", MESSAGES, 196)
    ed._append_base_responses([(key, "Four.")])

    monkeypatch.setattr(ed, "_BASE_RESPONSE_CACHE", None)  # force reload from disk
    assert ed._load_base_response_cache()[key] == "Four."


def test_load_skips_torn_lines(tmp_path, monkeypatch):
    path = tmp_path / "cache.jsonl"
    good = {"key": "k1", "response": "ok"}
    path.write_text(json.dumps(good) + "\n" + '{"key": "k2", "resp' + "\n")
    _reset_cache(monkeypatch, path)
    cache = ed._load_base_response_cache()
    assert cache == {"k1": "ok"}


def test_cached_batched_generate_mixes_hits_and_misses(tmp_path, monkeypatch):
    path = tmp_path / "cache.jsonl"
    _reset_cache(monkeypatch, path)

    m1 = [{"role": "user", "content": "a"}]
    m2 = [{"role": "user", "content": "b"}]
    k1 = ed._base_response_key("qwen", m1, 100)
    ed._append_base_responses([(k1, "resp-a")])
    monkeypatch.setattr(ed, "_BASE_RESPONSE_CACHE", None)

    runner = ed.PrismRunner()
    runner._cfg = {"model_id": "qwen"}
    generated = []

    def fake_generate(all_messages, max_new_tokens):
        generated.extend(all_messages)
        return [f"gen-{m[0]['content']}" for m in all_messages]

    monkeypatch.setattr(runner, "_batched_generate_base", fake_generate)
    out = runner._cached_batched_generate_base([m1, m2], 100)

    assert out == ["resp-a", "gen-b"]
    assert generated == [m2]  # only the miss was generated
    # the miss was persisted for the next process
    monkeypatch.setattr(ed, "_BASE_RESPONSE_CACHE", None)
    k2 = ed._base_response_key("qwen", m2, 100)
    assert ed._load_base_response_cache()[k2] == "gen-b"
