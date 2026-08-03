"""Tests for the batched annotate pre-pass and the itm_annotate cache short-circuit.

Three invariants this file pins down:

1. **Cache short-circuit (step 2 of the parallelize-annotate work)**:
   When ``_PREDICT_CACHE`` already has an entry for (runner, eval_id),
   ``itm_annotate`` returns the cached output without acquiring run_lock
   or calling runner.run_eval. This is what makes the parallel
   trace-recording stage in step 3 work without GPU contention.

2. **Batched fast path (step 3)**: For a runner that exposes
   ``run_batch``, the pre-pass calls run_batch once per chunk, never
   calls run_eval, populates the cache, and emits one itm_annotate trace
   per record.

3. **Fallback path (step 3)**: For a runner that does NOT expose
   ``run_batch``, the pre-pass loops run_eval per record (acquiring
   run_lock each time, just like the v1 serial path) and still
   populates the cache so the trace-recording stage parallelises.

All tests use stub runners — no Qwen, no GPU. The real
batched-vs-sequential output equivalence test for PrismRunner
needs a GPU checkpoint and is left as a manual smoke (run_mc_offline_batched.py
exercises the same code path end-to-end).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest
import weave

from prism_eval import weave_eval
from prism_eval.schema import EvalRecord, EvalResult


# ─── Stub runners ────────────────────────────────────────────────────────────


def _make_record(eval_id: str) -> EvalRecord:
    """Minimal single-turn EvalRecord shaped like the v2 suites."""
    return EvalRecord(
        eval_id=eval_id,
        setting="AP",
        category="test",
        structural_tags=["single-turn:plain"],
        difficulty="easy",
        source="test",
        prompt=f"Write something about {eval_id}.",
        instruction_sources={"original": [f"talk about {eval_id}"]},
    )


def _stub_eval_result(eval_id: str, runner_name: str) -> EvalResult:
    return EvalResult(
        eval_id=eval_id,
        runner=runner_name,
        itm_report=f"- claim about {eval_id}",
        model_response=f"response for {eval_id}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_tokens=10,
        output_tokens=5,
        activation_tokens=8,
    )


class _StubRunnerWithBatch:
    """Records every call to run_batch / run_eval for assertions.

    Mirrors the public surface of PrismRunner (run_eval +
    run_batch) without loading a model.
    """

    runner_name = "stub_batched"

    def __init__(self) -> None:
        self.batch_calls: list[list[str]] = []
        self.eval_calls: list[str] = []
        self._lock = threading.Lock()

    def run_eval(self, record: EvalRecord) -> EvalResult:
        with self._lock:
            self.eval_calls.append(record.eval_id)
        return _stub_eval_result(record.eval_id, self.runner_name)

    def run_batch(self, records: list[EvalRecord]) -> list[EvalResult]:
        with self._lock:
            self.batch_calls.append([r.eval_id for r in records])
        return [_stub_eval_result(r.eval_id, self.runner_name) for r in records]


class _StubRunnerWithoutBatch:
    """Only implements run_eval — exercises the per-record fallback path."""

    runner_name = "stub_seq"

    def __init__(self) -> None:
        self.eval_calls: list[str] = []
        self._lock = threading.Lock()

    def run_eval(self, record: EvalRecord) -> EvalResult:
        with self._lock:
            self.eval_calls.append(record.eval_id)
        return _stub_eval_result(record.eval_id, self.runner_name)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def fresh_caches(monkeypatch):
    """Clear weave_eval's module-level caches and replace _get_runner with a stub.

    Yields ``(install_runner, runner_holder)`` — call ``install_runner(stub)``
    once you have your stub instance, and ``runner_holder["runner"]`` reads
    back whatever ``_get_runner`` returned for the configured triple.
    """
    weave_eval.clear_predict_cache()
    # Wipe the runner cache so a previous test's stub doesn't leak in.
    with weave_eval._runner_cache_lock:
        weave_eval._runner_cache.clear()
        weave_eval._runner_locks.clear()

    holder: dict = {"runner": None}

    def _install(stub_runner) -> None:
        holder["runner"] = stub_runner

        def _fake_get_runner(rt: str, ck: str, dev: str):
            # Seed the lock dict alongside the runner — production
            # _get_runner does this atomically, but our stub bypasses
            # that path so _get_runner_lock would otherwise KeyError.
            key = weave_eval._runner_cache_key(rt, ck, dev)
            if key not in weave_eval._runner_locks:
                with weave_eval._runner_cache_lock:
                    weave_eval._runner_locks.setdefault(key, threading.Lock())
            return holder["runner"]

        monkeypatch.setattr(weave_eval, "_get_runner", _fake_get_runner)

    yield _install, holder

    weave_eval.clear_predict_cache()
    with weave_eval._runner_cache_lock:
        weave_eval._runner_cache.clear()
        weave_eval._runner_locks.clear()


# Init weave once with a local mode so @weave.op-decorated functions don't
# try to reach an actual W&B project during these tests.
@pytest.fixture(scope="module", autouse=True)
def _local_weave():
    weave.init("disable", autopatch_settings={"openai": {"enabled": False}})
    yield


# ─── 1. itm_annotate short-circuit ───────────────────────────────────────────


def test_itm_annotate_returns_cached_without_calling_run_eval(fresh_caches):
    install, holder = fresh_caches
    stub = _StubRunnerWithBatch()
    install(stub)

    runner_type, ckpt, dev = "stub_batched", "ck", "cpu"
    rec = _make_record("T-001")

    # Pre-populate the cache so itm_annotate's early return fires.
    expected = {
        "itm_report": "pre-cached report",
        "model_response": "pre-cached response",
        "input_tokens": 1,
        "output_tokens": 2,
        "activation_tokens": 3,
    }
    with weave_eval._PREDICT_CACHE_LOCK:
        weave_eval._PREDICT_CACHE[
            weave_eval._predict_cache_key(runner_type, ckpt, dev, rec.eval_id)
        ] = expected

    with weave_eval.annotate_runner_context(runner_type, ckpt, dev):
        out = weave_eval.itm_annotate(**weave_eval.eval_record_to_row(rec))

    assert out == expected
    assert stub.eval_calls == [], "Cache hit must short-circuit run_eval"
    assert stub.batch_calls == [], "Cache hit must short-circuit run_batch"


def test_itm_annotate_falls_through_to_run_eval_on_cache_miss(fresh_caches):
    install, _ = fresh_caches
    stub = _StubRunnerWithBatch()
    install(stub)

    rec = _make_record("T-002")
    with weave_eval.annotate_runner_context("stub_batched", "ck", "cpu"):
        out = weave_eval.itm_annotate(**weave_eval.eval_record_to_row(rec))

    assert stub.eval_calls == ["T-002"]
    assert out["itm_report"] == "- claim about T-002"
    # And the output is now cached for the next call.
    cached = weave_eval._PREDICT_CACHE[
        weave_eval._predict_cache_key("stub_batched", "ck", "cpu", "T-002")
    ]
    assert cached == out


# ─── 2. batched_annotate_pre_pass with run_batch-capable runner ──────────────


def test_pre_pass_uses_run_batch_when_available(fresh_caches):
    install, _ = fresh_caches
    stub = _StubRunnerWithBatch()
    install(stub)

    records = [_make_record(f"B-{i:03d}") for i in range(5)]
    progress: list[tuple[int, int, int]] = []

    weave_eval.batched_annotate_pre_pass(
        records,
        runner_type="stub_batched",
        checkpoint="ck",
        device="cpu",
        batch_size=2,
        trace_workers=4,
        on_batch_complete=lambda b, done, total: progress.append((b, done, total)),
    )

    # 5 records at batch_size=2 → 3 batches (2, 2, 1)
    assert stub.batch_calls == [
        ["B-000", "B-001"],
        ["B-002", "B-003"],
        ["B-004"],
    ]
    assert stub.eval_calls == [], (
        "When run_batch is available, run_eval must not be called at all"
    )
    # Cache holds an entry for every record.
    for r in records:
        k = weave_eval._predict_cache_key("stub_batched", "ck", "cpu", r.eval_id)
        assert k in weave_eval._PREDICT_CACHE
        assert weave_eval._PREDICT_CACHE[k]["itm_report"] == f"- claim about {r.eval_id}"
    # Progress callback fired once per batch with monotonic n_done.
    assert len(progress) == 3
    assert [p[1] for p in progress] == [2, 4, 5]
    assert all(p[2] == 5 for p in progress)


def test_pre_pass_empty_records_is_noop(fresh_caches):
    install, _ = fresh_caches
    stub = _StubRunnerWithBatch()
    install(stub)

    weave_eval.batched_annotate_pre_pass(
        [],
        runner_type="stub_batched",
        checkpoint="ck",
        device="cpu",
    )

    assert stub.batch_calls == []
    assert stub.eval_calls == []
    assert not weave_eval._PREDICT_CACHE


# ─── 3. Fallback path: runner without run_batch ──────────────────────────────


def test_pre_pass_falls_back_to_run_eval_per_record(fresh_caches):
    install, _ = fresh_caches
    stub = _StubRunnerWithoutBatch()
    install(stub)

    records = [_make_record(f"S-{i:03d}") for i in range(3)]
    weave_eval.batched_annotate_pre_pass(
        records,
        runner_type="stub_seq",
        checkpoint="ck",
        device="cpu",
        batch_size=2,  # batches are still chunked; just loops run_eval within each
    )

    # 3 records, all processed via run_eval, in original order.
    assert stub.eval_calls == ["S-000", "S-001", "S-002"]
    # Cache is still populated identically — Stage 2's parallel trace fan-out
    # gets the same short-circuit benefit even on legacy runners.
    for r in records:
        k = weave_eval._predict_cache_key("stub_seq", "ck", "cpu", r.eval_id)
        assert k in weave_eval._PREDICT_CACHE


# ─── 4. Cache-key contract ───────────────────────────────────────────────────


def test_cache_key_shape_matches_itm_model_predict(fresh_caches):
    """Both itm_annotate (write) and ITMModel.predict (read) must use the same
    cache key shape. If this drifts, the Evaluation phase re-runs the model
    despite the pre-pass having populated the cache. Pin the shape explicitly.

    String keys (not tuples) so Weave's code-deps introspection doesn't
    warn on module globals — see weave_eval.py comment on _PREDICT_CACHE."""
    key = weave_eval._predict_cache_key("rt", "/path/to/ck.pt", "cuda:0", "E-001")
    assert key == "rt|/path/to/ck.pt|cuda:0|E-001"
    assert isinstance(key, str)
    # Contains all 4 identity components so the predict cache can't collide
    # across different (runner, checkpoint, device, eval) combinations.
    for component in ("rt", "/path/to/ck.pt", "cuda:0", "E-001"):
        assert component in key
