"""Tests for the cache-then-consume policy in weave_eval.

`itm_annotate` (the annotation root op) populates `_PREDICT_CACHE`;
`ITMModel.predict` reads from it so model inference runs exactly once per
row across the annotation pre-pass + weave.Evaluation passes.

These tests exercise the cache mechanics without a live Weave client by
monkeypatching the runner registry with a call-counting fake runner.
"""

from __future__ import annotations

import threading

import pytest

from prism_eval import weave_eval
from prism_eval.schema import EvalRecord, EvalResult


class _FakeRunner:
    """Counts `run_eval` invocations; returns a deterministic EvalResult."""

    def __init__(self) -> None:
        self.calls = 0

    def setup(self, checkpoint: str, device: str = "cuda") -> None:
        return None

    def run_eval(self, record: EvalRecord) -> EvalResult:
        self.calls += 1
        return EvalResult(
            eval_id=record.eval_id,
            runner="fake",
            model_response=f"resp::{record.eval_id}",
            itm_report=f"report::{record.eval_id}",
            timestamp="2026-04-20T00:00:00Z",
            input_tokens=1,
            output_tokens=2,
            activation_tokens=3,
        )


@pytest.fixture(autouse=True)
def _disable_weave_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `@weave.op`-decorated calls pass through without a client.

    Without this, calling `itm_annotate` outside a `weave.init(...)` context
    still attempts to log — disabling via `WEAVE_DISABLED` keeps ops as
    regular callables for the test.
    """
    monkeypatch.setenv("WEAVE_DISABLED", "true")


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> _FakeRunner:
    runner = _FakeRunner()
    monkeypatch.setattr(weave_eval, "_runner_cache", {"fake|ckpt|cpu": runner})
    monkeypatch.setattr(
        weave_eval,
        "_runner_locks",
        {"fake|ckpt|cpu": threading.Lock()},
    )
    weave_eval.clear_predict_cache()
    yield runner
    weave_eval.clear_predict_cache()


def _sample_record() -> EvalRecord:
    return EvalRecord(
        eval_id="unit-test-1",
        setting="BN",
        category="baseline",
        structural_tags=["single-turn"],
        difficulty="easy",
        source="hand_crafted",
        prompt="Please summarize the following text.",
        instruction_sources={"original": ["summarize briefly"]},
    )


def test_itm_annotate_populates_cache_and_predict_reads_it(
    fake_runner: _FakeRunner,
) -> None:
    record = _sample_record()
    row = weave_eval.eval_record_to_row(record)

    # Pre-pass: one run of the runner, cache now populated.
    with weave_eval.annotate_runner_context("fake", "ckpt", "cpu"):
        annotate_out = weave_eval.itm_annotate(**row)

    assert fake_runner.calls == 1
    assert annotate_out["itm_report"] == "report::unit-test-1"
    cache_key = weave_eval._predict_cache_key("fake", "ckpt", "cpu", record.eval_id)
    assert weave_eval._PREDICT_CACHE[cache_key]["itm_report"] == "report::unit-test-1"

    # Evaluation-side predict: should hit the cache, NOT invoke the runner again.
    model = weave_eval.ITMModel(
        name="fake", runner_type="fake", checkpoint="ckpt", device="cpu"
    )
    predict_out = model.predict(**row)

    assert fake_runner.calls == 1, "ITMModel.predict must not re-run the runner on cache hit"
    assert predict_out == annotate_out


def test_predict_falls_back_to_runner_when_cache_empty(
    fake_runner: _FakeRunner,
) -> None:
    record = _sample_record()
    row = weave_eval.eval_record_to_row(record)

    model = weave_eval.ITMModel(
        name="fake", runner_type="fake", checkpoint="ckpt", device="cpu"
    )
    out = model.predict(**row)

    assert fake_runner.calls == 1
    assert out["itm_report"] == "report::unit-test-1"


def test_clear_predict_cache_forces_rerun(fake_runner: _FakeRunner) -> None:
    record = _sample_record()
    row = weave_eval.eval_record_to_row(record)

    with weave_eval.annotate_runner_context("fake", "ckpt", "cpu"):
        weave_eval.itm_annotate(**row)
    assert fake_runner.calls == 1

    weave_eval.clear_predict_cache()

    model = weave_eval.ITMModel(
        name="fake", runner_type="fake", checkpoint="ckpt", device="cpu"
    )
    model.predict(**row)
    assert fake_runner.calls == 2, "After clear, predict should re-invoke the runner"


def test_itm_annotate_without_runner_context_raises() -> None:
    record = _sample_record()
    row = weave_eval.eval_record_to_row(record)
    with pytest.raises(RuntimeError, match="annotate_runner_context"):
        weave_eval.itm_annotate(**row)


# ─────────────────────────────────────────────────────────────────────────────
# JudgeLLMScorer.summarize() — overall + per-setting aggregation
# ─────────────────────────────────────────────────────────────────────────────


def _row(
    setting: str,
    recall: float,
    mean_halluc: float,
    reward: float,
    length_penalty: float = 0.0,
) -> dict:
    return {
        "setting": setting,
        "coverage": recall,
        "instruction_score": recall,
        "mean_hallucination_score": mean_halluc,
        "reward": reward,
        "length_penalty": length_penalty,
    }


def test_judge_llm_summarize_overall_means() -> None:
    """Flat-name schema (advdet_v1+): top-level scalars, no .mean wrapper.

    `hallucination_rate` is the leaderboard-facing key for the value that
    score() writes as `mean_hallucination_score` per row; `coverage` is the
    paper's Coverage Rate."""
    scorer = weave_eval.JudgeLLMScorer(judge_model="stub")
    rows = [
        _row("AP", recall=1.0, mean_halluc=0.0, reward=0.6),
        _row("BN", recall=0.0, mean_halluc=1.0, reward=-0.4),
    ]
    summary = scorer.summarize(rows)

    assert summary["coverage"] == pytest.approx(0.5)
    assert summary["hallucination_rate"] == pytest.approx(0.5)
    assert summary["reward"] == pytest.approx(0.1)
    assert summary["length_penalty"] == pytest.approx(0.0)


def test_judge_llm_summarize_by_setting() -> None:
    """Per-setting block lives at top-level under the setting key
    (path: AP.recall, no longer by_setting.AP.recall)."""
    scorer = weave_eval.JudgeLLMScorer(judge_model="stub")
    rows = [
        _row("AP", recall=0.8, mean_halluc=0.1, reward=0.44),
        _row("AP", recall=0.6, mean_halluc=0.3, reward=0.24),
        _row("BN", recall=1.0, mean_halluc=0.0, reward=0.6),
    ]
    summary = scorer.summarize(rows)

    assert summary["AP"]["n"] == 2
    assert summary["AP"]["coverage"] == pytest.approx(0.7)
    assert summary["AP"]["hallucination_rate"] == pytest.approx(0.2)
    assert summary["AP"]["reward"] == pytest.approx(0.34)
    assert summary["BN"]["n"] == 1
    assert summary["BN"]["reward"] == pytest.approx(0.6)


def test_judge_llm_summarize_empty_rows_does_not_crash() -> None:
    scorer = weave_eval.JudgeLLMScorer(judge_model="stub")
    summary = scorer.summarize([])
    assert summary["reward"] == 0.0
    assert summary["coverage"] == 0.0
    assert summary["hallucination_rate"] == 0.0
    assert summary["length_penalty"] == 0.0
    # No per-setting blocks emitted when no rows; by_source is present but empty.
    assert set(summary.keys()) == {
        "reward", "coverage", "hallucination_rate", "length_penalty", "by_source"
    }
    assert summary["by_source"] == {}


def test_judge_llm_summarize_by_source() -> None:
    """Per-source block lives under `by_source.<source>` (parallel to the
    top-level per-setting blocks), so source names can't collide with
    setting names."""
    scorer = weave_eval.JudgeLLMScorer(judge_model="stub")

    def _src_row(source, recall, mean_halluc, reward):
        r = _row("AP", recall=recall, mean_halluc=mean_halluc, reward=reward)
        r["source"] = source
        return r

    rows = [
        _src_row("bipia", recall=0.8, mean_halluc=0.1, reward=0.44),
        _src_row("bipia", recall=0.6, mean_halluc=0.3, reward=0.24),
        _src_row("llmail", recall=1.0, mean_halluc=0.0, reward=0.6),
    ]
    summary = scorer.summarize(rows)

    assert summary["by_source"]["bipia"]["n"] == 2
    assert summary["by_source"]["bipia"]["coverage"] == pytest.approx(0.7)
    assert summary["by_source"]["bipia"]["hallucination_rate"] == pytest.approx(0.2)
    assert summary["by_source"]["llmail"]["n"] == 1
    assert summary["by_source"]["llmail"]["reward"] == pytest.approx(0.6)
    # Per-setting block is unaffected: both bipia+llmail rows are AP.
    assert summary["AP"]["n"] == 3


def test_judge_llm_summarize_unknown_setting_bucket() -> None:
    """Rows without a `setting` key bucket under `_unknown`."""
    scorer = weave_eval.JudgeLLMScorer(judge_model="stub")
    rows = [
        {"recall": 0.5, "mean_hallucination_score": 0.0, "reward": 0.3,
         "length_penalty": 0.0},  # no setting key
    ]
    summary = scorer.summarize(rows)
    assert "_unknown" in summary
    assert summary["_unknown"]["n"] == 1
