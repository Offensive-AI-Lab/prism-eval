"""Tests for `prism-eval evaluate --offline` (weave_eval.run_offline_evaluation).

Offline mode exists so the suite is usable without a W&B account. Its whole
value depends on producing the *same* numbers as the traced path, so these
tests pin the two properties that guarantee that:

  1. Every record is run exactly once through the runner, batched.
  2. Aggregation goes through each scorer's own `summarize()` — the same
     method that feeds the Weave leaderboard columns.

A fake runner stands in for the GPU; the scorers are the real ones.
"""

from __future__ import annotations

import json
import threading

import pytest

from prism_eval import weave_eval
from prism_eval.schema import EvalRecord, EvalResult


class _FakeRunner:
    """Records how many times each eval_id was run."""

    def __init__(self, report_for=None) -> None:
        self.calls: list[str] = []
        self._report_for = report_for or (lambda rec: f"1. {rec.eval_id}")

    def setup(self, checkpoint: str, device: str = "cuda") -> None:
        return None

    def run_eval(self, record: EvalRecord) -> EvalResult:
        self.calls.append(record.eval_id)
        return EvalResult(
            eval_id=record.eval_id,
            runner="fake",
            model_response=f"resp::{record.eval_id}",
            itm_report=self._report_for(record),
            timestamp="2026-07-31T00:00:00Z",
            input_tokens=1,
            output_tokens=2,
            activation_tokens=3,
        )


@pytest.fixture(autouse=True)
def _disable_weave_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEAVE_DISABLED", "true")


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> _FakeRunner:
    runner = _FakeRunner()
    monkeypatch.setattr(weave_eval, "_runner_cache", {"fake|ckpt|cpu": runner})
    monkeypatch.setattr(
        weave_eval, "_runner_locks", {"fake|ckpt|cpu": threading.Lock()}
    )
    weave_eval.clear_predict_cache()
    yield runner
    weave_eval.clear_predict_cache()


def _records(n: int) -> list[EvalRecord]:
    return [
        EvalRecord(
            eval_id=f"BN-{i:03d}",
            setting="BN",
            category="baseline",
            structural_tags=["single-turn"],
            difficulty="easy",
            source="hand_crafted",
            prompt=f"Prompt {i}",
            instruction_sources={"original": [f"instruction {i}"]},
        )
        for i in range(n)
    ]


def _run(records, scorers, batch_size=4):
    return weave_eval.run_offline_evaluation(
        records,
        scorers,
        runner_type="fake",
        checkpoint="ckpt",
        device="cpu",
        batch_size=batch_size,
    )


def test_runs_each_record_exactly_once(fake_runner: _FakeRunner) -> None:
    records = _records(10)
    _run(records, [weave_eval.ExactMatchScorer()])

    assert fake_runner.calls == [r.eval_id for r in records]
    assert len(fake_runner.calls) == len(set(fake_runner.calls)) == 10


def test_batching_covers_a_partial_final_batch(fake_runner: _FakeRunner) -> None:
    # 7 records at batch_size 4 => batches of 4 and 3.
    records = _records(7)
    result = _run(records, [weave_eval.ExactMatchScorer()], batch_size=4)

    assert len(fake_runner.calls) == 7
    assert len(result["rows"]) == 7


def test_rows_carry_output_and_per_scorer_scores(fake_runner: _FakeRunner) -> None:
    records = _records(3)
    result = _run(records, [weave_eval.ExactMatchScorer(), weave_eval.TokenF1Scorer()])

    assert [r["eval_id"] for r in result["rows"]] == [r.eval_id for r in records]
    for row in result["rows"]:
        assert row["output"]["itm_report"] == f"1. {row['eval_id']}"
        assert row["output"]["model_response"] == f"resp::{row['eval_id']}"
        assert set(row["scores"]) == {"ExactMatchScorer", "TokenF1Scorer"}
        assert row["setting"] == "BN"


def test_summary_has_one_entry_per_scorer(fake_runner: _FakeRunner) -> None:
    result = _run(_records(5), [weave_eval.ExactMatchScorer(), weave_eval.TokenF1Scorer()])
    assert set(result["summary"]) == {"ExactMatchScorer", "TokenF1Scorer"}


def test_summary_uses_the_scorers_own_summarize(
    fake_runner: _FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leaderboard columns come from Scorer.summarize; offline must use it.

    If offline ever grows its own aggregation, this test fails — which is the
    point: a parallel implementation is how offline numbers silently drift
    from published ones.
    """
    seen: dict = {}

    class _SummarizingScorer(weave_eval.ExactMatchScorer):
        def summarize(self, score_rows: list[dict]) -> dict:
            seen["n_rows"] = len(score_rows)
            return {"sentinel": 42}

    result = _run(_records(6), [_SummarizingScorer()])

    assert seen["n_rows"] == 6, "summarize must receive every row's score"
    assert result["summary"]["_SummarizingScorer"] == {"sentinel": 42}


def test_scorers_receive_only_their_declared_row_keys(fake_runner: _FakeRunner) -> None:
    """Offline dispatch mirrors weave.Evaluation: signature-matched kwargs.

    ExactMatchScorer declares (instructions, output) and must not be handed
    eval_id/setting/_record_json, or the call would raise TypeError.
    """
    captured: dict = {}

    class _StrictScorer(weave_eval.ExactMatchScorer):
        def score(self, instructions: list[str], output: dict) -> dict:
            captured["instructions"] = instructions
            captured["output_keys"] = sorted(output)
            return {"instruction_score": 1.0}

    _run(_records(1), [_StrictScorer()])

    assert captured["instructions"] == ["instruction 0"]
    assert "itm_report" in captured["output_keys"]


def test_json_serialisable(fake_runner: _FakeRunner) -> None:
    """`_evaluate_offline` writes these straight to disk — they must serialise."""
    result = _run(_records(3), [weave_eval.ExactMatchScorer(), weave_eval.TokenF1Scorer()])
    for row in result["rows"]:
        json.loads(json.dumps(row, default=str))
    json.loads(json.dumps(result["summary"], default=str))


def test_empty_record_list_is_handled(fake_runner: _FakeRunner) -> None:
    result = _run([], [weave_eval.ExactMatchScorer()])
    assert result["rows"] == []
    assert fake_runner.calls == []


# ─────────────────────────────────────────────────────────────────────────────
# CLI-level: `prism-eval evaluate --offline` writes the expected artifacts
# ─────────────────────────────────────────────────────────────────────────────


def test_cli_offline_writes_rows_and_summary(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end through the CLI: no Weave, no W&B key, files on disk."""
    from click.testing import CliRunner

    from prism_eval.cli import cli

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    identity = "${PRISM_EVAL_CHECKPOINT_DIR}/prism-qwen3.5-9b-grpo.pt"
    runner = _FakeRunner()
    # Seeding the cache means _get_runner never resolves the checkpoint path,
    # so no real weights are needed.
    monkeypatch.setattr(
        weave_eval, "_runner_cache", {f"prism|{identity}|cpu": runner}
    )
    monkeypatch.setattr(
        weave_eval,
        "_runner_locks",
        {f"prism|{identity}|cpu": threading.Lock()},
    )

    config = tmp_path / "smoke.yaml"
    config.write_text(
        f"""
experiment:
  name: offline_smoke
  weave_project: null
suite:
  path: data/eval_suite_v2_final.json
  settings: [BN]
  per_setting_limit:
    BN: 4
runner:
  type: prism
  checkpoint: {identity}
  device: cpu
scoring:
  bertscore: false
  judge_llm: false
annotation:
  emit_trace: true
  batch_size: 2
evaluation:
  evaluation_name: offline_smoke_v1
  publish_leaderboard: true
"""
    )

    result = CliRunner().invoke(
        cli,
        ["evaluate", "--config", str(config), "--offline",
         "--results-dir", str(tmp_path / "results")],
    )
    assert result.exit_code == 0, result.output

    out_dir = tmp_path / "results" / "offline_smoke"
    rows = (out_dir / "rows.jsonl").read_text().strip().splitlines()
    assert len(rows) == 4
    assert all(json.loads(line)["setting"] == "BN" for line in rows)

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["n_records"] == 4
    assert summary["runner"] == "prism"
    # The unexpanded identity is what gets recorded, so a run on another
    # machine produces a byte-identical summary header.
    assert summary["checkpoint"] == identity
    assert set(summary["summary"]) == {"ExactMatchScorer", "TokenF1Scorer"}
    assert len(runner.calls) == 4


def test_cli_refuses_weave_mode_without_a_project(tmp_path) -> None:
    """Without weave_project, plain `evaluate` must say what to do."""
    from click.testing import CliRunner

    from prism_eval.cli import cli

    config = tmp_path / "no_project.yaml"
    config.write_text(
        """
experiment:
  name: no_project
  weave_project: null
suite:
  path: data/eval_suite_v2_final.json
  settings: [BN]
runner:
  type: prism
  checkpoint: ${PRISM_EVAL_CHECKPOINT_DIR}/prism-qwen3.5-9b-grpo.pt
"""
    )
    result = CliRunner().invoke(cli, ["evaluate", "--config", str(config)])
    assert result.exit_code != 0
    assert "--offline" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent row scoring. Scoring is network-bound (~1s per judge call per
# scorer per row), so rows fan out across threads. These pin the invariants
# that make that safe.
# ─────────────────────────────────────────────────────────────────────────────


def test_row_order_is_preserved_under_concurrency(fake_runner: _FakeRunner) -> None:
    """Completion order must not leak into rows.jsonl or into summarize().

    Scorer sleeps longest on the *first* records, so completion order is the
    reverse of input order. Output must still be input order.
    """
    import time

    class _SkewedScorer(weave_eval.ExactMatchScorer):
        def score(self, instructions: list[str], output: dict) -> dict:
            idx = int(instructions[0].split()[-1])
            time.sleep((8 - idx) * 0.02)
            return {"instruction_score": float(idx)}

    result = weave_eval.run_offline_evaluation(
        _records(8), [_SkewedScorer()],
        runner_type="fake", checkpoint="ckpt", device="cpu",
        batch_size=4, scoring_workers=8,
    )

    assert [r["eval_id"] for r in result["rows"]] == [f"BN-{i:03d}" for i in range(8)]
    assert [r["scores"]["_SkewedScorer"]["instruction_score"]
            for r in result["rows"]] == [float(i) for i in range(8)]


def test_scorers_stay_ordered_within_a_row(fake_runner: _FakeRunner) -> None:
    """AdversarialDetectionScorer reads what JudgeLLMScorer cached for the SAME
    row, so scorers must not be reordered or run in parallel within a row."""
    seen: dict[str, list[str]] = {}

    class _First(weave_eval.ExactMatchScorer):
        def score(self, instructions: list[str], output: dict, eval_id: str) -> dict:
            seen.setdefault(eval_id, []).append("first")
            return {"instruction_score": 1.0}

    class _Second(weave_eval.ExactMatchScorer):
        def score(self, instructions: list[str], output: dict, eval_id: str) -> dict:
            seen.setdefault(eval_id, []).append("second")
            return {"instruction_score": 1.0}

    weave_eval.run_offline_evaluation(
        _records(12), [_First(), _Second()],
        runner_type="fake", checkpoint="ckpt", device="cpu",
        batch_size=4, scoring_workers=8,
    )

    assert len(seen) == 12
    for eval_id, order in seen.items():
        assert order == ["first", "second"], f"{eval_id}: scorers ran out of order"


def test_concurrent_and_serial_scoring_agree(fake_runner: _FakeRunner) -> None:
    """1 worker vs 8 workers must produce identical rows and summary."""
    kwargs = dict(runner_type="fake", checkpoint="ckpt", device="cpu", batch_size=4)

    serial = weave_eval.run_offline_evaluation(
        _records(10), [weave_eval.ExactMatchScorer(), weave_eval.TokenF1Scorer()],
        scoring_workers=1, **kwargs)
    weave_eval.clear_predict_cache()
    concurrent = weave_eval.run_offline_evaluation(
        _records(10), [weave_eval.ExactMatchScorer(), weave_eval.TokenF1Scorer()],
        scoring_workers=8, **kwargs)

    assert serial["rows"] == concurrent["rows"]
    assert serial["summary"] == concurrent["summary"]
