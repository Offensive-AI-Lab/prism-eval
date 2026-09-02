"""CLI for prism-eval.

Commands:
    evaluate  Run an experiment end-to-end from a YAML config. This is the
              entry point for every number in the paper; use --offline to
              skip Weave and write results to local JSON.
    run       Lower-level: run evals through a runner → EvalResult JSONL
    score     Lower-level: score a run JSONL → ScoredResult JSONL
    analyze   Per-eval metrics and scorer comparison from a scored JSONL
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from dotenv import load_dotenv

from prism_eval.runners import RUNNER_TYPES
from prism_eval.schema import EvalResult, ScoredResult

# Auto-load .env from project root (if it exists)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@click.group()
def cli():
    """ITM Eval Suite — evaluate instruction and intent extraction."""
    pass


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True), required=True,
              help="Experiment YAML config (see configs/main/qwen3.5-9b-grpo.yaml)")
@click.option("--offline", is_flag=True,
              help="Run without Weave: no W&B account needed. Writes "
                   "results/<experiment>/{rows.jsonl,summary.json} instead of "
                   "tracing and publishing a leaderboard.")
@click.option("--results-dir", type=click.Path(), default="results",
              help="Root for --offline output (default: results/).")
@click.option("--no-publish-leaderboard", is_flag=True,
              help="Run the Evaluation but skip publishing the Leaderboard")
def evaluate(config_path, offline, results_dir, no_publish_leaderboard):
    """Run an experiment through the `weave.Evaluation` framework.

    Builds a `weave.Evaluation` (one row per eval, configurable scorers) and
    runs the configured runner as an `ITMModel` against it. For human
    annotation, a dedicated ``itm_annotate`` root trace is emitted per row
    *before* the Evaluation runs, with flat level-1 inputs/outputs shaped
    for the annotation-queue UI. Model inference is cached across the two
    passes so each row runs predict exactly once (cache-then-consume).

    Aggregated metrics show up in the Weave UI under the Evaluation tab,
    and (unless --no-publish-leaderboard) a Leaderboard is created/updated
    so different runners are directly comparable.

    Phases:
      1. annotate pre-pass (if annotation.emit_trace) — `itm_annotate`
         root traces per row, caching predict outputs
      2. evaluate — runs the Evaluation against the configured ITMModel
         (predict reads from the cache)
      3. leaderboard — publish/update the Leaderboard for this Evaluation
    """
    import asyncio
    import os

    import weave

    from prism_eval.config import load_config
    from prism_eval.weave_eval import (
        AdversarialDetectionScorer,
        BERTScoreScorer,
        ExactMatchScorer,
        ITMModel,
        JudgeLLMScorer,
        TokenF1Scorer,
        annotate_runner_context,
        batched_annotate_pre_pass,
        build_evaluation,
        clear_judge_scores_cache,
        clear_predict_cache,
        prewarm_runner,
        publish_leaderboard,
    )

    cfg = load_config(Path(config_path))

    if not offline and not cfg.experiment.weave_project:
        raise click.UsageError(
            "experiment.weave_project is not set, so there is nowhere to trace "
            "to.\nEither run with --offline (no W&B account needed), or set "
            "weave_project to '<entity>/<project>' in the config."
        )

    click.echo(f"Experiment: {cfg.experiment.name}")
    click.echo(f"  mode:       {'offline' if offline else 'weave'}")
    click.echo(f"  evaluation: {cfg.resolved_evaluation_name()}")
    for p in cfg.suite.paths:
        click.echo(f"  suite:      {p}")
    click.echo(f"  runner:     {cfg.runner.type}")
    click.echo(f"  checkpoint: {cfg.runner.checkpoint or cfg.runner.identity()}")

    evals = cfg.load_records()
    click.echo(f"  records:    {len(evals)}")

    # ── Build scorers ──────────────────────────────────────────────────────
    # exact_match + token_f1 are cheap and always on (matches `score` semantics).
    scorers: list = [ExactMatchScorer(), TokenF1Scorer()]
    if cfg.scoring.bertscore:
        scorers.append(BERTScoreScorer())
    if cfg.scoring.judge_llm:
        # Judge LLM env overrides. The Scorer constructor takes the model
        # name explicitly so it becomes part of the scorer's identity in
        # Weave (different judge = different column).
        if cfg.scoring.judge_base_url:
            os.environ["PRISM_EVAL_BASE_URL"] = cfg.scoring.judge_base_url
        judge_model = cfg.scoring.judge_model or os.environ.get("PRISM_EVAL_MODEL")
        judge_base_url = cfg.scoring.judge_base_url or os.environ.get("PRISM_EVAL_BASE_URL")
        if not judge_model:
            raise click.UsageError(
                "scoring.judge_llm is true but no judge_model is set "
                "(neither in config nor PRISM_EVAL_MODEL env var)."
            )
        scorers.append(
            JudgeLLMScorer(
                judge_model=judge_model,
                judge_base_url=judge_base_url,
            )
        )
    if cfg.scoring.adversarial_detection:
        # AdversarialDetectionScorer reads JudgeLLMScorer's per-bullet
        # scores from _JUDGE_SCORES_CACHE — must run AFTER the calibrated
        # judge in the scorer list, and judge_llm must be enabled.
        if not cfg.scoring.judge_llm:
            raise click.UsageError(
                "scoring.adversarial_detection requires scoring.judge_llm=true "
                "(the adversarial metric reuses the calibrated judge's "
                "per-bullet scores via an in-process cache)."
            )
        adv_model = (
            cfg.scoring.adversarial_judge_model
            or cfg.scoring.judge_model
            or os.environ.get("PRISM_EVAL_MODEL")
        )
        adv_base_url = (
            cfg.scoring.adversarial_judge_base_url
            or cfg.scoring.judge_base_url
            or os.environ.get("PRISM_EVAL_BASE_URL")
        )
        # Cache key is (eval_id, judge_model); using a different model for
        # the identifier would silently miss the cache and skip every row.
        if adv_model != judge_model:
            raise click.UsageError(
                f"scoring.adversarial_judge_model ({adv_model!r}) must match "
                f"scoring.judge_model ({judge_model!r}); the score-reuse "
                f"cache is keyed on (eval_id, judge_model)."
            )
        scorers.append(
            AdversarialDetectionScorer(
                judge_model=adv_model,
                judge_base_url=adv_base_url,
            )
        )
    click.echo(f"  scorers:    {[type(s).__name__ for s in scorers]}")

    # Fresh caches per invocation. Stale entries across runs would let the
    # adversarial metric score rows against a previous run's judge output.
    clear_predict_cache()
    clear_judge_scores_cache()

    if offline:
        _evaluate_offline(cfg, evals, scorers, Path(results_dir))
        return

    # ── Phase 1: build + run the Evaluation ────────────────────────────────
    weave.init(cfg.experiment.weave_project)

    # Pre-warm the runner BEFORE evaluation.evaluate() spins up its thread
    # pool. weave.Evaluation runs predict() concurrently across worker threads;
    # if the cache is cold when the first batch hits, every worker races to
    # load its own copy of the model and OOMs the GPU. Loading once here,
    # synchronously, populates the cache so every worker just hits a warm one.
    click.echo(f"Loading runner: {cfg.runner.type} on {cfg.runner.device}...")
    runner_identity = cfg.runner.identity()
    prewarm_runner(cfg.runner.type, runner_identity, cfg.runner.device)
    click.echo("Runner ready.")

    model = ITMModel(
        # display_name (from RunnerConfig) becomes the leaderboard row
        # label. Multiple configs sharing a runner type would otherwise
        # collapse to one row in the Weave UI.
        name=cfg.runner.display_name or cfg.runner.type,
        runner_type=cfg.runner.type,
        checkpoint=runner_identity,
        device=cfg.runner.device,
    )
    evaluation = build_evaluation(
        name=cfg.resolved_evaluation_name(),
        records=evals,
        scorers=scorers,
    )

    # Override Weave's default random display name (e.g. "eval-2026-04-10-loyal-tiger")
    # so evaluation calls show as "prism / all_v1" in the Evaluations UI.
    eval_display = f"{cfg.runner.type} / {cfg.resolved_evaluation_name()}"
    evaluation.evaluate.__func__.call_display_name = lambda call, _n=eval_display: _n

    attrs = cfg.weave_attributes()
    click.echo(f"Running Evaluation with attributes: {attrs}")

    with weave.attributes(attrs):
        if cfg.annotation.emit_trace:
            batch_size = cfg.annotation.batch_size
            trace_workers = cfg.annotation.trace_workers
            max_batch_tokens = cfg.annotation.max_batch_tokens
            click.echo(
                f"Annotation pre-pass: emitting {len(evals)} "
                f"`{cfg.annotation.op_name}` root traces "
                f"(batch_size={batch_size}, max_batch_tokens={max_batch_tokens}, "
                f"trace_workers={trace_workers})..."
            )

            def _on_batch(b_idx: int, n_done: int, n_total: int) -> None:
                if (b_idx + 1) % 5 == 0 or n_done == n_total:
                    click.echo(f"  batch {b_idx + 1}: {n_done}/{n_total} traces emitted")

            with annotate_runner_context(
                cfg.runner.type, runner_identity, cfg.runner.device
            ):
                batched_annotate_pre_pass(
                    evals,
                    runner_type=cfg.runner.type,
                    checkpoint=runner_identity,
                    device=cfg.runner.device,
                    batch_size=batch_size,
                    max_batch_tokens=max_batch_tokens,
                    trace_workers=trace_workers,
                    on_batch_complete=_on_batch,
                )
            click.echo("Annotation pre-pass complete — predict outputs cached for Evaluation.")
        summary = asyncio.run(evaluation.evaluate(model))

    click.echo("\n=== EVALUATION SUMMARY ===")
    click.echo(json.dumps(summary, indent=2, default=str))

    # ── Phase 2: publish leaderboard ───────────────────────────────────────
    if no_publish_leaderboard or not cfg.evaluation.publish_leaderboard:
        click.echo("\nSkipping leaderboard publish.")
        return

    click.echo(f"\nPublishing leaderboard: {cfg.resolved_leaderboard_name()}")
    metrics: list[tuple[str, str]] | None = None
    if not cfg.scoring.judge_llm:
        # Default leaderboard columns read from JudgeLLMScorer's custom
        # summarize(); fall back to the always-on cheap scorers (recall proxy
        # only — no precision/F1/hallucination signal without the judge) when
        # the judge isn't running.
        metrics = [
            ("ExactMatchScorer", "instruction_score.mean"),
            ("TokenF1Scorer", "instruction_score.mean"),
        ]
    lb_ref = publish_leaderboard(
        evaluation,
        name=cfg.resolved_leaderboard_name(),
        # Intentionally not threading experiment.notes here: notes typically
        # carry runner-specific text ("…, cross_attention_bridge"), and the
        # description is part of the leaderboard spec content — using it churns
        # the leaderboard's digest on every runner switch. Notes are still
        # captured per-call via cfg.weave_attributes() for filtering in Weave.
        description="",
        metrics=metrics,
    )
    click.echo(f"Leaderboard published: {lb_ref}")


def _evaluate_offline(cfg, evals, scorers, results_root: Path) -> None:
    """`evaluate --offline`: run + score locally, write rows.jsonl + summary.json.

    Uses the same runner, the same scorer objects and each scorer's own
    `summarize()` as the Weave path, so the aggregate numbers are directly
    comparable to a published leaderboard row.
    """
    from prism_eval.weave_eval import prewarm_runner, run_offline_evaluation

    click.echo(f"Loading runner: {cfg.runner.type} on {cfg.runner.device}...")
    runner_identity = cfg.runner.identity()
    prewarm_runner(cfg.runner.type, runner_identity, cfg.runner.device)
    click.echo("Runner ready.")

    batch_size = cfg.annotation.batch_size
    max_batch_tokens = cfg.annotation.max_batch_tokens
    # trace_workers is the config's "concurrent network-bound calls" budget;
    # offline has no traces to emit, so the same budget covers judge calls.
    scoring_workers = cfg.annotation.trace_workers

    def _on_batch(b_idx: int, n_done: int, n_total: int) -> None:
        if (b_idx + 1) % 5 == 0 or n_done == n_total:
            click.echo(f"  batch {b_idx + 1}: {n_done}/{n_total} records run")

    click.echo(
        f"Running {len(evals)} records (batch_size={batch_size}, "
        f"max_batch_tokens={max_batch_tokens}, scoring_workers={scoring_workers})..."
    )
    result = run_offline_evaluation(
        evals,
        scorers,
        runner_type=cfg.runner.type,
        checkpoint=runner_identity,
        device=cfg.runner.device,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        scoring_workers=scoring_workers,
        on_batch_complete=_on_batch,
    )

    out_dir = results_root / cfg.experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = out_dir / "rows.jsonl"
    with open(rows_path, "w") as f:
        for row in result["rows"]:
            f.write(json.dumps(row, default=str) + "\n")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "experiment": cfg.experiment.name,
        "evaluation_name": cfg.resolved_evaluation_name(),
        "runner": cfg.runner.type,
        "display_name": cfg.runner.display_name or cfg.runner.type,
        "checkpoint": runner_identity,
        "n_records": len(evals),
        "scorers": [type(s).__name__ for s in scorers],
        "summary": result["summary"],
    }, indent=2, default=str))

    click.echo("\n=== EVALUATION SUMMARY ===")
    click.echo(json.dumps(result["summary"], indent=2, default=str))
    click.echo(f"\nRows:    {rows_path}")
    click.echo(f"Summary: {summary_path}")


@cli.command()
@click.option("--suite", "suite_path", type=click.Path(exists=True), required=True)
@click.option("--runner", type=click.Choice(RUNNER_TYPES), default="prism")
@click.option("--checkpoint", type=click.Path(), default=None,
              help="Checkpoint .pt. Required for every runner except text_only_baseline.")
@click.option("--device", default="cuda")
@click.option("--setting", default=None, help="Comma-separated settings to filter: AP,HO,BC,BN")
@click.option("-o", "--output", type=click.Path(), required=True)
def run(suite_path, runner, checkpoint, device, setting, output):
    """Run evals through an ITM runner.

    Produces a JSONL of EvalResults. This is the lower-level flow — for
    end-to-end Weave tracing + Leaderboard + annotation-ready traces, use
    `prism-eval evaluate --config <file>` instead.
    """
    from prism_eval.config import resolve_checkpoint
    from prism_eval.loader import filter_evals, load_suite
    from prism_eval.runners import build_runner

    suite = load_suite(Path(suite_path))
    evals = suite.evals

    if setting:
        settings = [s.strip() for s in setting.split(",")]
        evals = filter_evals(evals, settings=settings)

    click.echo(f"Running {len(evals)} evals with {runner} runner...")

    # text_only_baseline has no checkpoint — its identity is the
    # (response_model_id, baseline_llm, tail_tokens) tuple instead.
    if not checkpoint and runner != "text_only_baseline":
        raise click.UsageError(f"--checkpoint is required for the {runner} runner")

    r = build_runner(runner)
    r.setup(resolve_checkpoint(checkpoint) if checkpoint else None, device=device)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, eval_rec in enumerate(evals):
            result = r.run_eval(eval_rec)
            f.write(result.model_dump_json() + "\n")

            if (i + 1) % 10 == 0:
                click.echo(f"  {i + 1}/{len(evals)} complete")

    click.echo(f"Results written to {output_path}")


@cli.command()
@click.option("--suite", "suite_path", type=click.Path(exists=True), required=True, help="Eval suite JSON for ground truth")
@click.option("--results", type=click.Path(exists=True), required=True)
@click.option("--bertscore/--no-bertscore", default=False)
@click.option("--judge-llm/--no-judge-llm", default=False)
@click.option("-o", "--output", type=click.Path(), required=True)
def score(suite_path, results, bertscore, judge_llm, output):
    """Score ITM results with all enabled methods.

    Produces a JSONL of ScoredResult objects. To get Weave traces (for
    annotation + leaderboard), use `prism-eval evaluate` instead — it runs
    scoring inside a `weave.Evaluation` and produces the annotation-ready
    ``itm_annotate`` root traces in one shot.
    """
    from prism_eval.loader import load_suite
    from prism_eval.scoring.score_all import score_eval

    suite = load_suite(Path(suite_path))
    record_map = {r.eval_id: r for r in suite.evals}

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_lines = Path(results).read_text().strip().splitlines()

    click.echo("Scoring...")
    n_scored = 0
    with open(output_path, "w") as f:
        for line in results_lines:
            if not line.strip():
                continue
            eval_result = EvalResult.model_validate_json(line)
            record = record_map.get(eval_result.eval_id)
            if record is None:
                click.echo(f"  WARNING: no record for {eval_result.eval_id}, skipping")
                continue

            scored = score_eval(
                record,
                eval_result,
                use_bertscore=bertscore,
                use_judge_llm=judge_llm,
            )
            f.write(scored.model_dump_json() + "\n")
            n_scored += 1

    click.echo(f"Scored {n_scored} evals → {output_path}")


@cli.command()
@click.option("--suite", "suite_path", type=click.Path(exists=True), required=True)
@click.option("--scored", type=click.Path(exists=True), required=True)
@click.option("-o", "--output-dir", type=click.Path(), default="results/analysis")
def analyze(suite_path, scored, output_dir):
    """Per-eval metrics and scorer comparison from a scored JSONL.

    Also emits decay curves when the scored records carry multi-turn
    `instruction_positions`. The shipped suites are single-turn, so that
    file is normally absent.
    """
    from prism_eval.loader import load_suite
    from prism_eval.metrics import decay_analysis, per_eval_metrics, scorer_comparison

    suite = load_suite(Path(suite_path))
    record_map = {r.eval_id: r for r in suite.evals}

    scored_results = []
    for line in Path(scored).read_text().strip().splitlines():
        if line.strip():
            scored_results.append(ScoredResult.model_validate_json(line))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Per-eval metrics
    all_metrics = {}
    for sr in scored_results:
        all_metrics[sr.eval_id] = per_eval_metrics(sr)
    (out / "per_eval_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    click.echo(f"Per-eval metrics → {out / 'per_eval_metrics.json'}")

    # Scorer comparison
    comp = scorer_comparison(scored_results)
    (out / "scorer_comparison.json").write_text(json.dumps(comp, indent=2))
    click.echo(f"Scorer comparison → {out / 'scorer_comparison.json'}")

    # Decay analysis (per scorer)
    scorer_names = set()
    for sr in scored_results:
        scorer_names.update(sr.scores.keys())

    decay_results = {}
    for scorer_name in sorted(scorer_names):
        decay = decay_analysis(record_map, scored_results, scorer_name)
        if decay:
            decay_results[scorer_name] = {str(k): v for k, v in decay.items()}

    if decay_results:
        (out / "decay_curves.json").write_text(json.dumps(decay_results, indent=2))
        click.echo(f"Decay curves → {out / 'decay_curves.json'}")
    else:
        click.echo("No generated evals found for decay analysis")

    # Summary
    click.echo(f"\nAnalyzed {len(scored_results)} evals across {len(scorer_names)} scorers")
    for name, data in comp.items():
        click.echo(
            f"  {name}: instruction_recall={data['mean_instruction_recall']:.3f}, "
            f"mean_hallucination_score={data['mean_hallucination_score']:.3f}"
        )


if __name__ == "__main__":
    cli()
