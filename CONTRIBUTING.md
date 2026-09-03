# Contributing to PRISM-eval

## Development setup

```bash
uv sync --extra dev
uv run pytest tests -q
```

Default tests must not require checkpoints, GPU access, credentials, or live
endpoints. Put hardware-dependent instructions in
[the reproduction guide](docs/REPRODUCING.md).

## Add a runner

Implement the `setup` and `run_eval` methods in the
[`ITMRunner` protocol](prism_eval/runner.py). Then:

1. Add the implementation under `prism_eval/runners/`.
2. Register it in `RUNNER_TYPES` and `build_runner` in
   `prism_eval/runners/__init__.py`.
3. Extend `RunnerConfig.type` in `prism_eval/config.py`.
4. Add a configuration and tests using mocks for model loading and inference.

Optional `run_batch(records)` must preserve input order and match repeated
`run_eval` calls. For checkpoint-free runners, include all behavior-changing
parameters in `RunnerConfig.identity()`.

## Add a scorer

The evaluator uses `weave.Scorer` subclasses in `prism_eval/weave_eval.py`.
Implement `score` for per-record values and `summarize` for aggregation.
Register the scorer in `prism_eval/cli.py` and add its settings to
`ScoringConfig`; add leaderboard columns to `DEFAULT_LEADERBOARD_METRICS`
if needed.

Every scorer must accept `output`. Adversarial detection must run after
`JudgeLLMScorer` with the same judge identity because it reuses the coverage
scores. Offline and Weave evaluation must share scorer and aggregation logic.

## Data and evaluation conditions

Use the schema in [`prism_eval/schema.py`](prism_eval/schema.py). Ground-truth
instructions belong in `instruction_sources["original"]`, one directive per
item.

Keep `data/eval_suite.json` unchanged for paper reproduction. Add a separate
suite and configuration for new records, and document their sources, licenses,
construction, and selection criteria in [the data card](DATA_CARD.md).

Keep suite filters and scoring rules fixed when comparing systems. Changes to
the suite, judge model or prompt, reward, or scorer set need a new evaluation
name and a description of the changed condition.

## Pull requests

Keep changes focused, test changed logic, and update the affected documentation.
Do not commit generated results, checkpoints, credentials, or machine-specific
paths.
