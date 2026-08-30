# Contributing to PRISM-eval

This guide covers extensions to the evaluation harness. For the benchmark
definition and released configurations, start with the [README](README.md).

## Development setup

```bash
uv sync --extra dev
uv run pytest tests -q
```

Unit tests must not require model checkpoints, GPU access, credentials, or live
API endpoints. Document hardware-dependent reproduction commands in
[docs/REPRODUCING.md](docs/REPRODUCING.md).

## Add a runner

A runner implements the protocol in `prism_eval/runner.py`:

```python
class ITMRunner(Protocol):
    def setup(self, checkpoint_path: str, device: str = "cuda") -> None: ...
    def run_eval(self, eval_record: EvalRecord) -> EvalResult: ...
```

To add one:

1. Create `prism_eval/runners/<name>.py`.
2. Register the runner in `RUNNER_TYPES` and `build_runner` in
   `prism_eval/runners/__init__.py`.
3. Add its name to `RunnerConfig.type` in `prism_eval/config.py`.
4. Add a configuration under `configs/`.
5. Add tests for setup, single-record inference, and configuration validation.

Implement `run_batch(records) -> list[EvalResult]` when the model supports
batched inference. It must preserve input order and return results equivalent to
repeated `run_eval` calls.

A runner without a checkpoint must encode all behavior-changing parameters in
`RunnerConfig.identity()`. This keeps distinct systems from sharing a result or
leaderboard identity.

## Add a scorer

The end-to-end evaluator uses `weave.Scorer` subclasses in
`prism_eval/weave_eval.py`. A scorer declares the dataset fields it consumes in
its `score` signature and returns a dictionary of per-record values:

```python
class MyScorer(weave.Scorer):
    @weave.op()
    def score(self, instructions: list[str], output: dict) -> dict:
        return {"my_metric": ...}

    def summarize(self, score_rows: list[dict]) -> dict:
        return {"my_metric": ...}
```

Register the scorer in the `evaluate` command in `prism_eval/cli.py`, add a
field to `ScoringConfig`, and expose any leaderboard columns in
`DEFAULT_LEADERBOARD_METRICS`.

All scorers must accept `output`, even when they do not use it. Scorer ordering
is also significant: `AdversarialDetectionScorer` consumes cached per-claim
scores from `JudgeLLMScorer`, so it must run after that scorer with the same
judge identity.

The offline and Weave paths must use the same scorer implementation and
aggregation logic.

## Add evaluation records

The schema is defined by `EvalSuite` and `EvalRecord` in
`prism_eval/schema.py`. Ground-truth instructions belong in
`instruction_sources["original"]` and should be phrased as individual
directives.

Do not modify `data/eval_suite_v2_final.json` in place. It is the artifact used
for the published results. Add a new versioned suite and configuration instead.

For adversarial settings, retain a record only after verifying that the target
behavior occurred. Include a benign comparison arm when reporting detection or
false-positive metrics. Record the source, license, construction process, and
known selection effects in [DATA_CARD.md](DATA_CARD.md).

## Preserve comparability

Two runs are comparable only when all of the following match:

- suite contents and filters
- scorer set and scorer configuration
- `evaluation.evaluation_name`
- judge identity for judge-scored metrics

Changing the suite, judge prompt, judge model, reward formula, or scorer set
defines a new evaluation condition. Use a new explicit `evaluation_name` and
leaderboard name, and document the difference.

## Pull requests

Keep changes focused and avoid committing generated results, checkpoints,
credentials, or local paths. Update the README, data card, rubric, or
reproduction guide whenever a user-visible interface or evaluation condition
changes.
