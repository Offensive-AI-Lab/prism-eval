# Extending PRISM-eval

How to add a runner, add a scorer, or add eval records. For what the benchmark
measures, start with the [README](README.md).

## Setup

```bash
uv sync --extra dev
source .venv/bin/activate
pytest tests/ -q      # ~191 tests, no GPU or network required
```

The test suite runs entirely on fakes — a `_FakeRunner` stands in for the GPU
and stub clients stand in for the judge — so it stays fast and offline. Keep it
that way: a test that needs a checkpoint or a live endpoint belongs in
`docs/REPRODUCING.md` as a documented command, not in `tests/`.

## Adding a runner

A runner turns an `EvalRecord` into an `EvalResult` carrying an `itm_report`.
That is the entire contract — `prism_eval/runner.py`:

```python
class ITMRunner(Protocol):
    def setup(self, checkpoint_path: str, device: str = "cuda") -> None: ...
    def run_eval(self, eval_record: EvalRecord) -> EvalResult: ...
```

1. Write the class in `prism_eval/runners/your_runner.py`. Model components
   shared by the activation-bridge runners (hooks, projections)
   live in `prism_eval/runners/models.py` — reuse them rather than re-deriving.
2. Register it in **one** place, `prism_eval/runners/__init__.py`: add the name
   to `RUNNER_TYPES` and a branch to `build_runner`. Both the Weave path and
   the offline path resolve through it, so nothing else needs touching.
3. Add the name to the `Literal` on `RunnerConfig.type` in
   `prism_eval/config.py` (there's a comment on each pointing at the other).
4. Add a config under `configs/`. To be comparable with the published rows it
   must match `configs/main/*.yaml` on `suite.path`, `suite.settings`, the
   scorer set, and `evaluation.evaluation_name` — see "Comparability" below.

**Optional but worth it:** implement `run_batch(records) -> list[EvalResult]`.
If present, `_run_inference_batches` uses it and GPU work batches instead of
looping one record at a time. `scripts/sanity_batch_vs_sequential.py` checks
that your batched path agrees with `run_eval`; run it before trusting a new
`run_batch`.

Runners that need no checkpoint (like `text_only_baseline`) encode their
identity in `RunnerConfig.identity()` instead, so that changing a parameter
produces a distinct leaderboard row.

## Adding a scorer

Scorers are `weave.Scorer` subclasses in `prism_eval/weave_eval.py`. A scorer
declares whichever dataset-row keys it needs in its `score` signature, plus
`output`:

```python
class MyScorer(weave.Scorer):
    @weave.op()
    def score(self, instructions: list[str], output: dict, setting: str) -> dict:
        return {"my_metric": ...}

    def summarize(self, score_rows: list[dict]) -> dict:
        return {"my_metric": mean(r["my_metric"] for r in score_rows)}
```

Both execution paths dispatch by signature introspection, so you get the keys
you asked for and nothing else. Available keys are whatever
`eval_record_to_row` produces: `eval_id`, `dataset_type`, `setting`,
`difficulty`, `prompt`, `prompt_turns`, `instructions`, `context_length`.

`summarize` is what produces leaderboard columns **and** the offline
`summary.json` — implement it, or aggregation falls back to meaning every
numeric field.

Wire the scorer into `evaluate` in `prism_eval/cli.py` behind a `ScoringConfig`
flag, and add its columns to `DEFAULT_LEADERBOARD_METRICS`.

Two constraints worth knowing:

- **Scorer order matters.** `AdversarialDetectionScorer` reads the per-bullet
  scores `JudgeLLMScorer` writes to an in-process cache, so it must come after
  it in the list and use the same `judge_model`. The CLI enforces both.
- **Every `score` must accept `output`** even if unused — Weave's Scorer
  contract requires it, and omitting it fails mid-run rather than at startup.

## Adding eval records

The suite format is `EvalSuite` / `EvalRecord` in `prism_eval/schema.py`.
Ground truth is `instruction_sources["original"]`: the list of instructions the
model was actually given, phrased as directives.

`scripts/build_suite.py` builds the shipped settings and is the place to add a
new one — extend `SETTING_META`, add a fetcher or generator, and give the
records an `eval_id` prefix.

Two rules that keep the benchmark honest:

- **Verify adversarial records.** AP and HO records only ship if the injection
  actually landed / the objective was actually expressed, checked by running
  the prompt through the base model. An unverified adversarial record measures
  nothing.
- **Keep a benign arm.** Without BN-style records, hallucination and
  false-positive rates are unmeasurable and a detector that flags everything
  looks perfect.

Do not edit `data/eval_suite_v2_final.json` in place. Every published number is
computed against that exact file, and `tests/test_loader.py::TestCanonicalSuite`
will fail if its shape changes. Ship a new file instead. If you need a filtered
view at runtime, use `prism_eval.loader.filter_evals` or `suite.per_setting_limit`.

## Comparability

Two runs are comparable only if their **suite**, **scorer set** and
**`evaluation_name`** all match. Change any one and the evaluation digest
changes, which silently moves the row onto a different leaderboard rather than
raising an error. This is the single easiest thing to get wrong.

When you intentionally change something that shifts the digest — judge prompt,
judge model, reward formula, scorer set, suite file — bump
`leaderboard_name` across the affected configs, and suffix it with *what*
changed rather than a version number: `prism_main_leaderboard_judge_v3`, not
`_v2`.

## Style

Match the file you're editing: same comment density, same naming, same idiom.
Comments here explain *why*, especially where behaviour is non-obvious
(cache-then-consume, scorer ordering, the unexpanded checkpoint identity). Keep
that — those comments are load-bearing.
