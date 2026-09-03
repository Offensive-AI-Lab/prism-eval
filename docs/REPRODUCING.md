# Reproducing the results

Complete the [quickstart](../README.md#quick-start) to install the environment,
download the main checkpoint, and configure the judge. The README lists
[checkpoint configurations](../README.md#checkpoints-and-baselines).

For example, evaluate PRISM without RL:

```bash
uv run python scripts/download_weights.py --only prism-qwen3.5-9b-sft
uv run prism-eval evaluate --config configs/main/qwen3.5-9b-sft.yaml --offline
```

The [text-only control](../configs/main/text_only_baseline.yaml) also requires
credentials for its separate `runner.baseline_llm` endpoint.

## Outputs and comparison

Each run writes `rows.jsonl` and `summary.json` under
`results/<experiment.name>/`. In `summary.json`, the paper metrics are:

| Field | Meaning |
|---|---|
| `summary.JudgeLLMScorer.reward` | Coverage adjusted for hallucination and report length |
| `summary.JudgeLLMScorer.coverage` | Mean ground-truth instruction coverage |
| `summary.JudgeLLMScorer.hallucination_rate` | Mean per-bullet hallucination score |
| `summary.AdversarialDetectionScorer.detect_rate_avg` | Fraction of scored AP/HO records with mean adversarial-instruction coverage ≥ 0.5 |

The summaries also contain per-setting aggregates. Compare them with the
[reference results](RESULTS.md), using `data/eval_suite.json` and the same
checkpoint, judge, and generation settings. Small run-to-run differences can
occur.

`--offline` disables Weave tracing and leaderboard publication, not judge
calls. To enable tracing, set `experiment.weave_project` and `WANDB_API_KEY`,
then omit the flag.

## Runtime and memory

A reference 1,000-record run on an RTX PRO 6000 with a co-located judge took
about 20 minutes. Runtime depends on the target model, batching, and judge
endpoint.

Reduce `annotation.batch_size` after a GPU out-of-memory error. Use
`runner.device` to select the inference GPU and `annotation.trace_workers`
to control concurrent judge calls.

## Ablations

The drivers vary the activation read while keeping the PRISM checkpoint fixed.
If the judge is remote, export `PRISM_EVAL_BASE_URL`, `PRISM_EVAL_MODEL`, and
`PRISM_EVAL_API_KEY` before launching them: these shell scripts set defaults
before the CLI reads `.env`.

```bash
scripts/run_ablation_window.sh
scripts/run_ablation_window_chunks.sh
scripts/run_ablation_window_chunks_deep.sh
scripts/run_ablation_context.sh
```

See the [ablation report](ABLATION_REPORT.md) for results and
[configs/template.yaml](../configs/template.yaml) for individual window and
context overrides. Set `PRISM_EVAL_BASE_RESPONSE_CACHE` to reuse generated
responses across compatible runs.

## Optional installation check

To check model loading, activation extraction, and report generation without
a judge:

```bash
uv run prism-eval evaluate --config configs/smoke.yaml --offline
```

This eight-record check does not compute the paper metrics.
