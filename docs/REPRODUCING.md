# Reproducing the results

Complete the [quickstart](../README.md#quick-start) to install the environment,
download the main checkpoint, and configure the judge. The target model and
PRISM checkpoint run on your CUDA GPU; the judge can run on a separate host.
Paper scoring uses `google/gemma-4-31B-it` with reasoning disabled.

## Configurations

Download the other checkpoints with `uv run python scripts/download_weights.py`.
Then select a configuration:

| System | Configuration |
|---|---|
| PRISM, Qwen3.5-9B | [qwen3.5-9b-grpo.yaml](../configs/main/qwen3.5-9b-grpo.yaml) |
| PRISM without RL, Qwen3.5-9B | [qwen3.5-9b-sft.yaml](../configs/main/qwen3.5-9b-sft.yaml) |
| Text-only control | [text_only_baseline.yaml](../configs/main/text_only_baseline.yaml) |
| PRISM, Gemma-2-9B-it | [gemma-2-9b-it-grpo.yaml](../configs/main/gemma-2-9b-it-grpo.yaml) |
| PRISM, Ministral-3-8B | [ministral-3-8b-grpo.yaml](../configs/main/ministral-3-8b-grpo.yaml) |

For example, evaluate the model without RL:

```bash
uv run prism-eval evaluate --config configs/main/qwen3.5-9b-sft.yaml --offline
```

The text-only control generates a Qwen response locally, then sends its last
128 tokens to `runner.baseline_llm`. This is a separate call from judge scoring
and requires the provider credentials specified in its configuration.
Set `HF_TOKEN` for gated target models.

LatentQA and Activation Oracles use upstream implementations rather than this
repository's runners. Their revisions and checkpoint links are in
[Results](RESULTS.md#upstream-baselines).

## Outputs and comparison

Each run writes `rows.jsonl` and `summary.json` under
`results/<experiment.name>/`. In `summary.json`, the paper metrics are:

| Field | Meaning |
|---|---|
| `summary.JudgeLLMScorer.reward` | Coverage adjusted for hallucination and report length |
| `summary.JudgeLLMScorer.coverage` | Mean ground-truth instruction coverage |
| `summary.JudgeLLMScorer.hallucination_rate` | Mean unsupported-claim score |
| `summary.AdversarialDetectionScorer.detect_rate_avg` | Mean coverage of adversarial instructions on AP and HO |

The scorer summaries also contain per-setting aggregates. Compare them with the
[reference results](RESULTS.md). Use the canonical `data/eval_suite.json`;
regenerating a suite or changing the judge defines a different evaluation.

Generation is greedy, but GPU kernels and low-precision arithmetic can affect
borderline tokens; repeated judge calls can also vary. For a substantial
difference, compare the checkpoint, suite, judge, scorer settings, target-model
revision, and generation overrides.

`--offline` disables Weave tracing and leaderboard publication, not judge
calls. To enable tracing, set `experiment.weave_project` and `WANDB_API_KEY`,
then omit the flag.

## Runtime and memory

A reference 1,000-record run on an RTX PRO 6000 with a co-located judge took
about 20 minutes: 18 for inference and 2 for scoring. Runtime depends on the
target model, batching, and judge endpoint.

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

The drivers include configurations beyond those tabulated in
[the ablation report](ABLATION_REPORT.md). Set
`PRISM_EVAL_BASE_RESPONSE_CACHE` to reuse generated responses across compatible
runs. The context driver generates donor pairs for its `swapped` condition.

[configs/template.yaml](../configs/template.yaml) describes window and context
overrides for individual runs. `scripts/prefill_response_cache.py --help`
describes cache construction from existing Weave traces.

## Optional installation check

To check model loading, activation extraction, and report generation without
a judge:

```bash
uv run prism-eval evaluate --config configs/smoke.yaml --offline
```

This eight-record check does not compute the paper metrics.
