# Reproducing the results

This guide maps the released configurations to runnable commands. Reference
values are reported in [RESULTS.md](RESULTS.md).

## Prerequisites

Install the locked environment and download the checkpoint for the main result:

```bash
uv sync --extra dev
uv run python scripts/download_weights.py --only prism-qwen3.5-9b-grpo
cp .env.example .env
```

The target model is downloaded from Hugging Face on first use. Configure
`HF_HUB_CACHE` when model weights should be shared across runs or users. Set
`HF_TOKEN` for gated models.

The published metrics require an OpenAI-compatible judge endpoint. The paper
used `google/gemma-4-31B-it` with reasoning disabled. Start the supplied vLLM
server or point the environment variables at another endpoint:

```bash
scripts/serve_judge.sh
```

```dotenv
PRISM_EVAL_CHECKPOINT_DIR=./checkpoints
PRISM_EVAL_MODEL=gemma4-31B-it
PRISM_EVAL_BASE_URL=http://localhost:8088/v1
PRISM_EVAL_API_KEY=EMPTY
```

Changing the judge changes the evaluation condition. The judge model is part of
the scorer identity so that differently scored runs do not share a leaderboard
row.

## Smoke run

Use the included eight-record configuration before a full evaluation:

```bash
uv run prism-eval evaluate --config configs/smoke.yaml --offline
```

This command loads the Qwen checkpoint, generates responses, extracts
activations, decodes reports, and writes local output. Judge scoring and Weave
are disabled. Exact match and token F1 from this run are implementation checks,
not substitutes for the reported metrics.

## Main configurations

Download all four released checkpoints before running every PRISM row:

```bash
uv run python scripts/download_weights.py
```

Then execute the configurations of interest:

```bash
# Qwen 3.5 9B target model
uv run prism-eval evaluate --config configs/main/qwen3.5-9b-grpo.yaml --offline
uv run prism-eval evaluate --config configs/main/qwen3.5-9b-sft.yaml --offline
uv run prism-eval evaluate --config configs/main/text_only_baseline.yaml --offline

# Transfer configurations
uv run prism-eval evaluate --config configs/main/gemma-2-9b-it-grpo.yaml --offline
uv run prism-eval evaluate --config configs/main/ministral-3-8b-grpo.yaml --offline
```

The comparison between `qwen3.5-9b-grpo` and `qwen3.5-9b-sft` isolates the
GRPO stage: the architecture, target model, hook layer, and activation window
are the same. The Gemma and Ministral configurations change the target model and
use hook layers stored in their checkpoints.

The text-only control uses the final 128 response tokens and does not read
activations. Its `baseline_llm` call requires the provider credentials specified
by the configuration and environment.

LatentQA and Activation Oracles were evaluated through their upstream code and
are not implemented in this repository. Commit and checkpoint references are in
[RESULTS.md](RESULTS.md).

## Outputs

With `--offline`, each run writes:

```text
results/<experiment>/rows.jsonl
results/<experiment>/summary.json
```

The paper's main fields are under `summary.JudgeLLMScorer`:

| Field | Meaning |
|---|---|
| `reward` | Coverage adjusted by hallucination and report length |
| `coverage` | Mean ground-truth instruction coverage |
| `hallucination_rate` | Mean fraction of unsupported report claims |
| `length_penalty` | Mean penalty for excess report claims |
| `AP.*`, `HO.*`, `BC.*`, `BN.*` | The same metrics by setting |

`summary.AdversarialDetectionScorer` contains `detect_rate_any`,
`detect_rate_avg`, and `detect_rate_all` for AP and HO records.

To trace through Weave, set `experiment.weave_project`, set `WANDB_API_KEY`,
and omit `--offline`. The local and Weave paths use the same runner, scorers,
and aggregation functions.

## Runtime and memory

A full main-suite run processes 1,000 records. On one RTX PRO 6000 with a
co-located judge, a PRISM run took approximately 20 minutes: about 18 minutes
for inference and 2 minutes for scoring. These figures are reference
measurements, not guarantees.

The principal tuning fields are:

- `annotation.batch_size` for GPU inference
- `annotation.trace_workers` for concurrent judge calls
- `runner.device` for device placement

Lower `batch_size` after an out-of-memory error. Lower `trace_workers` when a
hosted judge enforces rate limits. Scorers within a record remain ordered
because adversarial detection depends on the coverage judge's per-claim output.

## Ablations

The activation ablations use the Qwen GRPO checkpoint and alter only the
activation read:

```bash
scripts/run_ablation_window.sh
scripts/run_ablation_window_chunks.sh
scripts/run_ablation_window_chunks_deep.sh
scripts/run_ablation_context.sh
```

Window and context settings are supplied through environment variables listed
in `configs/template.yaml`. Chunk-position runs can reuse generated responses
through `PRISM_EVAL_BASE_RESPONSE_CACHE`; use
`scripts/prefill_response_cache.py --help` to construct that cache. The
`swapped` context condition additionally requires donor pairs from
`scripts/make_act_swap_pairs.py`.

Definitions, support counts, and results are in
[ABLATION_REPORT.md](ABLATION_REPORT.md).

## Expected sources of variation

- Base-model and report generation are greedy in the released evaluator, but
  GPU architecture, attention kernels, and low-precision arithmetic can change
  borderline tokens.
- The LLM judge can vary across repeated calls. Small changes in aggregate
  coverage and hallucination are expected even with the same endpoint.
- A different judge model or prompt is a different evaluation condition and may
  produce a larger shift.
- `scripts/build_suite.py` does not regenerate the canonical suite exactly. AP
  and HO generation is sampled, and the generator revision is not pinned. Use
  `data/eval_suite_v2_final.json` when comparing with the paper.

## Compare with the reference values

Inspect the main aggregate with:

```bash
uv run python -c "import json; p='results/qwen3.5-9b-grpo/summary.json'; d=json.load(open(p)); print(json.dumps(d['summary']['JudgeLLMScorer'], indent=2))"
```

If a result differs materially from [RESULTS.md](RESULTS.md), first compare the
checkpoint digest, suite record count, judge model, enabled scorer set, target
model revision, and generation overrides. These fields account for the main
differences between otherwise similar runs.
