# Reproducing the results

This page maps each published result to the exact command that produces it.
The results themselves are in [RESULTS.md](RESULTS.md); this is the how, that
is the what.

## Prerequisites

1. **Install** — `uv sync --extra dev` (see the README; use `uv sync`, not
   `uv pip install -e`, so the pinned cu128 torch stack resolves correctly).
2. **Weights** — `python scripts/download_weights.py --only prism-qwen3.5-9b-grpo`
   (the published model; omit `--only` for all four), then
   `export PRISM_EVAL_CHECKPOINT_DIR=...`.
3. **Base models** — each row's target model is pulled from Hugging Face on
   first run (Qwen3.5-9B, Gemma-2-9B, Ministral-3-8B; ~17-18 GB each). Set
   `HF_HUB_CACHE` to shared storage if several people run this, or you will
   download each one per user.
4. **Judge** — the headline metrics are judge-scored. Either serve the same
   judge (`./scripts/serve_judge.sh`) or point `PRISM_EVAL_BASE_URL` at any
   OpenAI-compatible endpoint. See "Judge sensitivity" below.
5. **GPU** — one 80 GB card runs the prism and baseline runners.

`--offline` needs no W&B account and writes
`results/<experiment>/{rows.jsonl,summary.json}`. Drop it (and set
`experiment.weave_project`) to trace to Weave and publish a leaderboard.

### Runtime and tuning

A full 1000-record run has two phases:

| Phase | Bound by | Knob | Measured (1000 records) |
|---|---|---|---|
| Inference | GPU | `annotation.batch_size` (16 in the shipped configs) | ~18 min |
| Scoring | Judge latency × 2 calls per row | `annotation.trace_workers` (16) | ~2 min |

Measured on one RTX PRO 6000 (97 GB) with a co-located vLLM judge — about
20 minutes end to end for `prism` on the full suite.

Scoring rows fan out across `trace_workers` threads, because one judge call is
~1 s and doing 2000 of them serially would dominate the run. Raise it if your
judge endpoint has headroom; lower it if you are rate-limited by a hosted API.
Scorers *within* a row always run in order regardless — the adversarial scorer
depends on the calibrated judge's output for the same row.

## Main comparison

Five rows, one suite, one scorer set, one `evaluation_name` — so every row is
directly comparable.

```bash
# Qwen3.5-9B target model
prism-eval evaluate --config configs/main/qwen3.5-9b-grpo.yaml               --offline  # PRISM, RL-tuned
prism-eval evaluate --config configs/main/qwen3.5-9b-sft.yaml                --offline  # PRISM, SFT only
prism-eval evaluate --config configs/main/text_only_baseline.yaml --offline  # control

# Other target models — same method, different model being monitored
prism-eval evaluate --config configs/main/gemma-2-9b-it-grpo.yaml             --offline  # Gemma-2-9B
prism-eval evaluate --config configs/main/ministral-3-8b-grpo.yaml         --offline  # Ministral-3-8B
```

`grpo` vs `sft` is the same architecture and the same activation window — only
the training recipe differs, so the gap between those two rows isolates what
the RL stage bought.

`gemma2` and `ministral3` swap the *target model* — the model being monitored —
keeping the method fixed. Each reads its own hook layer (21 and 17 vs 16), which
comes from the checkpoint, not the config. They pull their own base models from
Hugging Face on first run (`google/gemma-2-9b-it`,
`mistralai/Ministral-3-8B-Instruct-2512-BF16`), so budget the download.

The control reads only the last 128 tokens of the response text — no
activations. If an activation method does not beat it, the activations are not
contributing beyond what is already legible in the response.

Headline fields in each `summary.json`, under `summary.JudgeLLMScorer`:

| Field | Meaning |
|---|---|
| `reward` | Headline scalar; recall penalised by hallucination and length. |
| `coverage` | Fraction of ground-truth instructions recovered (1.0 / 0.5 / 0.0 per claim). Coverage Rate (Cvg) in the paper. |
| `hallucination_rate` | Mean hallucination score. Lower is better. Hallucination Rate (H) in the paper. |
| `length_penalty` | Penalty for padding the report to catch claims by volume. |
| `AP.* / HO.* / BC.* / BN.*` | The same four, per setting. |

And under `summary.AdversarialDetectionScorer` (AP and HO only): `detect_rate_any`, 
`detect_rate_avg`, `detect_rate_all`.

## Ablations

All three use the GRPO checkpoint and vary only how the monitor reads
activations. They are driven by env vars rather than separate checkpoints —
see the bottom of `configs/template.yaml` for the full list.

```bash
./scripts/run_ablation_window.sh              # A: window size, last k ∈ {16,32,64,128}
./scripts/run_ablation_window_chunks.sh       # B: position, chunks 0-5
./scripts/run_ablation_window_chunks_deep.sh  # B: position, chunks 6-15
./scripts/run_ablation_context.sh             # C: what the extraction pass may attend to
```

Results and interpretation: [ABLATION_REPORT.md](ABLATION_REPORT.md).

Ablation B tiles the *whole* response in 128-token chunks, so it needs a higher
generation cap than the published setting. Chunk runs share base responses
through an on-disk cache keyed on (base model, messages, generation cap), so
only the reading pass varies:

```bash
python scripts/prefill_response_cache.py --help   # populate the cache first
```

Ablation C's `swapped` mode needs donor-prompt pairs:

```bash
python scripts/make_act_swap_pairs.py --help
```

## Quick smoke run

Before committing a GPU for hours, confirm the whole path works on 8 records
with the judge off:

```bash
cp configs/main/qwen3.5-9b-grpo.yaml /tmp/smoke.yaml
# then edit /tmp/smoke.yaml:
#   suite.per_setting_limit: {AP: 2, HO: 2, BC: 2, BN: 2}
#   scoring.judge_llm: false
#   scoring.adversarial_detection: false
prism-eval evaluate --config /tmp/smoke.yaml --offline
```

## What will and will not reproduce exactly

**Deterministic given the same weights and base model.** Base-model responses
are generated greedily, and the activation read is deterministic. Two runs on
the same hardware give the same `itm_report` per record.

**Judge scores vary slightly.** The judge is an LLM. Expect small movement in
`coverage` / `hallucination_rate` between runs, and larger movement if you swap judge
models. The judge model name is part of the scorer identity precisely so that
runs scored by different judges do not silently land in one table.

**Hardware can move the last digits.** Different GPU architectures, batch
sizes, or attention kernels can perturb bf16 numerics enough to change a few
borderline generations. Aggregate metrics over 1000 records are stable to
roughly ±0.01; individual records may differ.

**`scripts/build_suite.py` does not reproduce the suite.** AP and HO are
LLM-generated and the generator model is not pinned. The shipped JSON is the
artifact of record — see [DATA_CARD.md](../DATA_CARD.md).

## Comparing your run to ours

The published values are in [RESULTS.md](RESULTS.md). `summary.json` records the
runner, the unexpanded checkpoint identity, the scorer set and the record count
alongside the metrics, so two runs can be diffed directly:

```bash
python - <<'PY'
import json
a = json.load(open("results/qwen3.5-9b-grpo/summary.json"))
print(json.dumps(a["summary"]["JudgeLLMScorer"], indent=2))
PY
```

If your numbers differ materially, check in this order: the judge model, the
scorer set (`adversarial_detection` on?), the record count, and the checkpoint
identity — those four explain nearly every discrepancy.
