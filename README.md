# PRISM-eval

Evaluation code and data for [PRISM: Recovering Instruction Sets from Language
Model Activations](https://arxiv.org/abs/2606.09563), accepted to the EMNLP 2026
Main Conference.

PRISM reads a language model's activations and reports the instructions
represented in them. This repository evaluates those reports against known
instruction sets. To try PRISM interactively or train a model, see
[prism](https://github.com/Offensive-AI-Lab/prism#try-prism).

## Quick start

Use Python 3.11 or later, [uv](https://docs.astral.sh/uv/), and a CUDA GPU.
This eight-record smoke run loads Qwen3.5-9B and the released PRISM checkpoint;
it does not need a judge endpoint or a W&B account.

```bash
git clone https://github.com/Offensive-AI-Lab/prism-eval.git
cd prism-eval
uv sync
cp .env.example .env
uv run python scripts/download_weights.py --only prism-qwen3.5-9b-grpo
uv run prism-eval evaluate --config configs/smoke.yaml --offline
```

The target model is downloaded on first use. The checkpoint downloader verifies
its SHA-256 digest and saves it in `./checkpoints` by default. The run writes
`results/smoke/rows.jsonl` and `results/smoke/summary.json`.

## Reproduce the paper results

Full evaluation uses two model roles:

| Component | Runs where | Purpose |
|---|---|---|
| Target model + PRISM checkpoint | Local CUDA GPU | Generate a response and recover its instructions |
| LLM judge | A separate OpenAI-compatible endpoint | Score the instruction report against ground truth |

The judge can run on another machine. If you host both components locally,
budget GPU memory for both; the judge alone needs roughly 80 GB in bf16 with
the supplied serving configuration. Inference memory also depends on the target
model, prompt length, and `annotation.batch_size`. See
[Reproducing](docs/REPRODUCING.md) for runtime and memory notes.

The paper uses the released Qwen GRPO checkpoint and
`google/gemma-4-31B-it` as the judge, with reasoning disabled. Configure the
endpoint in `.env`:

```dotenv
PRISM_EVAL_CHECKPOINT_DIR=./checkpoints
PRISM_EVAL_MODEL=gemma4-31B-it
PRISM_EVAL_BASE_URL=http://localhost:8088/v1
PRISM_EVAL_API_KEY=EMPTY
```

To host the judge, install vLLM in its serving environment and run
[scripts/serve_judge.sh](scripts/serve_judge.sh) in a separate terminal.
Otherwise, use the address and credentials of an existing endpoint.

Then run the 1,000-record suite:

```bash
uv run prism-eval evaluate --config configs/main/qwen3.5-9b-grpo.yaml --offline
```

`--offline` disables Weave tracing and leaderboard publication, not judge
calls. Use `configs/smoke.yaml` to run without a judge. A different judge model
or prompt changes the evaluation condition.

The [results table](docs/RESULTS.md) reports the paper's final checkpoint
results. The [reproduction guide](docs/REPRODUCING.md) covers other configurations,
ablations, and expected run-to-run variation.

## Checkpoints and baselines

| Checkpoint | Target model | Hook layer | Configuration |
|---|---|---:|---|
| [PRISM — Qwen](https://huggingface.co/Offensive-AI-Lab/prism-qwen3.5-9b-grpo) | `Qwen/Qwen3.5-9B` | 16 | [qwen3.5-9b-grpo.yaml](configs/main/qwen3.5-9b-grpo.yaml) |
| [PRISM w/o RL — Qwen](https://huggingface.co/Offensive-AI-Lab/prism-qwen3.5-9b-sft) | `Qwen/Qwen3.5-9B` | 16 | [qwen3.5-9b-sft.yaml](configs/main/qwen3.5-9b-sft.yaml) |
| [PRISM — Gemma](https://huggingface.co/Offensive-AI-Lab/prism-gemma-2-9b-it-grpo) | `google/gemma-2-9b-it` | 21 | [gemma-2-9b-it-grpo.yaml](configs/main/gemma-2-9b-it-grpo.yaml) |
| [PRISM — Ministral](https://huggingface.co/Offensive-AI-Lab/prism-ministral-3-8b-grpo) | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | 17 | [ministral-3-8b-grpo.yaml](configs/main/ministral-3-8b-grpo.yaml) |

Omit `--only` from the download command to fetch all four checkpoints. Target
model weights are downloaded separately; set `HF_TOKEN` for gated models.

The [text-only control](configs/main/text_only_baseline.yaml) reads the final
128 response tokens without activations and calls a separate language-model
endpoint. LatentQA and Activation Oracles were evaluated using their upstream
implementations; code revisions and checkpoint links are listed in
[Results](docs/RESULTS.md).

## Evaluation data

The main suite, `data/eval_suite.json`, has 1,000 records, with 250 per setting:

| Code | Setting | What it measures |
|---|---|---|
| AP | Adversarial Prompt | Recovery of instructions injected through documents, tool outputs, email, and other channels |
| HO | Hidden Objective | Recovery of covert objectives in the system prompt |
| BC | Behavioral Constraints | Recovery of persona, format, topic, and style constraints |
| BN | Benign Baseline | Recovery of ordinary instructions and hallucination on benign prompts |

Each record includes a prompt, a ground-truth instruction list, and a stable
evaluation ID. AP and HO examples were retained only when the intended target
behavior was observed during suite construction.

The BN records derive from Stanford Alpaca and retain its CC BY-NC 4.0
non-commercial restriction. See the [data card](DATA_CARD.md) for the schema,
source licenses, and construction process.

The separate XPIA corpus remains under provenance and redistribution review.
Do not redistribute `data/xpia_corpus.parquet` until that review is complete.

## Metrics

The judge follows the [coverage and hallucination rubric](RUBRIC.md).
Adversarial detection is scored on AP and HO using the
[adversarial-instruction rubric](RUBRIC_ADVDET.md).

| Metric | Definition |
|---|---|
| `coverage` | Mean per-instruction score: 1.0 for covered, 0.5 for partial, and 0.0 for missed |
| `hallucination_rate` | Fraction of report claims judged unsupported |
| `length_penalty` | Penalty for reports exceeding the allowed claim count |
| `reward` | Coverage minus hallucination and length penalties |
| `detect_rate_any` | Fraction of adversarial records with at least one adversarial instruction recovered |
| `detect_rate_avg` | Mean coverage over adversarial instructions |
| `detect_rate_all` | Fraction of adversarial records with all adversarial instructions recovered |

## Configuration and output

[configs/template.yaml](configs/template.yaml) documents the configuration
fields. Use `prism-eval evaluate --config <yaml>` for inference, scoring, and
aggregation. The lower-level `run`, `score`, and `analyze` commands are
described by `uv run prism-eval --help`.

Each run writes `results/<experiment>/rows.jsonl` with per-record outputs and
scores, and `summary.json` with overall and per-setting aggregates. To enable
Weave tracing, set `experiment.weave_project` and `WANDB_API_KEY`, then omit
`--offline`.

## Judge calibration

Human labels and saved agreement reports are in `data/calibration/`.
Recompute agreement from the shipped labels without a model endpoint:

```bash
uv run python scripts/calibrate_judge.py
uv run python scripts/calibrate_advdet.py --snapshot data/calibration/advdet_gold.jsonl
uv run python scripts/calibrate_follow.py --snapshot data/calibration/follow_snapshot.jsonl
```

Use `scripts/calibrate_judge.py --rescore` to evaluate another judge against
the coverage gold set. The [data card](DATA_CARD.md#judge-calibration-data)
describes the annotation procedure.

## Further reading

- [Results](docs/RESULTS.md): paper tables and baseline references.
- [Reproducing](docs/REPRODUCING.md): commands, hardware, and run-to-run variation.
- [Data card](DATA_CARD.md): sources, schema, annotation, and limitations.
- [Ablations](docs/ABLATION_REPORT.md): activation-window and context experiments.
- [Contributing](CONTRIBUTING.md): adding runners, scorers, and suites.

## Citation

```bibtex
@inproceedings{gressel2026prism,
  title     = {PRISM: Recovering Instruction Sets from Language Model Activations},
  author    = {Gressel, Gilad and Pankajakshan, Rahul and Diament, Julia and
               Hudis, Efim and Achuthan, Krishnashree and Mirsky, Yisroel},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.09563}
}
```

## License

The code and project-authored AP/HO records are licensed under the
[Apache License 2.0](LICENSE). Third-party datasets retain their original
licenses. The BN records are derived from Stanford Alpaca and are restricted to
non-commercial use under CC BY-NC 4.0; see [DATA_CARD.md](DATA_CARD.md) and
[NOTICE](NOTICE).
