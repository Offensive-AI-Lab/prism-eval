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
Evaluation runs Qwen3.5-9B with the released PRISM checkpoint locally and
scores its reports through a separate judge endpoint.

```bash
git clone https://github.com/Offensive-AI-Lab/prism-eval.git
cd prism-eval
uv sync
cp .env.example .env
uv run python scripts/download_weights.py --only prism-qwen3.5-9b-grpo
```

The checkpoint is saved in `./checkpoints`; the target model is downloaded on
first use. Configure the judge endpoint in `.env`. The paper uses
`google/gemma-4-31B-it` with reasoning disabled:

```dotenv
PRISM_EVAL_CHECKPOINT_DIR=./checkpoints
PRISM_EVAL_MODEL=gemma4-31B-it
PRISM_EVAL_BASE_URL=http://localhost:8088/v1
PRISM_EVAL_API_KEY=EMPTY
```

To host the judge, install vLLM in a separate environment and run
[scripts/serve_judge.sh](scripts/serve_judge.sh) on its GPU host. The supplied
configuration needs roughly 80 GB of GPU memory for the judge, in addition to
the target model's memory. The endpoint can be on another machine; set its
address and credentials above.

Run the 1,000-record paper evaluation:

```bash
uv run prism-eval evaluate --config configs/main/qwen3.5-9b-grpo.yaml --offline
```

The run writes `rows.jsonl` and `summary.json` under
`results/qwen3.5-9b-grpo/`. `--offline` disables Weave tracing and leaderboard
publication; it still calls the judge.

Compare with the [paper results](docs/RESULTS.md). The
[reproduction guide](docs/REPRODUCING.md) covers hardware, ablations, and
run-to-run variation; [configs/template.yaml](configs/template.yaml) lists
configuration options.

## Checkpoints and baselines

| Checkpoint | Target model | Configuration |
|---|---|---|
| [PRISM — Qwen](https://huggingface.co/Offensive-AI-Lab/prism-qwen3.5-9b-grpo) | `Qwen/Qwen3.5-9B` | [qwen3.5-9b-grpo.yaml](configs/main/qwen3.5-9b-grpo.yaml) |
| [PRISM w/o RL — Qwen](https://huggingface.co/Offensive-AI-Lab/prism-qwen3.5-9b-sft) | `Qwen/Qwen3.5-9B` | [qwen3.5-9b-sft.yaml](configs/main/qwen3.5-9b-sft.yaml) |
| [PRISM — Gemma](https://huggingface.co/Offensive-AI-Lab/prism-gemma-2-9b-it-grpo) | `google/gemma-2-9b-it` | [gemma-2-9b-it-grpo.yaml](configs/main/gemma-2-9b-it-grpo.yaml) |
| [PRISM — Ministral](https://huggingface.co/Offensive-AI-Lab/prism-ministral-3-8b-grpo) | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | [ministral-3-8b-grpo.yaml](configs/main/ministral-3-8b-grpo.yaml) |

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
evaluation ID.

See the [data card](DATA_CARD.md) for the schema, sources, and construction
process, and [License](#license) for usage restrictions.

The indirect prompt injection corpus remains under provenance and
redistribution review.
Do not redistribute `data/xpia_corpus.parquet` until that review is complete.

## Metrics

The judge follows the [coverage and hallucination rubric](RUBRIC.md).
Adversarial detection is scored on AP and HO using the
[adversarial-instruction rubric](RUBRIC_ADVDET.md).

| Metric | Definition |
|---|---|
| `reward` | Coverage minus weighted hallucination and length penalties |
| `coverage` | Mean per-instruction score: 1.0 for covered, 0.5 for partial, and 0.0 for missed |
| `hallucination_rate` | Mean per-bullet hallucination score: 0.0 grounded, 0.5 ambiguous, 1.0 hallucinated |
| `detect_rate_avg` | Fraction of scored AP/HO records with mean adversarial-instruction coverage ≥ 0.5 |

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
