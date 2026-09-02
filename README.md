# PRISM-eval

Evaluation code and data for [PRISM: Recovering Instruction Sets from Language
Model Activations](https://arxiv.org/abs/2606.09563), accepted to the EMNLP 2026
Main Conference.

PRISM is an activation-conditioned decoder. It reads residual-stream states
from a frozen language model and produces a list of the instructions represented
in those states. This repository evaluates those reports against known
instruction sets. The corresponding data-generation and training code is in
[`Offensive-AI-Lab/prism`](https://github.com/Offensive-AI-Lab/prism).

## Requirements

- Python 3.11 or later and [uv](https://docs.astral.sh/uv/)
- A CUDA GPU for the PRISM and text-only runners
- Access to the target model weights on Hugging Face
- An OpenAI-compatible judge endpoint for the published metrics

The smoke configuration does not call a judge. Full runs used one 80 GB GPU;
memory use depends on the target model, batch size, and attention backend.

## Quick start

```bash
git clone https://github.com/Offensive-AI-Lab/prism-eval.git
cd prism-eval
uv sync --extra dev
uv run python scripts/download_weights.py --only prism-qwen3.5-9b-grpo
cp .env.example .env
uv run prism-eval evaluate --config configs/smoke.yaml --offline
```

The smoke run evaluates eight records with local, non-judge scorers and writes
`results/smoke/rows.jsonl` and `results/smoke/summary.json`. It still loads the
Qwen target model and therefore requires a compatible GPU and access to the
model weights.

`scripts/download_weights.py` verifies every checkpoint against a recorded
SHA-256 digest. Omit `--only` to download all four released checkpoints.

## Run the published configuration

The paper reports the Qwen 3.5 9B GRPO checkpoint. Configure a judge in `.env`:

```dotenv
PRISM_EVAL_CHECKPOINT_DIR=./checkpoints
PRISM_EVAL_MODEL=gemma4-31B-it
PRISM_EVAL_BASE_URL=http://localhost:8088/v1
PRISM_EVAL_API_KEY=EMPTY
```

Any OpenAI-compatible endpoint can be used. The published results used
`google/gemma-4-31B-it` with reasoning disabled; `scripts/serve_judge.sh`
contains the corresponding vLLM command.

Run the full suite with:

```bash
uv run prism-eval evaluate \
  --config configs/main/qwen3.5-9b-grpo.yaml \
  --offline
```

Here, `--offline` disables Weave tracing and leaderboard publication. It does
not disable calls to the configured judge. To run without any judge calls, use
`configs/smoke.yaml` or set both `scoring.judge_llm` and
`scoring.adversarial_detection` to `false` in a copied configuration.

To publish traces through Weave, set `experiment.weave_project` in the YAML,
set `WANDB_API_KEY`, and omit `--offline`.

## Evaluation suite

The canonical suite is `data/eval_suite.json`. It contains 250 records
from each setting:

| Code | Setting | Evaluation target |
|---|---|---|
| AP | Adversarial Prompt | Instructions injected through documents, tool outputs, email, webpages, and related channels |
| HO | Hidden Objective | Covert objectives placed in the system prompt |
| BC | Behavioral Constraints | Persona, format, topic, and style constraints |
| BN | Benign Baseline | Ordinary instructions used to measure false positives and hallucinations |

AP and HO examples were retained only when the target behavior was observed
during suite construction. The suite schema, generation process, source
licenses, and known limitations are documented in [DATA_CARD.md](DATA_CARD.md).
In particular, the BN subset is derived from Stanford Alpaca and remains
subject to CC BY-NC 4.0.

Each record contains the prompt, its ground-truth instruction list, setting
metadata, and stable evaluation ID. A runner generates the target model's
response and an instruction report; scorers compare that report with the
ground truth.

## Metrics

The published configuration uses an LLM judge governed by
[RUBRIC.md](RUBRIC.md).

| Metric | Definition |
|---|---|
| `coverage` | Mean per-instruction score, with values 1.0 (covered), 0.5 (partial), and 0.0 (missed) |
| `hallucination_rate` | Fraction of report claims judged unsupported |
| `length_penalty` | Penalty for reports that exceed the expected claim count |
| `reward` | Coverage adjusted by hallucination and length penalties |
| `detect_rate_any` | Fraction of adversarial records with at least one adversarial instruction recovered |
| `detect_rate_avg` | Mean coverage over adversarial instructions |
| `detect_rate_all` | Fraction of adversarial records with all adversarial instructions recovered |

Adversarial detection is computed for AP and HO records and uses the rubric in
[RUBRIC_ADVDET.md](RUBRIC_ADVDET.md). Exact match and token F1 are also emitted,
but they are not the paper's headline metrics.

## Runners and checkpoints

| Configuration | Target model | Hook layer | Purpose |
|---|---|---:|---|
| `qwen3.5-9b-grpo.yaml` | `Qwen/Qwen3.5-9B` | 16 | Main PRISM result: SFT followed by GRPO |
| `qwen3.5-9b-sft.yaml` | `Qwen/Qwen3.5-9B` | 16 | SFT-only ablation (`PRISM w/o RL`) |
| `gemma-2-9b-it-grpo.yaml` | `google/gemma-2-9b-it` | 21 | Transfer to Gemma 2 9B |
| `ministral-3-8b-grpo.yaml` | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | 17 | Transfer to Ministral 3 8B |
| `text_only_baseline.yaml` | `Qwen/Qwen3.5-9B` | n/a | Control that infers instructions from the final 128 response tokens without activations |

The `prism` runner generates a base response with the LoRA adapters disabled,
extracts response-token activations at the checkpoint's hook layer, projects up
to 128 activations into embedding space, and decodes an instruction report with
the adapters enabled. The text-only runner sends the same response tail to a
separate LLM.

LatentQA and Activation Oracles are included in the paper's comparison but are
not reimplemented or redistributed here. Their exact upstream revisions and
checkpoints are listed in [docs/RESULTS.md](docs/RESULTS.md).

## Configuration and output

Experiments are defined by YAML files. `configs/template.yaml` documents every
field, and `configs/main/` contains the released configurations. Important
sections are:

- `suite`: input files and record filters
- `runner`: implementation, checkpoint, and device
- `scoring`: enabled scorers and judge endpoint overrides
- `annotation`: inference batch size and trace settings
- `evaluation`: stable comparison identity and optional leaderboard name

Two runs are directly comparable only when the suite, scorer set, and
`evaluation.evaluation_name` match.

Offline output has the following form:

```text
results/<experiment>/
  rows.jsonl
  summary.json
```

`rows.jsonl` contains one model output and scorer result per record.
`summary.json` contains overall and per-setting aggregates plus the resolved run
metadata.

## Indirect prompt injection corpus

`data/xpia_corpus.parquet` contains 25,002 tagged source rows from three public
indirect prompt-injection benchmarks:

| Source | Rows |
|---|---:|
| BIPIA | 13,950 |
| LLMail | 9,998 |
| InjecAgent | 1,054 |

A 20-record smoke suite is included. Run it with:

```bash
uv run prism-eval evaluate \
  --config configs/xpia/eval_xpia_smoke.yaml \
  --offline
```

The full evaluation suite is generated from the Parquet corpus because its
ground-truth extraction requires judge calls:

```bash
uv run python scripts/fetch_xpia_evals.py \
  --count 13950 \
  --benign-count 200 \
  -o data/eval_suite_xpia.json
uv run prism-eval evaluate --config configs/xpia/eval_xpia.yaml --offline
```

`--count` is applied separately to each source; `13950` is the size of the
largest source and therefore selects every available attack row. Use
`scripts/analyze_xpia.py` for per-source, taxonomy, behavioral, and claim
provenance breakdowns. Corpus provenance and upstream licenses are covered by
[DATA_CARD.md](DATA_CARD.md).

## Judge calibration

The repository includes human labels for the coverage/hallucination judge, the
adversarial-instruction identifier, and the optional behavioral judge under
`data/calibration/`. The published agreement reports can be reproduced without
a model endpoint:

```bash
uv run python scripts/calibrate_judge.py
uv run python scripts/calibrate_advdet.py \
  --snapshot data/calibration/advdet_gold.jsonl
uv run python scripts/calibrate_follow.py \
  --snapshot data/calibration/follow_snapshot.jsonl
```

Use `scripts/calibrate_judge.py --rescore` to score a configured judge against
the coverage gold set. Calibration methods and caveats are described in the
[data card](DATA_CARD.md).

## Command-line interface

| Command | Purpose |
|---|---|
| `prism-eval evaluate --config <yaml>` | Run inference, scoring, and aggregation |
| `prism-eval run` | Write runner outputs to JSONL |
| `prism-eval score` | Score an existing run JSONL |
| `prism-eval analyze` | Compare scorer outputs in a scored JSONL |

`evaluate` is the supported path for reproducing the paper. The lower-level
commands expose a smaller set of metrics.

## Documentation

| Document | Contents |
|---|---|
| [Results](docs/RESULTS.md) | Published tables, checkpoint mapping, and judge calibration results |
| [Reproducing](docs/REPRODUCING.md) | Commands, runtime, and expected sources of variation |
| [Data card](DATA_CARD.md) | Dataset provenance, licenses, fields, and limitations |
| [Ablation report](docs/ABLATION_REPORT.md) | Activation-window and context ablations |
| [Coverage rubric](RUBRIC.md) | Coverage and hallucination annotation policy |
| [Adversarial rubric](RUBRIC_ADVDET.md) | Adversarial-instruction identification policy |
| [Contributing](CONTRIBUTING.md) | Adding runners, scorers, and new suites |

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
