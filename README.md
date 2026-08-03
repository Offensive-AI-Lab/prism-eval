# PRISM-eval

Evaluation harness for **PRISM** — *Recovering Instruction Sets from Language
Model Activations* ([arXiv:2606.09563](https://arxiv.org/abs/2606.09563)).

When an LLM acts as an agent, monitoring it means knowing not just what it
output but **which instructions are steering it** — including subgoals it
inferred, contextual cues it picked up, injections it swallowed, and objectives
it was told to hide. The paper formalises this as **instruction set retrieval**.

PRISM is an **activation-conditioned interpreter**: it decodes residual-stream
hidden states from a *frozen* target model into a bullet list of the
instructions currently active. It is trained with **judge-guided GRPO** —
rewarding covered instructions and penalising unsupported ones. A trained
checkpoint is just a linear projection plus LoRA adapters; the target model
itself is never modified.

This repository provides the evaluation suite (1000 records), the scorers, the
trained checkpoints, and configurations for five comparable rows, including the
same monitor trained against three different target models.

**`prism-qwen3.5-9b-grpo` is the model to use.** It is the one the paper reports
and the strongest of the four; the other three exist to isolate the RL stage and
to show the method transfers to other target models. Unless you have a specific
reason to want one of those, download only this one:

```bash
uv sync --extra dev
python scripts/download_weights.py --only prism-qwen3.5-9b-grpo   # 266 MB
export PRISM_EVAL_CHECKPOINT_DIR=./checkpoints

# Smoke run: 8 records, no judge, no W&B. Confirms the whole path works.
prism-eval evaluate --config configs/smoke.yaml --offline
```

(Omit `--only` to fetch all four, ~1.4 GB.)

That requires a GPU and nothing else. The headline metrics additionally require
a judge — an LLM that scores each report against the ground-truth instructions.
The published results use `gemma-4-31B-it`; `./scripts/serve_judge.sh` starts
that model under vLLM, and the `PRISM_EVAL_*` variables accept any
OpenAI-compatible endpoint. Then:

```bash
cp .env.example .env      # fill in PRISM_EVAL_MODEL / _BASE_URL / _API_KEY
prism-eval evaluate --config configs/main/qwen3.5-9b-grpo.yaml --offline
```

No W&B account needed either way. Results land in
`results/<experiment>/summary.json`.

---

## What is being measured

Each record is a prompt with known ground-truth instructions. A **runner**
produces an *ITM report* — a bulleted list of the instructions it believes the
model was given — and scorers compare that report to ground truth.

| Code | Setting | n | What it tests |
|---|---|---|---|
| **AP** | Adversarial Prompt | 250 | An injected instruction hidden in a document, tool response, email, web page … Did the monitor surface it? |
| **HO** | Hidden Objective | 250 | A covert objective in the system prompt (steer to a product, fabricate stats) that the response deliberately conceals. |
| **BC** | Behavioral Constraints | 250 | Persona, formatting and topic rules of the kind found in ordinary system prompts. |
| **BN** | Benign Baseline | 250 | Plain instructions with no adversarial content. Establishes the false-positive rate. |

AP and HO records are **verified**: a scenario is only included if the
injection actually landed or the hidden objective was actually expressed. See
[DATA_CARD.md](DATA_CARD.md) for provenance and licences.

### A second domain: indirect prompt injection

The four settings above are single-turn prompts. A separate suite tests the
same capability in an **agentic** setting, where the attacker never talks to the
model directly.

`data/xpia_corpus.parquet` holds **25,002 tagged rows** — 24,802 attack
and 200 benign — drawn from three public indirect-prompt-injection (XPIA)
benchmarks:

| Source | Rows | Attack surface |
|---|---|---|
| BIPIA | 13,950 | Injected text inside emails, tables, code and QA context |
| LLMail | 9,998 | Injected instructions inside email an assistant is asked to process |
| InjecAgent | 1,054 | Injected instructions inside tool-call results |

Each row carries the attack taxonomy from the source dataset — `attacker_goal`,
`delivery_technique`, `evasion_technique`, `injection_position`,
`tool_output_type` — so results can be sliced by how the attack was delivered
rather than only by whether it was caught.

`scripts/fetch_xpia_evals.py` reconstructs each row into a real multi-message
conversation, so the injected span arrives in a `tool` or document turn,
separated from the legitimate user task, and is rendered through the target
model's chat template at run time. This matters: the monitor has to distinguish
an instruction the *user* gave from one that arrived inside data the agent was
asked to read. Records are built as `setting: AP`, category
`Indirect Prompt Injection`, and scored by the same judge and
adversarial-detection rubric as the main suite.

```bash
# 20-record smoke suite ships ready to run
prism-eval evaluate --config configs/xpia/eval_xpia_smoke.yaml --offline

# the full suite is built from the parquet (needs a judge endpoint to
# extract the ground-truth instruction list per row)
python scripts/fetch_xpia_evals.py -o data/eval_suite_xpia.json
prism-eval evaluate --config configs/xpia/eval_xpia.yaml --offline
```

`scripts/analyze_xpia.py` produces the breakdowns behind the XPIA results —
per-source and per-stratum coverage, adversarial detection, and optionally the
behavioural analysis (`--with-behavior`, `--with-provenance`, one judge call per
record each, cached):

```bash
python scripts/analyze_xpia.py \
    --rows results/xpia/rows.jsonl \
    --suite data/eval_suite_xpia.json \
    -o results/xpia/analysis.json
```

Only the 20-record smoke suite and the corpus ship; the full suite is a build
artifact because reconstructing it calls an LLM per row. Provenance and the
upstream licences are in [DATA_CARD.md](DATA_CARD.md); published results for the
25k run, including how much of the monitor's output reflects what the model was
*told* rather than what it *did*, are in [docs/RESULTS.md](docs/RESULTS.md).

## Metrics

Reports are scored by an LLM judge — **`gemma-4-31B-it`** in all published
results — against the rubric in [RUBRIC.md](RUBRIC.md):

- **coverage** — fraction of ground-truth instructions recovered (1.0 / 0.5 / 0.0 per claim). Coverage Rate (Cvg) in the paper.
- **hallucination_rate** — fabricated claims. Lower is better.
- **length_penalty** — penalises padding the report to capture claims by volume
- **reward** — the scalar combining the three, and the GRPO training signal
- **adversarial detection** (`detect_rate_any` / `detect_rate_avg` / `detect_rate_all`)
  — on AP and HO only. See [RUBRIC_ADVDET.md](RUBRIC_ADVDET.md).

Coverage weighs every ground-truth instruction equally, which on AP and HO
conflates two different questions: did the monitor recover the *legitimate task*,
and did it catch the *attack*? A report that fluently summarises the surface task
while missing the injection still scores around 0.83 coverage. **Adversarial
detection separates them**: a second judge names which ground-truth bullets carry
the adversarial intent, and the detection score is the coverage restricted to
those bullets. It reuses the per-bullet scores the calibrated judge already
produced, so it costs one extra call per AP/HO row and never touches the
calibrated judge itself.

All are reported overall and per setting.

### Is the judge trustworthy?

All three judges were calibrated against human annotators before use, and
**the labels ship in `data/calibration/`** — per-claim, not just summary
statistics. No Weave, no W&B account, no network.

The **scoring judge** produces the coverage and hallucination numbers. Two
annotators scored every claim independently, then reconciled each disagreement
into a single **gold** label. On the pilot round the paper reports (§G, Table 4),
49 reports / 170 gold labels: judge-vs-gold **κ = 0.800**, against a
pre-reconciliation human ceiling of 0.824 and a calibration gate of κ ≥ 0.70.

Every record ships with its prompt, the target model's response and the report,
so the same benchmark points at your own judge:

```bash
python scripts/calibrate_judge.py                 # reproduce the published table
python scripts/calibrate_judge.py --rescore       # score your judge against gold
```

A judge should be read against the human row rather than against 1.0: two
trained annotators agree only to 0.824 with each other. κ is quadratic-weighted
on the ordinal missed/partial/covered scale; the unweighted value on the same
labels is 0.60. Both are reported throughout.

The **adversarial-detection judge** has its own 50-report gold set and matches
it on 49/50.

The **FOLLOWED judge** (`prism_eval/scoring/behavior_judge.py`) labels whether
the model acted on an instruction, and is **calibrated**: 184 records, 92
double-annotated, human-vs-human κ = 0.786, judge-vs-human κ = 0.734. The
prompt shipped here is the one those numbers were measured against. It powers
the behavioural breakdown of the XPIA suite in
[docs/RESULTS.md](docs/RESULTS.md) rather than the headline metrics, so no
Table 1 number depends on it.

`prism_eval/scoring/claim_provenance.py` classifies each claim by grounding
source. It has no gold set of its own, so its output is indicative rather than
calibrated.

`tests/test_coverage_goldset.py` and `tests/test_calibration_goldset.py`
recompute all of it from the raw labels on every test run. One caveat is
documented in [DATA_CARD.md](DATA_CARD.md): the paper prints judge-vs-gold as
0.817, and no surviving judge-scoring run reproduces that exact value — the gold
labels are identical across runs, the judge is sampled.

## Runners

| Runner | Approach | Checkpoint |
|---|---|---|
| `prism` | Hooks a mid layer of the target model, projects the last 128 response-token activations into embedding space, prepends them as soft tokens, and decodes with LoRA. This is the method the paper describes. | **`prism-qwen3.5-9b-grpo.pt`** (published), plus `prism-qwen3.5-9b-sft.pt`, `prism-gemma-2-9b-it-grpo.pt`, `prism-ministral-3-8b-grpo.pt` |
| `text_only_baseline` | Reads no activations. Shows a frontier LLM the last 128 tokens of the response text and asks it to infer the instructions. This is the control condition: it bounds how much of the task is solvable from the output text alone. | none |

### The five rows

Each row is one config in `configs/main/`. The target model is the model being
monitored; PRISM itself is the projection and LoRA adapters read off it.

| Config | Target model | Hook layer | Role |
|---|---|---|---|
| **`qwen3.5-9b-grpo`** | `Qwen/Qwen3.5-9B` | 16 | **The published model.** SFT followed by judge-guided GRPO, and the row the paper reports. Use this one. |
| `qwen3.5-9b-sft` | `Qwen/Qwen3.5-9B` | 16 | Same architecture without the RL stage — "PRISM w/o RL" in the paper. |
| `gemma-2-9b-it-grpo` | `google/gemma-2-9b-it` | 21 | Whether the method transfers to a second target model. |
| `ministral-3-8b-grpo` | `mistralai/Ministral-3-8B-Instruct` | 17 | The same question for a third architecture. |
| `text_only_baseline` | `Qwen/Qwen3.5-9B` | — | Reads no activations, only the response text. Control condition. |

All five share one suite, scorer set and `evaluation_name`, so they land on a
single leaderboard as directly comparable rows. The target model is named in
each row label because it changes the difficulty, not just the model under test.

### Baselines from prior work

The paper also compares against two published activation-to-text methods,
**LatentQA** (Pan et al., 2024) and **Activation Oracles** (Karvonen et al.,
2025). Those rows were produced by running each method's **own released code
and checkpoints**, unmodified, over the same 1000-record suite and scoring the
resulting reports with the same `gemma-4-31B-it` judge. No runner for either
method is included here, and their weights are not redistributed — use the
authors' repositories:

| Method | Code | Checkpoints |
|---|---|---|
| LatentQA | [aypan17/latentqa](https://github.com/aypan17/latentqa) @ `a2dcb6f` | [`aypan17/latentqa_llama-3-8b-instruct`](https://huggingface.co/aypan17/latentqa_llama-3-8b-instruct) |
| Activation Oracles | [adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles) @ `55f153f` | [Activation Oracles collection](https://huggingface.co/collections/adamkarvonen/activation-oracles), e.g. `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` |

Both methods require their authors' inference paths — LatentQA substitutes
target-model hidden states into its decoder, Activation Oracles applies a LoRA
adapter through forward hooks — so re-implementing them here would risk
misreporting their results. The published numbers for these rows are in
[docs/RESULTS.md](docs/RESULTS.md).

Runners implement the `ITMRunner` protocol in `prism_eval/runner.py`. Adding one
means writing the class, registering it in `prism_eval/runners/__init__.py`, and
adding its name to the `Literal` on `RunnerConfig.type` in
`prism_eval/config.py` — miss the last step and every config using it fails
validation. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Install

```bash
uv sync --extra dev
source .venv/bin/activate
pytest tests/ -q          # ~218 tests; no GPU or network needed
```

Use `uv sync`, not `uv pip install -e .` — the lockfile pins torch's cu128
nvidia deps, and `uv pip install` has produced torn installs in practice.

## Configure

Copy `.env.example` to `.env` and fill in:

```bash
PRISM_EVAL_CHECKPOINT_DIR=./checkpoints   # where download_weights.py put them
PRISM_EVAL_MODEL=gemma4-31B-it            # judge model
PRISM_EVAL_BASE_URL=http://localhost:8088/v1
PRISM_EVAL_API_KEY=EMPTY
```

**The judge** produces the headline metrics. Any OpenAI-compatible endpoint
works — a hosted API or a local server. To reproduce the published judge
exactly, `./scripts/serve_judge.sh` starts it under vLLM. The exact prompts are
checked in at `prism_eval/prompts/judge/`.

The judge model name is part of the scorer's identity, so runs scored by
different judges don't silently end up in the same table.

## Run

```bash
# Offline — no W&B account. Writes results/<experiment>/{rows.jsonl,summary.json}.
prism-eval evaluate --config configs/main/qwen3.5-9b-grpo.yaml --offline

# Traced — set experiment.weave_project in the config, then drop --offline.
# Emits per-row Weave traces and publishes a Leaderboard.
prism-eval evaluate --config configs/main/qwen3.5-9b-grpo.yaml
```

Both modes use the same runner, the same scorer objects and the same
aggregation, so the numbers match.

Full command-per-result mapping: [docs/REPRODUCING.md](docs/REPRODUCING.md).

## Ablations

Where does the instruction signal actually live in the activations?

```bash
./scripts/run_ablation_window.sh              # how many tokens does the monitor need?
./scripts/run_ablation_window_chunks.sh       # where in the response does signal live?
./scripts/run_ablation_context.sh             # is it just an echo of the prompt?
```

Findings and full tables: [docs/ABLATION_REPORT.md](docs/ABLATION_REPORT.md).
Short version: coverage climbs monotonically with window size with no plateau by
128 tokens; signal does *not* decay through the response once you condition on
support; and hiding the prompt at extraction time cuts coverage roughly in half —
so the monitor reads more than a prompt echo, but hidden objectives in
particular are carried by prompt-attending features.

## Commands

| Command | Use |
|---|---|
| `prism-eval evaluate --config <yaml>` | The entry point. Runs, scores, aggregates. Add `--offline` to skip Weave. |
| `prism-eval run` | Lower level: runner → `EvalResult` JSONL. |
| `prism-eval score` | Lower level: run JSONL → `ScoredResult` JSONL. |
| `prism-eval analyze` | Per-eval metrics and scorer comparison from a scored JSONL. |

`run`/`score`/`analyze` predate `evaluate` and compute a narrower metric set —
no adversarial detection, no reward, no per-setting breakdown. Prefer
`evaluate --offline` unless you specifically want the JSONL pipeline.

## Layout

```
configs/
  main/          the five comparable rows
  ablation/      window/ (24) and context/ (5)
  template.yaml  annotated reference for every config field
data/            eval_suite_v2_final.json (1000) + the four per-setting files
  xpia_corpus.parquet  25k agent-injection rows (2nd domain)
  calibration/   gold labels + per-annotator labels for all three judges,
                 frozen agreement reports, and the originals behind the paper
prism_eval/
  cli.py         Click CLI
  config.py      YAML → ExperimentConfig; checkpoint path resolution
  weave_eval.py  scorers, leaderboard, and the offline evaluator
  runners/       prism, text_only_baseline
  scoring/       exact_match, token_f1, bertscore, judge_llm, adversarial_identifier
  prompts/judge/ the exact published judge prompts (+ tuning variants)
scripts/         download_weights, serve_judge, build_suite, calibrate_judge,
                 analyze_xpia, strip_checkpoint, ablation drivers
docs/            RESULTS.md, REPRODUCING.md, ABLATION_REPORT.md,
                 ANNOTATION_{FOLLOW,RECALL}.md
```

## Documentation

| Document | Contents |
|---|---|
| [DATA_CARD.md](DATA_CARD.md) | Suite provenance, licences, limitations, and the judge-calibration data. **Read before redistributing** — the BN arm is CC BY-NC. |
| [docs/ANNOTATION_FOLLOW.md](docs/ANNOTATION_FOLLOW.md) | Instructions the human annotators worked from. |
| [RUBRIC.md](RUBRIC.md) | Judge scoring rubric. |
| [RUBRIC_ADVDET.md](RUBRIC_ADVDET.md) | Adversarial-detection rubric. |
| [docs/RESULTS.md](docs/RESULTS.md) | The published numbers for all seven rows. |
| [docs/REPRODUCING.md](docs/REPRODUCING.md) | The command behind each one; what does and doesn't reproduce exactly. |
| [docs/ABLATION_REPORT.md](docs/ABLATION_REPORT.md) | Where the signal lives. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Adding a runner or scorer. |

## Licence and citation

Code and the AP/HO records: Apache-2.0 ([LICENSE](LICENSE)).

**The BN records are CC BY-NC 4.0** (derived from Stanford Alpaca) — that
restriction travels with them regardless of this repo's licence. See
[DATA_CARD.md](DATA_CARD.md).

```bibtex
@article{gressel2026prism,
  title   = {PRISM: Recovering Instruction Sets from Language Model Activations},
  author  = {Gilad Gressel and Rahul Pankajakshan and Julia Diament and Efim Hudis and Krishnashree Achuthan and Yisroel Mirsky},
  journal = {arXiv preprint arXiv:2606.09563},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.09563}
}
```
