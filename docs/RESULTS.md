# Published results

These are the numbers from the paper — *PRISM: Recovering Instruction Sets from
Language Model Activations* (EMNLP 2026 Main Conference; preprint [arXiv:2606.09563](https://arxiv.org/abs/2606.09563)),
Table 1. Target model is **Qwen3.5-9B** throughout; activation-conditioned
systems read layer-16 residual-stream activations from the final 128 generated
response tokens.

## Main results

Values are split means; each row is one validation-selected checkpoint.
**R** = Judge Reward (↑) · **Cvg** = Coverage Rate (↑) · **H** = Hallucination Rate (↓).

| Method | BN R | BN Cvg | BN H | BC R | BC Cvg | BC H | HO R | HO Cvg | HO H | AP R | AP Cvg | AP H | **Avg R** | **Avg Cvg** | **Avg H** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-5.5 (text only) | 0.396 | 0.919 | 0.000 | 0.360 | 0.444 | 0.000 | 0.316 | 0.461 | 0.000 | 0.373 | 0.536 | 0.000 | 0.361 | 0.590 | **0.000** |
| LatentQA (Llama3-8B) | 0.805 | 0.808 | 0.008 | 0.278 | 0.279 | 0.004 | 0.277 | 0.277 | 0.000 | 0.330 | 0.333 | 0.012 | 0.423 | 0.424 | 0.006 |
| Activation Oracles (Qwen3-8B) | 0.821 | 0.832 | 0.028 | 0.362 | 0.362 | 0.000 | 0.243 | 0.243 | 0.000 | 0.342 | 0.344 | 0.004 | 0.442 | 0.445 | 0.008 |
| LatentQA (Qwen3.5-9B) | 0.761 | 0.767 | 0.016 | 0.305 | 0.311 | 0.016 | 0.366 | 0.366 | 0.000 | 0.304 | 0.325 | 0.052 | 0.434 | 0.442 | 0.021 |
| Activation Oracles (Qwen3.5-9B) | 0.892 | 0.898 | 0.004 | 0.396 | 0.398 | 0.004 | 0.417 | 0.417 | 0.000 | 0.423 | 0.429 | 0.015 | 0.532 | 0.536 | 0.006 |
| **PRISM w/o RL** (Qwen3.5-9B) | 0.964 | 0.971 | 0.002 | 0.617 | 0.661 | 0.033 | 0.563 | 0.599 | 0.028 | 0.468 | 0.531 | 0.086 | 0.653 | 0.691 | 0.037 |
| **PRISM** (Qwen3.5-9B) | **0.967** | **0.977** | 0.002 | **0.751** | **0.761** | 0.022 | **0.641** | **0.646** | 0.008 | **0.658** | **0.685** | 0.048 | **0.754** | **0.767** | 0.020 |

BN = Benign · BC = Behavioral Constraints · HO = Hidden Objectives · AP = Adversarial Prompt.

PRISM has the highest reward and coverage in each setting. Relative to PRISM
without RL, AP coverage increases from 0.531 to 0.685 and AP hallucination
decreases from 0.086 to 0.048. Several baselines have lower hallucination rates
but also substantially lower coverage.

Adversarial-instruction detection is 0.844 on HO and 0.676 on AP, for a mean of
0.760 across the two adversarial settings.

The GPT-5.5 text-only control sees the response but no hidden states. Its benign
coverage is 0.919, while coverage is lower on the adversarial settings (0.461
for HO and 0.536 for AP).

## Mapping the paper's names onto this repo

| Paper | This repo |
|---|---|
| PRISM | `configs/main/qwen3.5-9b-grpo.yaml`, `runner.type: prism` |
| PRISM w/o RL | `configs/main/qwen3.5-9b-sft.yaml` |
| GPT-5.5 (text only) | `configs/main/text_only_baseline.yaml` |
| LatentQA, Activation Oracles | authors' own code and checkpoints — see below |
| Judge Reward (R) | `reward` in `summary.json` |
| Coverage Rate (Cvg) | `coverage` in `summary.json` |
| Hallucination Rate (H) | `hallucination_rate` in `summary.json` |
| target model | the model being monitored (`model_id` inside a checkpoint) |

The two activation-to-text baselines in Table 1 were produced by running each
method's own released code and checkpoints, unmodified, over this suite and
scoring with the same `gemma-4-31B-it` judge. Neither is re-implemented here:

- **LatentQA** (Pan et al., 2024) — [aypan17/latentqa](https://github.com/aypan17/latentqa)
  @ `a2dcb6f`, decoder [`aypan17/latentqa_llama-3-8b-instruct`](https://huggingface.co/aypan17/latentqa_llama-3-8b-instruct)
- **Activation Oracles** (Karvonen et al., 2025) —
  [adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles)
  @ `55f153f`, adapters from the
  [Activation Oracles collection](https://huggingface.co/collections/adamkarvonen/activation-oracles)

Each was run against its own target model as labelled in the table above.

## Reproducing a row

The **PRISM** and **PRISM w/o RL** rows use the released GRPO and SFT
checkpoints, respectively. The GRPO checkpoint is the default model for
evaluation; use the SFT checkpoint only to isolate the effect of RL.

```bash
python scripts/download_weights.py --only prism-qwen3.5-9b-grpo
export PRISM_EVAL_CHECKPOINT_DIR=./checkpoints
cp .env.example .env                             # judge endpoint
prism-eval evaluate --config configs/main/qwen3.5-9b-grpo.yaml --offline
```

A repeated `qwen3.5-9b-grpo` run should be close to, but may not exactly match,
the PRISM row. Generation is greedy in the released evaluator, but GPU kernels
can affect borderline tokens and judge calls can vary. See
[REPRODUCING.md](REPRODUCING.md) for expected sources of variation.

Two rows in this repo have no counterpart in the paper at all —
`configs/main/gemma-2-9b-it-grpo.yaml` and `configs/main/ministral-3-8b-grpo.yaml` apply the same
method to Gemma-2-9B and Ministral-3-8B target models. They exist to show the
approach transfers; there are no published numbers to compare them against.

## Indirect prompt injection (XPIA)

A second domain, not part of the paper's Table 1: 24,953 successfully evaluated
indirect-injection records built from BIPIA, LLMail, and InjecAgent (see
[../DATA_CARD.md](../DATA_CARD.md)). Run with `qwen3.5-9b-grpo` and scored by
the same `gemma-4-31B-it` judge.

| Source | n | Coverage | Hallucination | Full-recovery rate |
|---|---|---|---|---|
| BIPIA | 13,901 | 0.781 | 0.036 | 50.8% |
| LLMail | 9,998 | 0.772 | 0.047 | 45.5% |
| InjecAgent | 1,054 | **0.667** | 0.026 | **20.2%** |
| **Overall** | **24,953** | **0.772** | **0.040** | — |

InjecAgent coverage is approximately 0.11 lower than BIPIA and LLMail. Most
errors on that subset are partial recoveries rather than complete misses.

**Conditioned on the attack actually firing**, adversarial-instruction coverage
is 94.5% / 94.1% / 81.1% (any / avg / all, n = 9,494 rows), against 98.7% /
98.6% / 93.7% on benign instructions (n = 21,216). The attack success rate over
the corpus is 33.2%.

### Instruction presence and model behavior

The behavioral analysis indicates stronger recovery for instructions the model
followed, with weaker but nonzero recovery for instructions that were present
but not followed:

- Instructions the model followed are recovered at 0.87–0.96 mean coverage,
  against 0.48–0.54 for those it did not.
- That gap persists (0.43) within *identical* instruction text across 285
  matched templates, so it is not explained by not-followed instructions simply
  being harder.
- 46.5% of injections that the model did not act on are still fully recovered.
- Claim-side, 89.3% of the 66k claims correspond to instructions present in the
  prompt (69.4% executed, 16.1% refused-but-mentioned, 3.7% invisible in the
  response), against 3.1% pure behaviour description and 7.6% fabrication.

The first three bullets come from the calibrated FOLLOWED judge
(`prism_eval/scoring/behavior_judge.py`, judge-vs-human κ = 0.734 against a
human ceiling of 0.786 — see [../DATA_CARD.md](../DATA_CARD.md)). The
claim-side split comes from `prism_eval/scoring/claim_provenance.py`, which has
no gold set of its own and should be read as indicative.

Neither judge is part of the default scoring path, and no Table 1 number depends
on them. They are applied to saved per-record outputs after evaluation.

## Ablations

Where the signal lives in the activation window:
[ABLATION_REPORT.md](ABLATION_REPORT.md).

## Judge calibration

The scoring judge was calibrated against a **reconciled gold set** before use
(paper §G, Table 4): two annotators scored each claim independently, then
resolved every disagreement together. κ is quadratic-weighted on the ordinal
missed/partial/covered scale.

Pilot round — 49 reports / 170 gold labels, the one the paper reports:

| Comparison | κ (weighted) | Gwet's AC2 |
|---|---|---|
| **Judge vs gold** | **0.817** | 0.930 |
| Human A vs Human B (pre-reconciliation) | 0.823 | 0.939 |

The calibration gate was κ ≥ 0.70. Read the judge against the human row, not
against 1.0.

```bash
python scripts/calibrate_judge.py                 # compute agreement from the shipped labels
python scripts/calibrate_judge.py --rescore       # score your own judge
```

See [DATA_CARD.md](../DATA_CARD.md), which also covers the
adversarial-detection gold set (judge matches gold 49/50) and the FOLLOWED
judge.
