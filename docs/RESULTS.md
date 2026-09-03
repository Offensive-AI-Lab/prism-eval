# Published results

[PRISM](https://arxiv.org/abs/2606.09563), EMNLP 2026 Main Conference.
The main PRISM rows use Qwen3.5-9B and layer-16 activations from up to the
last 128 response tokens. Baseline target models are labelled in the table.

## Main results

Values are means over the 1,000-record suite, with 250 records per setting.
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

Adversarial-instruction detection is 0.844 on HO and 0.676 on AP, for a mean of
0.760 across the two adversarial settings.

## Upstream baselines

LatentQA and Activation Oracles were evaluated using their upstream
implementations and the same scoring judge:

- **LatentQA** (Pan et al., 2024) — [aypan17/latentqa](https://github.com/aypan17/latentqa)
  @ `a2dcb6f`, decoder [`aypan17/latentqa_llama-3-8b-instruct`](https://huggingface.co/aypan17/latentqa_llama-3-8b-instruct)
- **Activation Oracles** (Karvonen et al., 2025) —
  [adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles)
  @ `55f153f`, adapters from the
  [Activation Oracles collection](https://huggingface.co/collections/adamkarvonen/activation-oracles)

Each was run against its own target model as labelled in the table above.

## Reproduction

The PRISM and PRISM w/o RL rows use the released GRPO and SFT checkpoints.
[The reproduction guide](REPRODUCING.md) lists configurations, commands, and
output metrics.

## Indirect prompt injection benchmarks

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
(`prism_eval/scoring/behavior_judge.py`, judge-vs-human κ = 0.734 and
human-vs-human κ = 0.786 — see [../DATA_CARD.md](../DATA_CARD.md)). The
claim-side split comes from `prism_eval/scoring/claim_provenance.py`, which has
no gold set of its own and should be read as indicative.

Neither judge is part of the default scoring path, and no Table 1 number depends
on them. They are applied to saved per-record outputs after evaluation.

## Ablations

Where the signal lives in the activation window:
[ABLATION_REPORT.md](ABLATION_REPORT.md).

## Judge calibration

Calibration methods, results, and commands are in the
[data card](../DATA_CARD.md#judge-calibration-data).
