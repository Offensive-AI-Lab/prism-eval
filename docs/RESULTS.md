# Results

[PRISM](https://arxiv.org/abs/2606.09563), EMNLP 2026 Main Conference.
The main PRISM rows use Qwen3.5-9B and layer-16 activations from up to the
last 128 response tokens. The main benchmark and indirect prompt injection
sections report results from the paper. The behavior analysis is a post-paper
extension and is marked separately below.

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

## Baselines

All baseline outputs were evaluated with the same scoring judge as PRISM. The
Qwen3.5-9B rows use the exact released adapters:

- [LatentQA adapter](https://huggingface.co/Offensive-AI-Lab/prism-baseline-latentqa-qwen3.5-9b)
- [Activation Oracles adapter](https://huggingface.co/Offensive-AI-Lab/prism-baseline-activation-oracles-qwen3.5-9b)

The implementations are adapted from [aypan17/latentqa](https://github.com/aypan17/latentqa)
at `a2dcb6f` and
[adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles)
at `55f153f`. The original-model rows use the upstream published artifacts:

- **LatentQA (Llama3-8B):** [`aypan17/latentqa_llama-3-8b-instruct`](https://huggingface.co/aypan17/latentqa_llama-3-8b-instruct)
- **Activation Oracles (Qwen3-8B):** adapters from the [Activation Oracles collection](https://huggingface.co/collections/adamkarvonen/activation-oracles)

## Reproduction

The PRISM and PRISM w/o RL rows use the released GRPO and SFT checkpoints.
[The reproduction guide](REPRODUCING.md) lists configurations, commands, and
output metrics.

## Indirect prompt injection benchmarks

The paper also evaluates 24,953 successfully processed indirect prompt
injection records built from BIPIA, LLMail-Inject, and InjecAgent (see
[../DATA_CARD.md](../DATA_CARD.md)). These results use `qwen3.5-9b-grpo` and
the same `gemma-4-31B-it` scoring judge.

| Source | n | Reward | Coverage | Hallucination |
|---|---:|---:|---:|---:|
| BIPIA | 13,901 | 0.746 | 0.781 | 0.035 |
| LLMail-Inject | 9,998 | 0.744 | 0.772 | 0.047 |
| InjecAgent | 1,054 | 0.656 | 0.667 | 0.026 |
| **Overall** | **24,953** | **0.742** | **0.772** | **0.040** |

## Post-paper behavior analysis

This optional analysis was added after the paper. It uses a separate behavior
judge to label whether the target model carried out each ground-truth
instruction. It does not change the paper results above, the released
checkpoints, or the default evaluation metrics.

Using the paper's adversarial-detection definition, which requires mean
adversarial-instruction coverage of at least 0.5, PRISM detects 94.1% of attacks
that the target model followed (n = 9,494 rows). The corresponding recovery rate
for followed benign instructions is 98.6% (n = 21,216). The adversarial-instruction
follow rate over the corpus is 33.2%.

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
  response), against 3.1% pure behavior description and 7.6% fabrication.

The first three bullets come from the calibrated behavior judge
(`prism_eval/scoring/behavior_judge.py`, judge-vs-human κ = 0.734 and
human-vs-human κ = 0.786). The claim-side split comes from
`prism_eval/scoring/claim_provenance.py`, which has no gold set and should be
read as indicative. Both are optional post-processing steps applied to saved
evaluation outputs. See [Behavior analysis](BEHAVIOR_ANALYSIS.md) for the
method and commands.

## Ablations

Where the signal lives in the activation window:
[ABLATION_REPORT.md](ABLATION_REPORT.md).

## Judge calibration

Calibration methods, results, and commands are in the
[data card](../DATA_CARD.md#judge-calibration-data).
