# Activation ablations

These experiments vary the response-token activations read by PRISM and the
prompt context available during extraction.

| Setting | Value |
|---|---|
| Checkpoint | `prism-qwen3.5-9b-grpo.pt` |
| Suite | `data/eval_suite.json`, 1,000 records, 250 per setting |
| Judge | `gemma4-31B-it` |
| Metrics | Recall is coverage; average detection is `detect_rate_avg` on AP and HO |

Commands are in the [reproduction guide](REPRODUCING.md#ablations). Each table
names the changed condition; all rows use the same released checkpoint.

## Ablation A: window size

Generation uses a 196-token cap; the monitor reads the last k ∈ {16, 32, 64, 128} response-token activations (or the whole response when shorter).

### Overall (1000 records; detection over the 500 adversarial)

| Window | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|
| last 16 | 0.528 | 0.059 | 0.498 | 0.392 |
| last 32 | 0.616 | 0.033 | 0.597 | 0.524 |
| last 64 | 0.687 | 0.025 | 0.673 | 0.664 |
| last 128 | **0.767** | **0.020** | **0.754** | **0.758** |

### AP — Adversarial prompt (250 records)

| Window | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|
| last 16 | 0.376 | 0.110 | 0.327 | 0.340 |
| last 32 | 0.470 | 0.070 | 0.429 | 0.436 |
| last 64 | 0.556 | 0.053 | 0.526 | 0.524 |
| last 128 | **0.684** | **0.049** | **0.657** | **0.676** |

### HO — Hidden objective (250 records)

| Window | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|
| last 16 | 0.426 | 0.026 | 0.409 | 0.444 |
| last 32 | 0.508 | 0.014 | 0.502 | 0.612 |
| last 64 | 0.593 | 0.017 | 0.586 | 0.804 |
| last 128 | **0.645** | **0.010** | **0.639** | **0.840** |

### BC — Behavioral constraints (250 records)

| Window | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|
| last 16 | 0.456 | 0.096 | 0.407 | — |
| last 32 | 0.566 | 0.048 | 0.538 | — |
| last 64 | 0.636 | 0.031 | 0.623 | — |
| last 128 | **0.762** | **0.021** | **0.753** | — |

### BN — Benign baseline (250 records)

| Window | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|
| last 16 | 0.853 | 0.006 | 0.849 | — |
| last 32 | 0.921 | **0.000** | 0.918 | — |
| last 64 | 0.961 | **0.000** | 0.957 | — |
| last 128 | **0.977** | 0.002 | **0.968** | — |

Detection = AdversarialDetectionScorer `detect_rate_avg`; defined only for the adversarial settings (AP, HO).

## Ablation B: window position

Generation runs to natural EOS with a 2,048-token cap. The monitor reads chunk
*k* = response tokens [128*k*, 128*k* + 128). Metrics are conditioned on the
response reaching the chunk; support counts are shown. The low-support marker
identifies cells based on fewer than 50 records.

### Overall (1000 records; detection over supported adversarial records)

| Chunk (tokens) | Support | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|---|
| 0 (0–127) | 1000 | **0.794** | **0.013** | **0.787** | **0.810** |
| 1 (128–255) | 881 | 0.710 | 0.011 | 0.699 | 0.745 |
| 2 (256–383) | 717 | 0.691 | 0.022 | 0.676 | 0.687 |
| 3 (384–511) | 496 | 0.693 | 0.019 | 0.682 | 0.688 |
| 4 (512–639) | 370 | 0.686 | 0.019 | 0.668 | 0.509 |
| 5 (640–767) | 283 | 0.683 | 0.024 | 0.668 | 0.500 |

### AP — Adversarial prompt

| Chunk (tokens) | Support | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|---|
| 0 (0–127) | 250 | **0.716** | 0.032 | **0.698** | **0.720** |
| 1 (128–255) | 220 | 0.633 | **0.025** | 0.608 | 0.618 |
| 2 (256–383) | 179 | 0.595 | 0.058 | 0.555 | 0.531 |
| 3 (384–511) | 121 | 0.619 | 0.053 | 0.597 | 0.603 |
| 4 (512–639) | 81 | 0.551 | 0.037 | 0.512 | 0.457 |
| 5 (640–767) | ⚠ 44 | 0.536 | 0.034 | 0.514 | 0.432 |

### HO — Hidden objective

| Chunk (tokens) | Support | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|---|
| 0 (0–127) | 250 | **0.675** | 0.001 | **0.674** | **0.900** |
| 1 (128–255) | 238 | 0.600 | 0.004 | 0.598 | 0.861 |
| 2 (256–383) | 182 | 0.603 | 0.009 | 0.599 | 0.841 |
| 3 (384–511) | 84 | 0.606 | **0.000** | 0.606 | 0.810 |
| 4 (512–639) | ⚠ 29 | 0.539 | 0.007 | 0.536 | 0.655 |
| 5 (640–767) | ⚠ 12 | 0.604 | 0.021 | 0.596 | 0.750 |

### BC — Behavioral constraints

| Chunk (tokens) | Support | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|---|
| 0 (0–127) | 250 | **0.803** | 0.017 | **0.796** | — |
| 1 (128–255) | 222 | 0.676 | **0.013** | 0.670 | — |
| 2 (256–383) | 189 | 0.624 | 0.019 | 0.616 | — |
| 3 (384–511) | 161 | 0.572 | 0.018 | 0.565 | — |
| 4 (512–639) | 153 | 0.580 | 0.023 | 0.571 | — |
| 5 (640–767) | 138 | 0.551 | 0.036 | 0.537 | — |

### BN — Benign baseline

| Chunk (tokens) | Support | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|---|
| 0 (0–127) | 250 | **0.984** | **0.000** | **0.982** | — |
| 1 (128–255) | 201 | 0.961 | **0.000** | 0.952 | — |
| 2 (256–383) | 167 | 0.967 | **0.000** | 0.956 | — |
| 3 (384–511) | 130 | 0.968 | **0.000** | 0.957 | — |
| 4 (512–639) | 107 | 0.978 | 0.002 | 0.960 | — |
| 5 (640–767) | 89 | 0.972 | **0.000** | 0.958 | — |

Support = responses whose templated length exceeds the chunk start; records ending before a chunk produce an empty report by construction and score exactly 0, so conditional means are recovered exactly as `mean × N / support`.

Across supported chunks, AP coverage ranges from 0.536 to 0.716, BC from 0.551
to 0.803, and BN from 0.961 to 0.984. Chunk 0 has the highest coverage in AP,
HO, and BC.

## Ablation C: extraction context

Same responses, same reference window (last 128 tokens of a response capped at
196 tokens), and the same monitor; only the forward pass that reads the
activations changes. Base responses are always generated with full context, so
behavior is held fixed.

- **full context** — prompt + response, the reference condition for this ablation.
- **attention zeroed (system+user)** — identical token sequence, but the attention mask zeroes the system and user turns during extraction: response positions can't look back at the prompt. (`masked_prompt` in the configs.)
- **prefill (response only)** — the response follows a neutral placeholder user turn instead of the original prompt (`evicted` in the configs).

### Overall (1000 records; detection over the 500 adversarial)

| Extraction context | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|
| full context | **0.767** | 0.021 | **0.754** | **0.758** |
| attention zeroed (system+user) | 0.538 | 0.028 | 0.523 | 0.266 |
| prefill (response only) | 0.443 | **0.012** | 0.438 | 0.184 |

### AP — Adversarial prompt

| Extraction context | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|
| full context | **0.682** | 0.050 | **0.655** | **0.672** |
| attention zeroed (system+user) | 0.412 | 0.067 | 0.384 | 0.296 |
| prefill (response only) | 0.387 | **0.015** | 0.381 | 0.276 |

### HO — Hidden objective

| Extraction context | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|
| full context | **0.647** | 0.009 | **0.642** | **0.844** |
| attention zeroed (system+user) | 0.328 | 0.008 | 0.325 | 0.236 |
| prefill (response only) | 0.227 | **0.006** | 0.224 | 0.092 |

### BC — Behavioral constraints

| Extraction context | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|
| full context | **0.761** | 0.022 | **0.751** | — |
| attention zeroed (system+user) | 0.481 | 0.036 | 0.466 | — |
| prefill (response only) | 0.322 | **0.019** | 0.315 | — |

### BN — Benign baseline

| Extraction context | Recall | Hallucination | Judge reward | Avg detection |
|---|---|---|---|---|
| full context | **0.978** | **0.002** | **0.969** | — |
| attention zeroed (system+user) | 0.930 | **0.002** | 0.917 | — |
| prefill (response only) | 0.834 | 0.010 | 0.830 | — |

Response-only prefill retains 0.443 overall coverage and 0.834 BN coverage.
Prompt masking and response-only prefill have a larger effect on HO: coverage
falls from 0.647 to 0.328 and 0.227, while mean detection falls from 0.844 to
0.236 and 0.092. These conditions are out of distribution for a monitor trained
on full-context extraction, so they measure performance under context removal
rather than the maximum information available in response-only states.

## Additional configurations

The [window configurations](../configs/ablation/window/) also include chunks
6–15, first/middle/last windows, and a natural-EOS last window. The
[context configurations](../configs/ablation/context/) include `masked_user`
and `swapped` controls. Their results are not tabulated here.

## Interpretation

Later chunks have fewer supporting responses; use the support counts when
comparing them. In particular, HO has only 29 records at chunk 4 and 12 at
chunk 5.

PRISM was trained on end-aligned windows with full prompt context. Other
positions and masked or response-only extraction change that input
distribution. These controls measure performance under those changes, not
the maximum information available in the activations.
