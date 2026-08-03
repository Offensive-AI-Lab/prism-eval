#!/usr/bin/env bash
# Serve the judge model with vLLM on an OpenAI-compatible endpoint.
#
# The judge scores every ITM report against ground truth (see RUBRIC.md) and
# identifies adversarial bullets (RUBRIC_ADVDET.md). Self-hosting is optional:
# scoring.judge_model / scoring.judge_base_url accept ANY OpenAI-compatible
# endpoint, including hosted APIs. This script reproduces the exact judge the
# published numbers were produced with.
#
# Usage:
#   ./scripts/serve_judge.sh                       # downloads the model if needed
#   ./scripts/serve_judge.sh /path/to/local/model  # use a local copy
#
# Then, in another shell (or .env):
#   export PRISM_EVAL_MODEL=gemma4-31B-it
#   export PRISM_EVAL_BASE_URL=http://localhost:8088/v1
#   export PRISM_EVAL_API_KEY=EMPTY
#
# IMPORTANT: --served-model-name is part of the scorer's identity, which is
# part of the evaluation digest. Changing it puts your rows on a different
# leaderboard than the published ones. Leave it alone unless you mean to.
#
# Requires: vllm, and a GPU with enough memory for a 31B model in bf16
# (~80 GB; use --tensor-parallel-size to shard across several).

set -euo pipefail

MODEL="${1:-google/gemma-4-31B-it}"
SERVED_NAME="${PRISM_JUDGE_SERVED_NAME:-gemma4-31B-it}"
PORT="${PRISM_JUDGE_PORT:-8088}"
MAX_LEN="${PRISM_JUDGE_MAX_LEN:-32384}"

echo "Serving judge:"
echo "  model            $MODEL"
echo "  served-model-name $SERVED_NAME   (part of the scorer identity — do not change casually)"
echo "  endpoint         http://localhost:${PORT}/v1"

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --max_model_len "$MAX_LEN" \
  --enable-prefix-caching \
  --reasoning-parser gemma4
