#!/usr/bin/env bash
# Chunk ablation: 6 sequential runs of the GRPO 19k checkpoint, each reading
# a different consecutive 128-token chunk of the (near-uncapped, 2048-token)
# base response. Same board as the size/position sweep.
#
# chunkK = response tokens [K*128, K*128+128). Records whose response ends
# before the chunk produce an empty ITM report (activation_tokens=0 in the
# traces — used offline for per-chunk support counts + conditional metrics).
#
# Launch detached:
#   nohup setsid bash scripts/run_ablation_window_chunks.sh > results/window_ablation/chunks_driver.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

export PRISM_EVAL_BASE_URL="${PRISM_EVAL_BASE_URL:-http://localhost:8088/v1}"
export PRISM_EVAL_MODEL="${PRISM_EVAL_MODEL:-gemma4-31B-it}"
export PRISM_EVAL_API_KEY="${PRISM_EVAL_API_KEY:-EMPTY}"

# All chunk runs share identical greedy base responses (same base model,
# prompts, generation cap) — cache them so only the first run generates.
export PRISM_EVAL_BASE_RESPONSE_CACHE="${PRISM_EVAL_BASE_RESPONSE_CACHE:-results/window_ablation/base_responses.jsonl}"

mkdir -p results/window_ablation

run_one() {  # $1=config basename (no .yaml)   remaining args: NAME=VALUE env overrides
  local name="$1"; shift
  echo "=== [$(date '+%F %T')] start $name (env: $*)"
  env "$@" .venv/bin/python -m prism_eval.cli evaluate --offline \
      --config "configs/ablation/window/${name}.yaml" \
      > "results/window_ablation/${name}.log" 2>&1 \
      && echo "=== [$(date '+%F %T')] done  $name" \
      || echo "=== [$(date '+%F %T')] FAILED $name (continuing)"
}

# chunk0 = response tokens 0-127. The first run populates the shared base
# response cache, so it is slower than the rest.
for k in 0 1 2 3 4 5; do
  run_one "$(printf chunk%02d "$k")" \
      "PRISM_EVAL_ACT_WINDOW_POS=chunk${k}" \
      "PRISM_EVAL_BASE_GEN_MAX_NEW_TOKENS=2048"
done

echo "=== [$(date '+%F %T')] chunk ablation complete"
