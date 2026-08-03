#!/usr/bin/env bash
# Deep-chunk ablation (round 3): chunks 6-15 of the 2048-cap response,
# extending scripts/run_ablation_window_chunks.sh past token 768 to the generation
# cap. Same board (prism_window_ablation_leaderboard), same cached base responses.
# Supports shrink to ~34 by chunk 15 — report these rows conditioned on
# support with the small-n flag, as in docs/ABLATION_REPORT.md.
#
# Launch detached:
#   nohup setsid bash scripts/run_ablation_window_chunks_deep.sh > results/window_ablation/chunks_deep_driver.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

export PRISM_EVAL_BASE_URL="${PRISM_EVAL_BASE_URL:-http://localhost:8088/v1}"
export PRISM_EVAL_MODEL="${PRISM_EVAL_MODEL:-gemma4-31B-it}"
export PRISM_EVAL_API_KEY="${PRISM_EVAL_API_KEY:-EMPTY}"
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

for k in 6 7 8 9 10 11 12 13 14 15; do
  run_one "$(printf chunk%02d "$k")" \
      "PRISM_EVAL_ACT_WINDOW_POS=chunk${k}" \
      "PRISM_EVAL_BASE_GEN_MAX_NEW_TOKENS=2048"
done

echo "=== [$(date '+%F %T')] deep chunk ablation complete"
