#!/usr/bin/env bash
# Activation-window ablation: 7 sequential runs of the released checkpoint
# with different activation windows, all landing on prism_window_ablation_leaderboard.
#
# The runner's window/generation params live in the checkpoint config, so
# each run overrides them via env vars (read by runners/prism.py):
#   PRISM_EVAL_MAX_ACT_TOKENS        window size (default 128)
#   PRISM_EVAL_ACT_WINDOW_POS        start | middle | end (default end)
#   PRISM_EVAL_BASE_GEN_MAX_NEW_TOKENS  base-response generation cap (default 196)
#
# Launch detached:
#   nohup setsid bash scripts/run_ablation_window.sh > results/window_ablation/driver.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

export PRISM_EVAL_BASE_URL="${PRISM_EVAL_BASE_URL:-http://localhost:8088/v1}"
export PRISM_EVAL_MODEL="${PRISM_EVAL_MODEL:-gemma4-31B-it}"
export PRISM_EVAL_API_KEY="${PRISM_EVAL_API_KEY:-EMPTY}"

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

# ── Axis A: window size, end-aligned, generation capped at 196 ──────────
run_one size_last16          PRISM_EVAL_MAX_ACT_TOKENS=16
run_one size_last32          PRISM_EVAL_MAX_ACT_TOKENS=32
run_one size_last64          PRISM_EVAL_MAX_ACT_TOKENS=64
run_one size_last128         # published setting — ablation-board baseline row

# ── Axis B: window position, 128 tokens of the full (uncapped) response ─
run_one pos_first128         PRISM_EVAL_ACT_WINDOW_POS=start  PRISM_EVAL_BASE_GEN_MAX_NEW_TOKENS=768
run_one pos_mid128           PRISM_EVAL_ACT_WINDOW_POS=middle PRISM_EVAL_BASE_GEN_MAX_NEW_TOKENS=768
run_one pos_last128          PRISM_EVAL_ACT_WINDOW_POS=end    PRISM_EVAL_BASE_GEN_MAX_NEW_TOKENS=768

echo "=== [$(date '+%F %T')] window ablation complete"
