#!/usr/bin/env bash
# Extraction-context ablation: 5 sequential runs of the
# released checkpoint that vary only what the activation-extraction forward
# pass can see — the base responses are identical in every condition (shared
# greedy cache), so any score delta is attributable to prompt visibility at
# extraction time. All rows land on prism_context_ablation_leaderboard (its own
# board, separate from the main and window-ablation leaderboards).
#
#   full           prompt + response (published behavior; baseline row)
#   masked_prompt  same tokens, prompt hidden from attention
#   masked_user    same tokens, user turn hidden, system visible
#   evicted        response behind a neutral placeholder (true overflow sim)
#   swapped        response behind a donor prompt (directional test)
#
# The runner reads the condition from PRISM_EVAL_ACT_CONTEXT (env override —
# runner _cfg comes from the checkpoint, so YAML can't carry it).
#
# Post-run, score the swapped run's reports against the DONOR ground truth
# (results/act_context_ablation/donor_suite.json) offline via `prism-eval score`
# for the directional readout.
#
# Launch detached:
#   nohup setsid bash scripts/run_ablation_context.sh > results/act_context_ablation/driver.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

export PRISM_EVAL_BASE_URL="${PRISM_EVAL_BASE_URL:-http://localhost:8088/v1}"
export PRISM_EVAL_MODEL="${PRISM_EVAL_MODEL:-gemma4-31B-it}"
export PRISM_EVAL_API_KEY="${PRISM_EVAL_API_KEY:-EMPTY}"

# Base responses depend only on (base model, messages, 196-token cap) — the
# window-ablation cache already holds them, so no condition regenerates.
export PRISM_EVAL_BASE_RESPONSE_CACHE="${PRISM_EVAL_BASE_RESPONSE_CACHE:-results/window_ablation/base_responses.jsonl}"

mkdir -p results/act_context_ablation

SWAP_PAIRS=results/act_context_ablation/swap_pairs.json
if [ ! -f "$SWAP_PAIRS" ]; then
  echo "=== [$(date '+%F %T')] generating swap pairs"
  .venv/bin/python scripts/make_act_swap_pairs.py \
      --suite data/eval_suite_v2_final.json \
      --pairs-out "$SWAP_PAIRS" \
      --donor-suite-out results/act_context_ablation/donor_suite.json \
      || { echo "swap-pair generation FAILED — aborting"; exit 1; }
fi

run_one() {  # $1=config basename (no .yaml)   remaining args: NAME=VALUE env overrides
  local name="$1"; shift
  echo "=== [$(date '+%F %T')] start $name (env: $*)"
  env "$@" .venv/bin/python -m prism_eval.cli evaluate --offline \
      --config "configs/ablation/context/${name}.yaml" \
      > "results/act_context_ablation/${name}.log" 2>&1 \
      && echo "=== [$(date '+%F %T')] done  $name" \
      || echo "=== [$(date '+%F %T')] FAILED $name (continuing)"
}

run_one full          PRISM_EVAL_ACT_CONTEXT=full
run_one masked_prompt PRISM_EVAL_ACT_CONTEXT=masked_prompt
run_one masked_user   PRISM_EVAL_ACT_CONTEXT=masked_user
run_one evicted       PRISM_EVAL_ACT_CONTEXT=evicted
run_one swapped       PRISM_EVAL_ACT_CONTEXT=swapped "PRISM_EVAL_ACT_SWAP_FILE=${SWAP_PAIRS}"

echo "=== [$(date '+%F %T')] act-context ablation complete"
