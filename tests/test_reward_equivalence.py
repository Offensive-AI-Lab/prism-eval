"""Reward-formula equivalence: eval_suite leaderboard ≡ rl_phase1 training reward.

The whole point of the v2 spec migration is that eval_suite and rl_phase1 are
driven by *the same number*: a JudgeLLMScorer's per-row ``reward`` and an
rl_phase1 GRPO step's per-candidate ``reward`` should be identical when given
the same (recall, mean_hallucination_score, n_gt, n_itm) inputs. If anyone
retunes the weights in one place without updating the other, this test fires.

We deliberately don't import rl_phase1 (separate repo) — instead, we
re-derive its formula inline using the canonical default weights and assert
the eval_suite JudgeLLMScorer's ``.score()`` produces the same scalar.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from prism_eval import weave_eval


# Canonical reward formula, ported from rl_phase1/judge_pref.py. Any change
# here must be matched in rl_phase1/config.py and weave_eval.py defaults.
_W_INST_CANONICAL = 1.0
_W_HALLUC_CANONICAL = 0.4
_LP_K_CANONICAL = 1.5
_LP_LAMBDA_CANONICAL = 0.15


def _rl_phase1_reward(
    recall: float,
    mean_halluc: float,
    n_gt: int,
    n_itm: int,
    w_inst: float = _W_INST_CANONICAL,
    w_halluc: float = _W_HALLUC_CANONICAL,
    k: float = _LP_K_CANONICAL,
    lam: float = _LP_LAMBDA_CANONICAL,
) -> float:
    """Verbatim reproduction of rl_phase1's reward formula."""
    length_penalty = lam * max(0.0, n_itm - k * n_gt)
    return w_inst * recall - w_halluc * mean_halluc - length_penalty


class _CapturingClient:
    """Stub OpenAI client that returns a fixed two-line judge response."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        choice = SimpleNamespace(
            message=SimpleNamespace(content=self._response_text)
        )
        return SimpleNamespace(choices=[choice])


def _stub_judge_text(inst_csv: str, halluc_csv: str) -> str:
    return f"INSTRUCTIONS: {inst_csv}\nHALLUCINATIONS: {halluc_csv}"


def _run_scorer(
    inst_csv: str,
    halluc_csv: str,
    instructions: list[str],
    n_itm: int,
) -> dict:
    """Drive JudgeLLMScorer end-to-end against a stub LLM client.

    Returns the per-row score dict the scorer would emit during a Weave
    Evaluation, with the v2 reward already computed.
    """
    # Build an ITM report whose split_report_bullets count matches n_itm so
    # the scorer's length-penalty math sees the right ITM bullet count.
    report = "\n".join(f"- bullet {i + 1}" for i in range(n_itm)) or "(empty)"
    scorer = weave_eval.JudgeLLMScorer(judge_model="stub")

    # Inject the stub client by patching _get_judge_client for the duration
    # of the call. The scorer's own machinery (length penalty, etc.) runs
    # for real.
    stub = _CapturingClient(_stub_judge_text(inst_csv, halluc_csv))
    import prism_eval.weave_eval as _we

    original = _we._get_judge_client
    _we._get_judge_client = lambda *_a, **_kw: stub  # type: ignore[assignment]
    try:
        return scorer.score(
            instructions=instructions,
            output={"itm_report": report, "model_response": "(test)"},
            setting="AP",
            eval_id="T-001",
            prompt="test prompt",
        )
    finally:
        _we._get_judge_client = original  # type: ignore[assignment]


@pytest.mark.parametrize(
    "inst_csv, halluc_csv, n_instructions, n_itm",
    [
        # Perfect recall, no halluc, no length overrun → reward = w_inst
        ("1.0,1.0", "0.0,0.0", 2, 2),
        # Zero recall, full halluc, no length overrun → reward = -w_halluc
        ("0.0,0.0", "1.0,1.0", 2, 2),
        # Mixed: 0.5 recall, 0.25 mean halluc, no length penalty
        ("1.0,0.5,0.0", "0.0,0.0,0.5,0.5", 3, 4),
        # Length penalty fires (n_itm > k*n_gt = 1.5*2 = 3): overrun = 4 - 3 = 1
        ("1.0,1.0", "0.0,0.0,0.0,0.0", 2, 4),
        # Length penalty + halluc + partial recall, the busiest path
        ("0.5,0.5", "0.5,0.5,1.0", 2, 3),
    ],
)
def test_scorer_reward_matches_rl_phase1_formula(
    inst_csv: str, halluc_csv: str, n_instructions: int, n_itm: int
) -> None:
    instructions = [f"gt_{i}" for i in range(n_instructions)]
    out = _run_scorer(inst_csv, halluc_csv, instructions, n_itm)

    expected = _rl_phase1_reward(
        recall=out["coverage"],
        mean_halluc=out["mean_hallucination_score"],
        n_gt=n_instructions,
        n_itm=n_itm,
    )
    assert out["reward"] == pytest.approx(expected, abs=1e-9), (
        f"eval_suite reward {out['reward']} != rl_phase1 formula {expected} "
        f"for inst={inst_csv} halluc={halluc_csv} n_gt={n_instructions} n_itm={n_itm}"
    )


def test_scorer_defaults_match_canonical_weights() -> None:
    """If someone retunes JudgeLLMScorer defaults without updating
    rl_phase1/config.py, the parametrised test above would still pass
    (it uses the scorer's own weights). This catches drift directly."""
    s = weave_eval.JudgeLLMScorer(judge_model="stub")
    assert s.instruction_weight == _W_INST_CANONICAL
    assert s.hallucination_weight == _W_HALLUC_CANONICAL
    assert s.length_penalty_enabled is True
    assert s.length_penalty_k == _LP_K_CANONICAL
    assert s.length_penalty_lambda == _LP_LAMBDA_CANONICAL


def test_length_penalty_zeroed_when_disabled() -> None:
    """Length penalty is the only term explicitly toggleable."""
    instructions = ["gt_0", "gt_1"]
    stub = _CapturingClient(_stub_judge_text("1.0,1.0", "0.0,0.0,0.0,0.0,0.0"))
    import prism_eval.weave_eval as _we

    original = _we._get_judge_client
    _we._get_judge_client = lambda *_a, **_kw: stub
    try:
        scorer = weave_eval.JudgeLLMScorer(
            judge_model="stub", length_penalty_enabled=False
        )
        out = scorer.score(
            instructions=instructions,
            output={"itm_report": "- a\n- b\n- c\n- d\n- e", "model_response": "(test)"},
            setting="AP",
            eval_id="T-LP-OFF",
            prompt="test",
        )
    finally:
        _we._get_judge_client = original

    # 5 ITM > k*n_gt = 1.5*2 = 3 → overrun is 2, but penalty disabled → 0.
    assert out["length_penalty"] == 0.0
    # Reward = 1.0*1.0 - 0.4*0.0 - 0.0 = 1.0
    assert out["reward"] == pytest.approx(1.0)
