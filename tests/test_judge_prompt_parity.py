"""Judge LLM prompt-parity tests (v2 — per-bullet hallucination scores).

The judge must see the same inputs as a human annotator — prompt, model
response, GT instructions, ITM report — so a claim grounded in the prompt
or response isn't miscounted as a hallucination.

These tests verify the user-message content without hitting a real LLM:
we inject a stub OpenAI client that records the messages it was called
with and returns a fixed two-line text payload.
"""

from __future__ import annotations

from types import SimpleNamespace

from prism_eval.scoring import judge_llm


class _CapturingClient:
    """Minimal OpenAI-compatible stub that records the last messages list."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.last_messages: list[dict] | None = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Accept any kwargs (model, temperature, messages, max_tokens, etc.) so
        # this stub keeps working when the v2 judge call gains new knobs.
        self.last_messages = kwargs.get("messages")
        choice = SimpleNamespace(message=SimpleNamespace(content=self._response_text))
        return SimpleNamespace(choices=[choice])


def _stub_judge_response(n_instructions: int, n_report_bullets: int = 0) -> str:
    """Build a v2 two-line text response that the judge would emit."""
    inst = ",".join(["1.0"] * n_instructions)
    halluc = ",".join(["0.0"] * n_report_bullets)
    return f"INSTRUCTIONS: {inst}\nHALLUCINATIONS: {halluc}"


def test_judge_user_message_includes_prompt_and_response() -> None:
    client = _CapturingClient(_stub_judge_response(n_instructions=2, n_report_bullets=2))

    out = judge_llm.score(
        instructions=["be funny", "act like a comedian"],
        report="- mentioned being funny\n- respond in haiku format",
        prompt="Be funny. Act like a comedian. Also respond in haiku format.",
        model_response="A short poem\nAbout a mischievous cat\nSpring blossoms unseen",
        model="stub-judge",
        client=client,
    )

    # v2: per-bullet hallucination scores, hallucination_count deprecated → 0
    assert out.hallucination_scores == [0.0, 0.0]
    assert out.hallucination_count == 0
    user_content = client.last_messages[1]["content"]
    assert "## Original Prompt" in user_content
    assert "Be funny. Act like a comedian" in user_content
    assert "## Model Response" in user_content
    assert "A short poem" in user_content
    assert "## Ground-Truth Instructions" in user_content
    assert "## ITM Report" in user_content


def test_judge_backward_compat_without_prompt_context() -> None:
    """Legacy callers passing only instructions+report still work."""
    client = _CapturingClient(_stub_judge_response(n_instructions=1, n_report_bullets=1))

    judge_llm.score(
        instructions=["be funny"],
        report="- mentioned being funny",
        model="stub-judge",
        client=client,
    )

    user_content = client.last_messages[1]["content"]
    assert "(not provided" in user_content


def test_system_prompt_specifies_all_four_inputs() -> None:
    """Belt-and-braces: the SYSTEM_PROMPT must name all four inputs so
    the judge knows what it's looking at — the rubric change is meaningless
    if the prompt still says 'you will be given 1. instructions 2. report'."""
    system = judge_llm.SYSTEM_PROMPT
    assert "ORIGINAL PROMPT" in system
    assert "MODEL RESPONSE" in system
    assert "GROUND-TRUTH INSTRUCTIONS" in system
    assert "ITM REPORT" in system
    # Hallucination rule must cite all three grounding sources.
    assert "prompt" in system.lower()
    assert "response" in system.lower()


def test_module_prompt_is_the_shipped_file():
    """RUBRIC.md §5 points at the file; the module must load exactly that.

    The rubric used to inline its own copy of the judge prompt, and the two
    silently diverged (4839 chars documented vs 6054 live). Now there is one
    source of truth and this asserts the module agrees with it.
    """
    from pathlib import Path

    import prism_eval
    from prism_eval.scoring import adversarial_identifier, judge_llm

    prompt_dir = Path(prism_eval.__file__).parent / "prompts" / "judge"
    for mod, fname in (
        (judge_llm, "published_judge.txt"),
        (adversarial_identifier, "published_advdet.txt"),
    ):
        assert mod.SYSTEM_PROMPT == (prompt_dir / fname).read_text(), (
            f"{mod.__name__}.SYSTEM_PROMPT has drifted from {fname}"
        )


def test_rubric_does_not_inline_a_second_copy_of_the_prompt():
    """Guard against someone pasting the prompt back into RUBRIC.md."""
    from pathlib import Path

    rubric = (Path(__file__).parent.parent / "RUBRIC.md").read_text()
    assert "You are an evaluation judge" not in rubric, (
        "RUBRIC.md inlines a judge prompt again — it will drift from the code. "
        "Point at prism_eval/prompts/judge/published_judge.txt instead."
    )


def test_rubric_documents_the_metrics_the_code_emits():
    """RUBRIC.md must not headline metrics that were dropped."""
    from pathlib import Path

    rubric = (Path(__file__).parent.parent / "RUBRIC.md").read_text()
    for metric in ("reward", "recall", "mean_halluc", "length_penalty"):
        assert metric in rubric, f"RUBRIC.md never mentions {metric}"
    # F1/Precision may appear only in the "superseded" note, not as the headline.
    headline = rubric.split("## 4. Aggregate Metrics")[0]
    assert "harmonic mean of Recall and Precision" not in headline
