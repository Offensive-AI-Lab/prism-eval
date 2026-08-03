"""Core data models for the ITM eval suite."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


class GenerationParams(BaseModel):
    """Parameters used to generate a parameterized eval."""

    n_turns: int
    n_instructions: int
    k_distribution: str  # "poisson" | "uniform" | "fixed"
    k_param: float  # lambda for poisson, or fixed K value
    flavor: str  # "benign" | "malicious"


class LongContextParams(BaseModel):
    """Metadata for long-context instruction-decay eval generation."""

    context_source: str  # "gutenberg", "arxiv", "synthetic"
    context_token_count: int  # tokens in the haystack (context after instruction)
    total_input_tokens: int  # full prompt token count (instruction + context)


# ---------------------------------------------------------------------------
# v2 schema models
# ---------------------------------------------------------------------------


class ProbePoint(BaseModel):
    """A location where the ITM is run and scored."""

    probe_id: str
    location: str  # "after_turn_N" or named anchor ("after_prompt", "end_of_response")
    expected_instructions: list[str]  # source keys from instruction_sources, or ["TBD_post_hoc"]


class EvalMetadata(BaseModel):
    """Human-readable metadata — never a scoring target.

    Allows arbitrary extra keys so loaders can stash free-form provenance
    (``_source_tag``) and dataset taxonomy (attacker_goal, injection_position,
    etc.) that survives a schema round-trip for later analysis/breakdowns.
    """

    model_config = ConfigDict(extra="allow")

    attack_pattern: str | None = None
    notes: str | None = None


class ProbeResult(BaseModel):
    """ITM output at a single probe point."""

    probe_id: str
    itm_report: str
    input_tokens: int | None = None
    activation_tokens: int | None = None


# ---------------------------------------------------------------------------
# Core eval record (supports v1 and v2)
# ---------------------------------------------------------------------------


class EvalRecord(BaseModel):
    """A single evaluation case with prompt(s) and ground-truth annotations."""

    schema_version: str = "2.0"
    eval_id: str
    setting: str  # "AP" | "HO" | "BC" | "BN"
    category: str
    structural_tags: list[str]
    difficulty: str  # "easy" | "medium" | "hard"
    source: str  # "hand_crafted" | "generated"

    prompt: str | None = None
    prompt_turns: list[str] | None = None
    # Structured multi-message conversation: [{role, content}], role in
    # system|user|assistant|tool. Lets a record place untrusted content in a
    # tool/document turn (XPIA framing) rather than a hand-concatenated string.
    # Rendered verbatim through the model's chat template by the runner.
    prompt_messages: list[dict] | None = None

    # --- v2 fields ---
    instruction_sources: dict[str, list[str]] | None = None
    probe_points: list[ProbePoint] | None = None
    metadata: EvalMetadata | None = None

    # --- v1 deprecated fields (kept for backward compat / auto-upgrade) ---
    constraints: list[str] | None = None
    goals: list[str] | None = None

    # Parameterized eval metadata
    generation_params: GenerationParams | None = None
    instruction_positions: list[int] | None = None
    long_context_params: LongContextParams | None = None

    model_override: str | None = None

    @model_validator(mode="after")
    def check_prompt_present(self) -> "EvalRecord":
        if (
            self.prompt is None
            and self.prompt_turns is None
            and self.prompt_messages is None
        ):
            raise ValueError(
                "One of prompt, prompt_turns, or prompt_messages must be provided"
            )
        return self

    @model_validator(mode="after")
    def check_instructions_present(self) -> "EvalRecord":
        has_v2 = self.instruction_sources is not None
        has_v1 = self.constraints is not None

        if not has_v2 and not has_v1:
            raise ValueError(
                "Either instruction_sources (v2) or constraints (v1) must be provided"
            )

        if has_v2 and "original" not in self.instruction_sources:
            raise ValueError("instruction_sources must contain an 'original' key")

        return self


class EvalSuite(BaseModel):
    """A collection of eval records."""

    schema_version: str = "2.0"
    evals: list[EvalRecord]


class EvalResult(BaseModel):
    """Output from running an eval through an ITM runner."""

    eval_id: str
    runner: str
    itm_report: str
    model_response: str | None = None
    per_turn_reports: list[str] | None = None
    probe_results: list[ProbeResult] | None = None
    timestamp: str

    # Runtime token counts (populated by runner, used for long-context analysis)
    input_tokens: int | None = None
    output_tokens: int | None = None
    activation_tokens: int | None = None  # how many tokens activations were extracted from


class ClaimScore(BaseModel):
    """Score for a single ground-truth claim."""

    claim: str
    score: float  # 1.0 / 0.5 / 0.0
    evidence: str | None = None


class ScorerOutput(BaseModel):
    """Output from a single scoring method.

    v2 hallucination spec (current): per-bullet ``hallucination_scores`` —
    one score per ITM-REPORT bullet in {0.0, 0.5, 1.0}, where higher = more
    hallucinated. ``mean(hallucination_scores)`` is the canonical halluc
    metric (drives the reward formula in rl_phase1 and the leaderboard).

    v1 (deprecated): ``hallucination_count`` (int) — count of fully
    fabricated claims. Kept on the model so legacy snapshots still load,
    but v2-produced records leave it at 0. New code MUST read
    ``hallucination_scores``.
    """

    scorer: str
    instruction_scores: list[ClaimScore] = []
    # v2 — per-ITM-bullet hallucination scores (0/0.5/1, higher = worse).
    hallucination_scores: list[float] = []
    # Per-hallucination descriptions from the judge LLM (populated only by
    # judge_llm; other scorers leave this empty). v2 keeps this for parity
    # with the optional `details` block the judge can still emit.
    hallucination_details: list[str] = []

    # v1 — deprecated. Auto-zeroed for v2-produced records.
    hallucination_count: int = 0

    # Deprecated v1 fields — kept so old JSONL results can still be loaded
    constraint_scores: list[ClaimScore] = []
    goal_scores: list[ClaimScore] = []


class ScoredResult(BaseModel):
    """An eval result scored by multiple methods."""

    eval_id: str
    runner: str
    scores: dict[str, ScorerOutput]  # keyed by scorer name
