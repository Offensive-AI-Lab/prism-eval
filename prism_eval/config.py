"""Experiment config schema for `prism-eval evaluate`.

A single YAML file describes one experiment end-to-end: which suite, which
runner+checkpoint, which scorers, and how it should appear in Weave.

The `experiment.name` flows into:
  - Weave call attributes (every traced row is filterable by experiment)
  - The `weave.Evaluation` name (defaults to `{experiment.name}_v1`)
  - Git sha auto-captured for reproducibility

Example:

    experiment:
      name: round2_layer18_qwen3judge
      notes: "Layer 18 activations, Qwen3-8B judge"
      tags: [round2, layer18]
      weave_project: null          # null = offline; "<entity>/<project>" to trace

    suite:
      path: data/eval_suite.json
      settings: [AP, BC, BN]

    runner:
      type: prism
      checkpoint: /mnt/.../best.pt
      device: cuda

    scoring:
      bertscore: false
      judge_llm: true
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ExperimentMeta(BaseModel):
    name: str = Field(..., description="Unique experiment identifier — appears in Weave and on disk")
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    weave_project: str | None = Field(
        default=None,
        description="W&B Weave project as '<entity>/<project>'. Leave null to "
        "run without Weave — `prism-eval evaluate` then requires --offline, "
        "which writes results to local JSON instead of tracing.",
    )


class SuiteConfig(BaseModel):
    path: Path | list[Path] = Field(
        ...,
        description="Path to eval suite JSON, or a list of paths to merge into one slice "
        "(used for the `all` union dataset).",
    )
    settings: list[str] | None = Field(
        default=None,
        description="Filter to these settings (e.g. [S1, BC, BN]). null = all.",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Filter to evals whose structural_tags intersect this set "
        "(e.g. [single-turn]). null = no tag filter.",
    )
    per_setting_limit: dict[str, int] | None = Field(
        default=None,
        description="Cap the number of samples per setting code after all filters are applied "
        "(e.g. {AP: 5, BN: 3}). Applied after union merge and dedup, so works correctly "
        "with multi-path configs. Settings not listed are uncapped.",
    )

    @property
    def paths(self) -> list[Path]:
        return list(self.path) if isinstance(self.path, list) else [self.path]


class RunnerConfig(BaseModel):
    # Keep in sync with RUNNER_TYPES in prism_eval/runners/__init__.py.
    type: Literal[
        "prism",
        "text_only_baseline",
    ] = "prism"
    checkpoint: Path | None = Field(
        default=None,
        description="Path to the runner's checkpoint. Required for every runner "
        "except `text_only_baseline`, which has no learned weights.",
    )
    device: str = "cuda"
    display_name: str | None = Field(
        default=None,
        description="Human-readable label for this model on the Weave "
        "leaderboard. Sets `weave.Model.name`, which becomes the row "
        "identifier. Defaults to `type` when unset (e.g. 'prism'), "
        "so multiple configs sharing a runner type would otherwise collapse "
        "to the same row label. Examples: 'SFT', 'GRPO_best_10k_steps'.",
    )

    # ── text_only_baseline params (ignored by other runners) ────────────────
    # Sanity-check baseline: take the last K tokens of the response model's
    # output and ask a frontier LLM to recover the instructions from text
    # alone. If the activation-based runners don't beat this on the
    # leaderboard, the activations aren't load-bearing.
    response_model_id: str | None = Field(
        default=None,
        description="HF model id for the base model that generates the response "
        "(text_only_baseline only). Should match the base model used by the "
        "activation runners so the response distribution is comparable.",
    )
    baseline_llm: str | None = Field(
        default=None,
        description="OpenAI-compatible model id called on the response tail "
        "(text_only_baseline only).",
    )
    baseline_base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible base URL for baseline_llm. Falls back to "
        "the OpenAI default when unset (text_only_baseline only).",
    )
    tail_tokens: int = Field(
        default=128,
        ge=1,
        description="How many trailing response tokens (in response_model_id's "
        "tokenizer) to show the baseline LLM (text_only_baseline only).",
    )

    @model_validator(mode="after")
    def _check_runner_specific_fields(self) -> "RunnerConfig":
        if self.type == "text_only_baseline":
            if self.checkpoint is not None:
                raise ValueError(
                    "text_only_baseline has no checkpoint; leave `checkpoint` unset."
                )
            if not self.response_model_id:
                raise ValueError("text_only_baseline requires `response_model_id`.")
            if not self.baseline_llm:
                raise ValueError("text_only_baseline requires `baseline_llm`.")
        else:
            if self.checkpoint is None:
                raise ValueError(f"runner.checkpoint is required for runner type {self.type!r}.")
        return self

    def identity(self) -> str:
        """Stable identity string used as the runner's checkpoint key.

        Flows through `ITMModel.checkpoint`, `_get_runner`'s cache key, and the
        Weave attributes — i.e. anything that compares "is this the same model".
        For runners with a real checkpoint file the identity is its path; for
        `text_only_baseline` it encodes the params that change the model's
        behaviour so a different baseline_llm / tail_tokens / response model
        becomes a different row on the leaderboard.
        """
        if self.type == "text_only_baseline":
            base_url = self.baseline_base_url or "default"
            return (
                f"text_only_baseline://{self.response_model_id}"
                f"|{self.baseline_llm}|t{self.tail_tokens}|{base_url}"
            )
        # Deliberately NOT expanded: the identity is what lands on the
        # leaderboard, and expanding here would make every machine's
        # PRISM_EVAL_CHECKPOINT_DIR a different row. resolve_checkpoint()
        # expands it at load time instead.
        return str(self.checkpoint)


def resolve_checkpoint(identity: str) -> str:
    """Expand a runner identity into a real filesystem path.

    Runner identities are stored unexpanded so they stay machine-independent
    (see `RunnerConfig.identity`). This turns one into something torch.load
    can open, and fails loudly rather than letting a literal '${VAR}' reach
    the filesystem as a confusing 'No such file' error.

    Pseudo-identities like 'text_only_baseline://…' pass through untouched.
    """
    if "://" in identity:
        return identity

    expanded = os.path.expanduser(os.path.expandvars(identity))

    unresolved = re.findall(r"\$\{?(\w+)\}?", expanded)
    if unresolved:
        raise ValueError(
            f"Checkpoint path {identity!r} references unset environment "
            f"variable(s): {', '.join(sorted(set(unresolved)))}. "
            f"Set PRISM_EVAL_CHECKPOINT_DIR to wherever you downloaded the "
            f"weights (see scripts/download_weights.py), or point "
            f"`runner.checkpoint` at an absolute path."
        )
    if not Path(expanded).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {expanded} (from {identity!r}). "
            f"Run `python scripts/download_weights.py` to fetch the published "
            f"checkpoints."
        )
    return expanded


class ScoringConfig(BaseModel):
    bertscore: bool = False
    judge_llm: bool = False
    judge_model: str | None = Field(
        default=None,
        description="Override PRISM_EVAL_MODEL env var for this experiment.",
    )
    judge_base_url: str | None = Field(
        default=None,
        description="Override PRISM_EVAL_BASE_URL env var for this experiment.",
    )

    # AdversarialDetectionScorer — runtime identifier that picks which GT
    # bullets are adversarial (AP injection / HO hidden objective), then
    # reuses the calibrated judge's per-bullet scores via _JUDGE_SCORES_CACHE.
    # Requires judge_llm=true (no cache otherwise). Default off so existing
    # configs are unaffected; enabling shifts the evaluation digest and
    # requires bumping leaderboard_name — see "Comparability" in CONTRIBUTING.md.
    adversarial_detection: bool = False
    adversarial_judge_model: str | None = Field(
        default=None,
        description="Judge model for the adversarial identifier. Defaults to "
        "`judge_model` (or PRISM_EVAL_MODEL). MUST match `judge_model` so the "
        "score-cache key (eval_id, model) is shared with JudgeLLMScorer.",
    )
    adversarial_judge_base_url: str | None = Field(
        default=None,
        description="Base URL for the identifier. Defaults to `judge_base_url`.",
    )


class AnnotationConfig(BaseModel):
    """Controls the dedicated root-trace pre-pass for human annotation.

    Weave's `Evaluation.predict_and_score` op is the only queueable call in
    the annotation UI today, but its input/output shape is owned by the
    Evaluation framework — annotators only see opaque level-1 keys like
    `inputs.example` and `outputs.scores`. To give annotators a clean,
    flat trace shaped exactly for annotation, `prism-eval evaluate` emits a
    parallel `itm_annotate` root op per row *before* running the
    Evaluation. The predict output is cached so the Evaluation's inner
    predict call reads from the cache — model inference runs exactly once
    per row.
    """

    emit_trace: bool = Field(
        default=True,
        description="If true, emit an `itm_annotate` root trace per row before the Evaluation runs.",
    )
    op_name: str = Field(
        default="itm_annotate",
        description="Op name for the annotation root trace. Annotators queue against this.",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        description=(
            "Batch size for the GPU stage of the annotate pre-pass. The runner's "
            "run_batch(records) is called once per batch; results pre-populate the "
            "predict cache so the per-row itm_annotate trace emission can fan out "
            "concurrently. Set to 1 for fully serial batches. "
            "8 is safe on RTX 6000 Pro (~25 GB on top of the loaded model)."
        ),
    )
    trace_workers: int = Field(
        default=16,
        ge=1,
        description=(
            "ThreadPool size for the parallel trace-recording stage. Capped per "
            "batch (workers = min(this, batch_size)) so we never spawn unused "
            "threads on the final partial batch. Trace recording is network-bound, "
            "so this can be > GPU batch size without contention."
        ),
    )
    max_batch_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Token budget per GPU batch for length-aware dynamic batching. When "
            "set (and the runner exposes prompt_token_len), records are sorted by "
            "prompt length and packed so that batch_records * max_prompt_len_in_batch "
            "<= this budget — short records still fill up to batch_size, while long "
            "outliers fall into small/singleton batches. This bounds the peak "
            "activation memory of the base-generation forward, which is what OOMs on "
            "high length-variance suites (e.g. XPIA: median ~350 tok, max ~16k). A "
            "record longer than the budget runs alone. Leave None for fixed-size "
            "slicing (legacy)."
        ),
    )


class EvaluationConfig(BaseModel):
    """Tunables for how `prism-eval evaluate` builds the `weave.Evaluation` and
    (optionally) publishes a Leaderboard.
    """

    evaluation_name: str | None = Field(
        default=None,
        description="Identity in Weave. Defaults to `{experiment.name}_v1`. "
        "Bump the suffix when the dataset content changes.",
    )
    publish_leaderboard: bool = Field(
        default=True,
        description="Create/update a Leaderboard pointing at this Evaluation.",
    )
    leaderboard_name: str | None = Field(
        default=None,
        description="Defaults to `{evaluation_name}_leaderboard`.",
    )


class ExperimentConfig(BaseModel):
    experiment: ExperimentMeta
    suite: SuiteConfig | None = None
    runner: RunnerConfig
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)

    @model_validator(mode="after")
    def _require_suite(self) -> "ExperimentConfig":
        if self.suite is None:
            raise ValueError("Config must define a `suite:` block.")
        return self

    def results_dir(self, root: Path = Path("results")) -> Path:
        return root / self.experiment.name

    def run_output(self, root: Path = Path("results")) -> Path:
        return self.results_dir(root) / "run.jsonl"

    def scored_output(self, root: Path = Path("results")) -> Path:
        return self.results_dir(root) / "scored.jsonl"

    def load_records(self):
        """Load (and dedupe + filter) records across all configured suite paths.

        Handles single-path and multi-path (`all` union) configs uniformly.
        Dedupes by `eval_id` so an eval that appears in two source files only
        contributes one row to the Evaluation.
        """
        from prism_eval.loader import filter_evals, load_suite

        records = []
        seen_ids: set[str] = set()
        for p in self.suite.paths:
            suite = load_suite(p)
            for r in suite.evals:
                if r.eval_id in seen_ids:
                    continue
                seen_ids.add(r.eval_id)
                records.append(r)
        if self.suite.settings or self.suite.tags:
            records = filter_evals(
                records,
                settings=self.suite.settings,
                tags=self.suite.tags,
            )
        if self.suite.per_setting_limit:
            counts: dict[str, int] = {}
            capped: list = []
            for r in records:
                limit = self.suite.per_setting_limit.get(r.setting)
                if limit is None or counts.get(r.setting, 0) < limit:
                    capped.append(r)
                    counts[r.setting] = counts.get(r.setting, 0) + 1
            records = capped
        return records

    def resolved_evaluation_name(self) -> str:
        """The Evaluation's identity in Weave — explicit override or `{experiment}_v1`."""
        return self.evaluation.evaluation_name or f"{self.experiment.name}_v1"

    def resolved_leaderboard_name(self) -> str:
        return self.evaluation.leaderboard_name or f"{self.resolved_evaluation_name()}_leaderboard"

    def weave_attributes(self) -> dict:
        """Attributes attached to every traced call in this experiment."""
        attrs: dict = {
            "experiment": self.experiment.name,
            "tags": self.experiment.tags,
            "runner": self.runner.type,
            "checkpoint": str(self.runner.checkpoint),
            "suite": ",".join(str(p) for p in self.suite.paths),
            "git_sha": _git_sha(),
        }
        if self.experiment.notes:
            attrs["notes"] = self.experiment.notes
        if self.scoring.judge_llm and self.scoring.judge_model:
            attrs["judge_model"] = self.scoring.judge_model
        return attrs


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text())
    return ExperimentConfig.model_validate(raw)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
