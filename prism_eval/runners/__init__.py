"""ITM runner registry.

Single source of truth for `runner.type` → runner class. Both the Weave path
(`weave_eval._get_runner`) and the plain-JSONL path (`cli.run`) resolve through
`build_runner`, so adding a runner means touching this file and nothing else.

Imports are deferred into the factory: every runner pulls in torch and
transformers, and `prism-eval --help` should not pay for that.
"""

from __future__ import annotations

from prism_eval.runner import ITMRunner

#: Runner type names accepted in `runner.type` and by `prism-eval run --runner`.
#: Keep in sync with the Literal on `RunnerConfig.type` in prism_eval/config.py.
RUNNER_TYPES: tuple[str, ...] = (
    "prism",
    "text_only_baseline",
)


def build_runner(runner_type: str) -> ITMRunner:
    """Instantiate a runner by type name. Does not call `setup`."""
    if runner_type == "prism":
        from prism_eval.runners.prism import PrismRunner

        return PrismRunner()
    if runner_type == "text_only_baseline":
        from prism_eval.runners.text_only_baseline import TextOnlyBaselineRunner

        return TextOnlyBaselineRunner()
    raise ValueError(
        f"Unknown runner_type: {runner_type!r}. Expected one of {RUNNER_TYPES}."
    )
