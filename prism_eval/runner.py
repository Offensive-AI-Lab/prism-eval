"""Runner protocol for ITM evaluation."""

from __future__ import annotations

from typing import Protocol

from prism_eval.schema import EvalRecord, EvalResult


class ITMRunner(Protocol):
    """Protocol that all ITM runners must implement."""

    def setup(self, checkpoint_path: str, device: str = "cuda") -> None:
        """Load model weights and prepare for inference."""
        ...

    def run_eval(self, eval_record: EvalRecord) -> EvalResult:
        """Run a single eval and return the ITM report."""
        ...
