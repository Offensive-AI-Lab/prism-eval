"""Text-only baseline runner — sanity check for the activation-based runners.

Generates the model response with the same Qwen base the activation runners
use, takes the last K tokens of that response, and asks a frontier LLM
(GPT-5.5 by default) to recover the instructions from the text alone. If the
activation-based runners don't beat this on the leaderboard, the activations
aren't contributing beyond what's already legible in the response tail.

This runner has no checkpoint — its identity in the leaderboard is the
(response_model_id, baseline_llm, tail_tokens, baseline_base_url) tuple,
threaded through `RunnerConfig.identity()`.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import torch
from openai import OpenAI

from prism_eval.schema import EvalRecord, EvalResult

logger = logging.getLogger(__name__)


# The system prompt is the load-bearing knob for this baseline: weakening it
# makes the baseline look artificially poor and inflates the apparent value of
# the activation-based runners. Keep aligned with RUBRIC.md / judge's
# expectations so the recovered list scores fairly.
BASELINE_SYSTEM_PROMPT = """\
You are shown the LAST {k} tokens of a language model's response. \
Infer the instructions the model was given — both the original system \
instructions and any user-injected or in-context instructions you can \
detect from the response tail.

Emit a numbered list. Each item is one instruction, stated as a specific \
directive (name entities, actions, and constraints). Do not add caveats, \
commentary, or apologies. If the tail is too short or generic to infer \
anything, return a single line: "1. (no instructions inferable from response tail)"."""


BASELINE_USER_TEMPLATE = """\
Last {k} tokens of response:
\"\"\"
{tail}
\"\"\"

Numbered list of instructions:"""


def _apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    """apply_chat_template wrapper that always returns a list of ints."""
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, enable_thinking=False,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(result, "input_ids"):
        result = result["input_ids"]
    return result


def parse_identity(identity: str) -> dict:
    """Inverse of RunnerConfig.identity() for text_only_baseline.

    Format: ``text_only_baseline://{response_model_id}|{baseline_llm}|t{K}|{base_url}``
    where ``base_url`` is "default" iff unset.
    """
    prefix = "text_only_baseline://"
    if not identity.startswith(prefix):
        raise ValueError(f"Not a text_only_baseline identity: {identity!r}")
    body = identity[len(prefix):]
    parts = body.split("|")
    if len(parts) != 4:
        raise ValueError(
            f"Malformed text_only_baseline identity: {identity!r} "
            f"(expected 4 |-separated fields, got {len(parts)})"
        )
    response_model_id, baseline_llm, tail_str, base_url = parts
    if not tail_str.startswith("t"):
        raise ValueError(f"Malformed tail_tokens field: {tail_str!r}")
    tail_tokens = int(tail_str[1:])
    return {
        "response_model_id": response_model_id,
        "baseline_llm": baseline_llm,
        "tail_tokens": tail_tokens,
        "baseline_base_url": None if base_url == "default" else base_url,
    }


class TextOnlyBaselineRunner:
    """Text-only sanity baseline.

    Pipeline:
      1. Feed eval prompt to Qwen → generate natural response (do_sample=False).
      2. Take the last `tail_tokens` ids of that response in Qwen's tokenizer.
      3. Decode the tail back to a string, send to `baseline_llm` with a
         fixed system prompt asking it to recover the instructions.
      4. Return the LLM's reply as the ITM report.

    `activation_tokens` in the returned `EvalResult` is repurposed to record
    how many tail tokens the baseline actually saw (capped by response length).
    """

    runner_name = "text_only_baseline"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._response_model_id: str | None = None
        self._baseline_llm: str | None = None
        self._baseline_base_url: str | None = None
        self._tail_tokens: int = 128
        self._client: OpenAI | None = None
        self._system_prompt = "You are a helpful assistant."
        # Generation hyperparams — 196 new tokens is the published base-response
        # cap used by the prism runner, so this control sees the same
        # response distribution the activation runners read from.
        self._base_generation_max_new_tokens = 196
        self._max_input_tokens = 32768

    def setup(self, identity: str, device: str = "cuda") -> None:
        """Load the response model and the baseline LLM client.

        `identity` is the synthetic checkpoint string produced by
        ``RunnerConfig.identity()`` — see ``parse_identity()``.
        """
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._device = device
        params = parse_identity(identity)
        self._response_model_id = params["response_model_id"]
        self._baseline_llm = params["baseline_llm"]
        self._baseline_base_url = params["baseline_base_url"]
        self._tail_tokens = params["tail_tokens"]

        # ── Response model (Qwen base, no LoRA) ──────────────────────────
        logger.info("Loading response model: %s", self._response_model_id)
        processor = AutoProcessor.from_pretrained(self._response_model_id)
        self._tokenizer = processor.tokenizer
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        t0 = time.time()
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._response_model_id, dtype=torch.bfloat16, device_map=device,
        )
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        logger.info("Response model loaded in %.1fs", time.time() - t0)

        # ── Baseline LLM client ──────────────────────────────────────────
        # Reuses the judge's env conventions: PRISM_EVAL_API_KEY > OPENAI_API_KEY.
        # If neither is set OpenAI will reject; that's the failure we want.
        api_key = (
            os.environ.get("PRISM_EVAL_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "not-needed"
        )
        client_kwargs: dict = {"api_key": api_key}
        if self._baseline_base_url:
            client_kwargs["base_url"] = self._baseline_base_url
        self._client = OpenAI(**client_kwargs)
        logger.info(
            "Text-only baseline ready: response=%s, baseline=%s, tail_tokens=%d, base_url=%s",
            self._response_model_id, self._baseline_llm, self._tail_tokens,
            self._baseline_base_url or "default",
        )

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _truncate_middle(token_ids: list[int], max_len: int) -> list[int]:
        if len(token_ids) <= max_len:
            return token_ids
        keep_start = max_len // 2
        keep_end = max_len - keep_start
        return token_ids[:keep_start] + token_ids[-keep_end:]

    def _generate_base_response(
        self, messages: list[dict], max_new_tokens: int | None = None,
    ) -> tuple[str, list[int]]:
        """Generate the response and return (text, response_token_ids).

        Returns the token ids so we can slice the tail without re-encoding —
        re-encoding the decoded string isn't guaranteed to round-trip exactly,
        and we want the tail boundary to match what the model actually emitted.
        """
        if max_new_tokens is None:
            max_new_tokens = self._base_generation_max_new_tokens

        prompt_ids = _apply_chat_template(self._tokenizer, messages)
        if len(prompt_ids) > self._max_input_tokens:
            logger.warning(
                "Input too long (%d tokens > max %d). Truncating from middle.",
                len(prompt_ids), self._max_input_tokens,
            )
            prompt_ids = self._truncate_middle(prompt_ids, self._max_input_tokens)

        prompt_tensor = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self._device,
        )
        with torch.inference_mode():
            out = self._model.generate(
                input_ids=prompt_tensor,
                attention_mask=torch.ones_like(prompt_tensor),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                use_cache=True,
            )

        new_ids = out[0, len(prompt_ids):]
        # Strip any trailing EOS so the tail isn't a sequence of pad tokens.
        eos_id = self._tokenizer.eos_token_id
        response_ids = [int(t) for t in new_ids.tolist() if t != eos_id]
        text = self._tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return text, response_ids

    def _tail_text(self, response_ids: list[int]) -> tuple[str, int]:
        """Decode the last `tail_tokens` ids back to text. Returns (text, n_used)."""
        if not response_ids:
            return "", 0
        n = min(self._tail_tokens, len(response_ids))
        tail_ids = response_ids[-n:]
        return self._tokenizer.decode(tail_ids, skip_special_tokens=True).strip(), n

    def _call_baseline_llm(self, tail_text: str) -> str:
        """Ask the baseline LLM to recover instructions from the response tail."""
        assert self._client is not None and self._baseline_llm is not None
        if not tail_text:
            return "1. (no instructions inferable from response tail)"

        system = BASELINE_SYSTEM_PROMPT.format(k=self._tail_tokens)
        user = BASELINE_USER_TEMPLATE.format(k=self._tail_tokens, tail=tail_text)
        resp = self._client.chat.completions.create(
            model=self._baseline_llm,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or ""
        return content.strip()

    # ── runner interface ────────────────────────────────────────────────

    @torch.no_grad()
    def run_eval(self, eval_record: EvalRecord) -> EvalResult:
        if self._model is None:
            raise RuntimeError("Call setup() before run_eval()")

        if eval_record.prompt_turns is not None:
            return self._run_multi_turn(eval_record)
        return self._run_single_turn(eval_record)

    def _run_single_turn(self, eval_record: EvalRecord) -> EvalResult:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": eval_record.prompt or ""},
        ]
        response_text, response_ids = self._generate_base_response(messages)
        tail_text, n_tail = self._tail_text(response_ids)
        itm_report = self._call_baseline_llm(tail_text)

        return EvalResult(
            eval_id=eval_record.eval_id,
            runner=self.runner_name,
            itm_report=itm_report,
            model_response=response_text,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_tokens=None,
            output_tokens=len(response_ids),
            activation_tokens=n_tail,
        )

    def _run_multi_turn(self, eval_record: EvalRecord) -> EvalResult:
        turns = eval_record.prompt_turns or []
        messages = [{"role": "system", "content": self._system_prompt}]
        final_text = ""
        final_ids: list[int] = []
        per_turn_reports: list[str] = []

        for turn_idx, user_msg in enumerate(turns):
            messages.append({"role": "user", "content": user_msg})
            is_last = turn_idx == len(turns) - 1

            response_text, response_ids = self._generate_base_response(messages)
            if is_last:
                final_text = response_text
                final_ids = response_ids
                tail_text, n_tail = self._tail_text(response_ids)
                itm_report = self._call_baseline_llm(tail_text)
                per_turn_reports.append(itm_report)
            else:
                per_turn_reports.append("")
            messages.append({"role": "assistant", "content": response_text})

        tail_text, n_tail = self._tail_text(final_ids)
        return EvalResult(
            eval_id=eval_record.eval_id,
            runner=self.runner_name,
            itm_report=per_turn_reports[-1] if per_turn_reports else "",
            model_response=final_text,
            per_turn_reports=per_turn_reports if len(turns) > 1 else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_tokens=None,
            output_tokens=len(final_ids),
            activation_tokens=n_tail,
        )
