"""Encoder-decoder ITM runner.

Loads a finetuned checkpoint (Qwen + LoRA + optional encoder + projection),
runs each eval prompt through the base model, extracts activations at a
specific layer, and generates an ITM report via LoRA-enabled decoding with
soft-token prepending.

Pipeline: hook the frozen target model at `hook_layer`, take the last
`max_act_tokens` response-token activations, pass them through the trained
linear projection and a norm match into embedding space, prepend them as soft
tokens, and decode the instruction list with LoRA enabled. A checkpoint is just
that projection plus the LoRA adapters — the target model is never modified.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone

import torch

from prism_eval.schema import EvalRecord, EvalResult

logger = logging.getLogger(__name__)


def _env_int(name: str, fallback: int) -> int:
    """Ablation override: read an int from the environment when set.

    The runner's _cfg comes from the training checkpoint, so window-ablation
    runs (which must not touch weave_eval.py or the checkpoints) override
    generation/window params via env vars instead.
    """
    val = os.environ.get(name)
    return int(val) if val else fallback


def _act_window_start(prompt_len: int, resp_len: int, n_act: int, pos: str) -> int:
    """Start index (into the full token sequence) of the activation window.

    pos: 'end' (default — last n_act response tokens), 'start', or 'middle'.
    When resp_len <= n_act all three positions coincide.
    """
    if pos == "start":
        return prompt_len
    if pos == "middle":
        return prompt_len + (resp_len - n_act) // 2
    if pos != "end":
        raise ValueError(f"Unknown activation window position: {pos!r}")
    return prompt_len + resp_len - n_act


def _resolve_act_window(
    prompt_len: int, resp_len: int, max_act: int, pos: str
) -> tuple[int, int]:
    """(start, n_act) of the activation window in the full token sequence.

    pos 'chunkK' selects the K-th consecutive max_act-token slice of the
    response (chunk0 = first max_act tokens). n_act is 0 when the response
    ends before the chunk starts — callers must handle the empty window.
    """
    if pos.startswith("chunk"):
        k = int(pos[len("chunk"):])
        offset = k * max_act
        n_act = max(0, min(max_act, resp_len - offset))
        return prompt_len + offset, n_act
    n_act = min(resp_len, max_act)
    return _act_window_start(prompt_len, resp_len, n_act, pos), n_act


_ACT_CONTEXT_MODES = ("full", "masked_prompt", "masked_user", "evicted", "swapped")


def _act_context_mode() -> str:
    """Ablation override: what the activation-extraction forward pass may see.

    Reviewer concern: response-position activations may just carry prompt
    content routed in by attention at extraction time. These modes vary only
    the extraction pass — the base response is generated (and cached) with
    full context in every mode, so behavior is held fixed.

      full          — prompt + response (default, published behavior)
      masked_prompt — same tokens, attention mask zeroed over system + user
      masked_user   — same tokens, attention mask zeroed over the user turn only
      evicted       — response re-forwarded behind a neutral placeholder user
                      turn (simulates the prompt scrolling out of the window)
      swapped       — response re-forwarded behind a *donor* prompt from
                      PRISM_EVAL_ACT_SWAP_FILE (keyed by eval_id); directional
                      test of where the reported instruction comes from
    """
    mode = os.environ.get("PRISM_EVAL_ACT_CONTEXT") or "full"
    if mode not in _ACT_CONTEXT_MODES:
        raise ValueError(
            f"Unknown PRISM_EVAL_ACT_CONTEXT: {mode!r} (expected one of {_ACT_CONTEXT_MODES})"
        )
    return mode


def _act_context_mask_span(mode: str, prompt_len: int, sys_len: int) -> tuple[int, int] | None:
    """Half-open [start, end) span of tokens to hide from attention, or None.

    masked_prompt hides everything up to the response (system + user +
    generation prompt); masked_user keeps the system portion visible.
    """
    if mode == "masked_prompt":
        return (0, prompt_len)
    if mode == "masked_user":
        return (min(sys_len, prompt_len), prompt_len)
    return None


def _longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the shared token prefix of two id sequences."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _build_extraction_messages(
    messages_with_response: list[dict], replacement_user: str
) -> list[dict]:
    """Conversation for an evicted/swapped extraction pass.

    Keeps the system turn (when present) and the final assistant response;
    every turn in between collapses into a single replacement user turn, as
    if the original context had scrolled out of the window.
    """
    if not messages_with_response or messages_with_response[-1]["role"] != "assistant":
        raise ValueError("messages_with_response must end with the assistant response")
    out: list[dict] = []
    if messages_with_response[0]["role"] == "system":
        out.append(messages_with_response[0])
    out.append({"role": "user", "content": replacement_user})
    out.append(messages_with_response[-1])
    return out


_SWAP_PROMPTS: dict[str, str] | None = None


def _swap_prompt_for(eval_id: str) -> str:
    """Donor user prompt for a swapped extraction pass.

    Fails loud on a missing file or eval_id — silently falling back to full
    context would poison the ablation row.
    """
    global _SWAP_PROMPTS
    if _SWAP_PROMPTS is None:
        path = os.environ.get("PRISM_EVAL_ACT_SWAP_FILE")
        if not path:
            raise RuntimeError(
                "PRISM_EVAL_ACT_CONTEXT=swapped requires PRISM_EVAL_ACT_SWAP_FILE "
                "(see scripts/make_act_swap_pairs.py)"
            )
        with open(path, encoding="utf-8") as fh:
            pairs = json.load(fh)["pairs"]
        _SWAP_PROMPTS = {eid: rec["donor_prompt"] for eid, rec in pairs.items()}
    try:
        return _SWAP_PROMPTS[eval_id]
    except KeyError:
        raise KeyError(
            f"eval_id {eval_id!r} missing from PRISM_EVAL_ACT_SWAP_FILE"
        ) from None


_BASE_RESPONSE_CACHE: dict[str, str] | None = None


def _base_response_cache_path() -> str | None:
    """On-disk cache of greedy base responses (opt-in via env, off by default).

    Base responses depend only on (base model, messages, generation cap) —
    NOT on the ITM checkpoint or activation window — so ablation runs that
    differ only in how activations are read can reuse them instead of
    regenerating identical text. JSONL of {"key": sha256, "response": str}.
    """
    return os.environ.get("PRISM_EVAL_BASE_RESPONSE_CACHE") or None


def _base_response_key(model_id: str, messages: list[dict], max_new_tokens: int) -> str:
    payload = json.dumps(
        [model_id, messages, max_new_tokens], sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_base_response_cache() -> dict[str, str]:
    global _BASE_RESPONSE_CACHE
    if _BASE_RESPONSE_CACHE is None:
        cache: dict[str, str] = {}
        path = _base_response_cache_path()
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        cache[rec["key"]] = rec["response"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # torn line from an interrupted write
        _BASE_RESPONSE_CACHE = cache
    return _BASE_RESPONSE_CACHE


def _append_base_responses(entries: list[tuple[str, str]]) -> None:
    path = _base_response_cache_path()
    if not path or not entries:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for key, response in entries:
            fh.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")


def _apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    """apply_chat_template wrapper that always returns a list of ints."""
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, enable_thinking=False,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(result, "input_ids"):
        result = result["input_ids"]
    return result


def _left_pad(
    seqs: list[list[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Left-pad a list of token-id sequences for batched generation.

    Left-padding is required because HF generate() expects the prompt to be
    at the right edge of the input — every sample's first generated token
    is appended after position max_len.

    Returns (input_ids, attn_mask, lens).
    """
    max_len = max(len(s) for s in seqs)
    pad_lens = [max_len - len(s) for s in seqs]
    padded = [[pad_id] * pl + s for s, pl in zip(seqs, pad_lens)]
    mask = [[0] * pl + [1] * (max_len - pl) for pl in pad_lens]
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
        [len(s) for s in seqs],
    )


class PrismRunner:
    """Runs evals through the encoder-decoder ITM pipeline.

    Pipeline:
      1. Feed eval prompt to Qwen (LoRA OFF) → generate natural response
      2. Forward pass on full conversation with hook → capture activations at hook_layer
      4. Projection: activations → linear projection → norm match to embedding space
      5. Decode: soft tokens prepended to decoder prompt, LoRA ON → ITM report
    """

    runner_name = "prism"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._projection = None
        self._target_norm = None
        self._act_store = None
        self._hook_handle = None
        self._cfg = None
        self._device = "cpu"
        self._profile = None
        self._gen_eos_id = None

    def setup(self, checkpoint_path: str, device: str = "cuda") -> None:
        """Load the PRISM checkpoint."""
        from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from prism_eval.runners.models import (
            build_projection,
            compute_target_norm,
            get_subject_profile,
            register_hook,
        )

        self._device = device
        logger.info("Loading checkpoint from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self._cfg = ckpt["config"]
        model_id = self._cfg["model_id"]

        # Target-model profile (qwen default reproduces original behaviour)
        profile = get_subject_profile(self._cfg)
        self._profile = profile
        logger.info("Subject profile: %s", profile["name"])

        # Tokenizer
        if profile["tokenizer_via_processor"]:
            processor = AutoProcessor.from_pretrained(model_id)
            self._tokenizer = processor.tokenizer
        else:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Subject base model
        logger.info("Loading %s ...", model_id)
        t0 = time.time()
        if profile["load_class"] == "causal_lm":
            from transformers import AutoModelForCausalLM
            model_cls = AutoModelForCausalLM
        else:
            model_cls = AutoModelForImageTextToText
        model_kwargs: dict = dict(dtype=torch.bfloat16, device_map=device)
        if profile["attn_implementation"] is not None:
            model_kwargs["attn_implementation"] = profile["attn_implementation"]
        self._model = model_cls.from_pretrained(model_id, **model_kwargs)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        logger.info("Target model loaded in %.1fs", time.time() - t0)

        # Generation stop token: gemma-2 turns end at <end_of_turn>, not eos.
        self._gen_eos_id = self._tokenizer.eos_token_id
        if profile["turn_end_token"] is not None:
            tid = self._tokenizer.convert_tokens_to_ids(profile["turn_end_token"])
            if tid is not None and tid != self._tokenizer.unk_token_id:
                self._gen_eos_id = tid
        logger.info("Generation eos id: %s", self._gen_eos_id)

        # LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self._cfg["lora_r"],
            lora_alpha=self._cfg["lora_alpha"],
            lora_dropout=0.0,
            target_modules=self._cfg["lora_target_modules"],
        )
        self._model = get_peft_model(self._model, lora_config)
        set_peft_model_state_dict(self._model, ckpt["lora_state"])
        self._model.eval()
        logger.info("LoRA weights loaded")

        # Hook at the configured layer
        hook_layer = self._cfg["hook_layer"]
        base_model = self._model.base_model.model
        self._act_store, self._hook_handle = register_hook(base_model, hook_layer)

        # Projection (optional). The arch is read from the checkpoint config;
        # every released checkpoint uses the linear projection.
        use_projection = self._cfg.get("_use_projection", True)
        proj_state = ckpt.get("projection_state")
        if use_projection and proj_state is not None:
            proj_dim = self._cfg.get("projection_dim", 4096)
            proj_arch = self._cfg.get("projection_arch", "linear")
            bottleneck_dim = self._cfg.get("projection_bottleneck_dim", 1024)
            self._projection = build_projection(proj_arch, proj_dim, bottleneck_dim=bottleneck_dim)
            self._projection.load_state_dict(proj_state)
            self._projection.to(device, dtype=torch.bfloat16)
            self._projection.eval()
            logger.info("Projection layer loaded (arch=%s)", proj_arch)
        else:
            self._projection = None

        # Target norm for embedding-space matching.
        #
        # embed_scale: gemma-2 scales embeddings by sqrt(hidden_size). WHERE
        # that happens moved across transformers versions: <=5.3 multiplied
        # inputs_embeds inside model.forward (soft tokens and text embeds
        # scaled together — what the checkpoints were trained with); >=5.5
        # scales at lookup inside Gemma2TextScaledWordEmbedding, so
        # user-passed soft tokens are NOT rescaled. Measure the module's
        # output/weight ratio and fold it into the target norm so soft
        # tokens always land at the same effective scale as text embeds
        # (ratio == 1.0 for qwen/ministral — unchanged behaviour).
        emb_module = self._model.get_input_embeddings()
        with torch.no_grad():
            probe_ids = torch.arange(16, device=device).unsqueeze(0)
            out_norms = emb_module(probe_ids)[0].float().norm(dim=-1)
            row_norms = emb_module.weight[probe_ids[0]].float().norm(dim=-1)
            self._embed_scale = (
                out_norms.sum() / row_norms.sum().clamp_min(1e-8)
            ).item()
        logger.info("Embedding output/weight scale ratio: %.4f", self._embed_scale)
        self._target_norm = (
            compute_target_norm(self._model).to(device) * self._embed_scale
        )
        logger.info(
            "Model loaded: %s, hook_layer=%d, encoder=%s, projection=%s, target_norm=%.4f",
            model_id, hook_layer,
            "yes" if self._projection else "no",
            self._target_norm.item(),
        )

    def _generate_base_response(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        """Generate Qwen's response with LoRA disabled."""
        if max_new_tokens is None:
            max_new_tokens = _env_int(
                "PRISM_EVAL_BASE_GEN_MAX_NEW_TOKENS",
                self._cfg.get("base_generation_max_new_tokens", 196),
            )

        cache_key = None
        if _base_response_cache_path():
            cache_key = _base_response_key(self._cfg["model_id"], messages, max_new_tokens)
            cached = _load_base_response_cache().get(cache_key)
            if cached is not None:
                return cached

        prompt_ids = _apply_chat_template(self._tokenizer, messages)

        # Safety check: warn if input is very long
        max_input = self._cfg.get("max_input_tokens", 32768)
        if len(prompt_ids) > max_input:
            logger.warning(
                "Input too long (%d tokens > max %d). Truncating from the middle "
                "to preserve instruction at edges.",
                len(prompt_ids), max_input,
            )
            prompt_ids = self._truncate_middle(prompt_ids, max_input)

        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)

        self._model.disable_adapter_layers()
        with torch.inference_mode():
            out = self._model.generate(
                input_ids=prompt_tensor,
                attention_mask=torch.ones_like(prompt_tensor),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._gen_eos_id,
                use_cache=True,
            )
        self._model.enable_adapter_layers()

        new_ids = out[0, len(prompt_ids):]
        response = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if cache_key is not None:
            _load_base_response_cache()[cache_key] = response
            _append_base_responses([(cache_key, response)])
        return response

    @staticmethod
    def _truncate_middle(token_ids: list[int], max_len: int) -> list[int]:
        """Truncate from the middle to preserve both edges.

        Keeps the first half and last half of the allowed budget,
        so instructions at depth 0% or 100% survive truncation.
        """
        if len(token_ids) <= max_len:
            return token_ids
        keep_start = max_len // 2
        keep_end = max_len - keep_start
        return token_ids[:keep_start] + token_ids[-keep_end:]

    def _extract_activations(
        self,
        messages_with_response: list[dict],
        messages_prompt_only: list[dict],
        max_act_tokens: int = 128,
        eval_id: str | None = None,
    ) -> tuple[torch.Tensor, int, int]:
        """Extract activations from the assistant response portion.

        PRISM_EVAL_ACT_CONTEXT (see _act_context_mode) controls what this
        forward pass may see; the base response itself is never affected.

        Returns: ([1, n_act, D] tensor, input_token_count, activation_token_count).
        """
        from prism_eval.runners.models import _EarlyExit

        mode = _act_context_mode()
        if mode in ("evicted", "swapped"):
            if mode == "swapped":
                if eval_id is None:
                    raise ValueError("PRISM_EVAL_ACT_CONTEXT=swapped requires eval_id")
                replacement = _swap_prompt_for(eval_id)
            else:
                replacement = os.environ.get("PRISM_EVAL_ACT_EVICT_PLACEHOLDER", "...")
            messages_with_response = _build_extraction_messages(
                messages_with_response, replacement,
            )
            messages_prompt_only = messages_with_response[:-1]

        full_ids = _apply_chat_template(
            self._tokenizer, messages_with_response, add_generation_prompt=False,
        )
        prefix_ids = _apply_chat_template(self._tokenizer, messages_prompt_only)

        total_len = len(full_ids)
        prompt_len = len(prefix_ids)
        resp_len = total_len - prompt_len
        n_act = min(resp_len, max_act_tokens)

        # Safety check: truncate if too long for a single forward pass
        max_input = self._cfg.get("max_input_tokens", 32768)
        if total_len > max_input:
            logger.warning(
                "Activation extraction: input too long (%d tokens > max %d). "
                "Truncating prompt portion from the middle.",
                total_len, max_input,
            )
            # Keep the response tokens intact, truncate the prompt portion
            prompt_budget = max_input - resp_len
            if prompt_budget < 64:
                prompt_budget = 64
            truncated_prefix = self._truncate_middle(full_ids[:prompt_len], prompt_budget)
            full_ids = truncated_prefix + full_ids[prompt_len:]
            total_len = len(full_ids)
            prompt_len = len(truncated_prefix)
            resp_len = total_len - prompt_len
            n_act = min(resp_len, max_act_tokens)

        # Masked modes: hide the prompt from attention so response tokens can
        # only attend to each other. Position ids follow the model's own
        # handling of the mask (Qwen derives them from the mask cumsum), so
        # masked response tokens also sit where they would after a real
        # overflow — which is the semantics we want.
        #
        # masked_user keeps the system portion visible. Chat templates refuse
        # to render a conversation without a user turn, so the system-block
        # boundary is found by re-templating with probe user content and
        # taking the longest common token prefix — divergence starts exactly
        # where user content begins (the visible remainder is inert turn
        # framing).
        sys_len = 0
        if mode == "masked_user":
            probe = [dict(m) for m in messages_prompt_only]
            for m in reversed(probe):
                if m["role"] == "user":
                    m["content"] = "␀ITM-PROBE␀"
                    break
            probe_ids = _apply_chat_template(self._tokenizer, probe)
            sys_len = _longest_common_prefix_len(prefix_ids, probe_ids)
        mask_span = _act_context_mask_span(mode, prompt_len, sys_len)

        # Forward pass with hook active (LoRA OFF)
        self._model.disable_adapter_layers()
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self._device)
        attention_mask = torch.ones_like(input_ids)
        if mask_span is not None:
            attention_mask[0, mask_span[0]:mask_span[1]] = 0

        self._act_store["active"] = True
        with torch.inference_mode():
            try:
                self._model(input_ids=input_ids, attention_mask=attention_mask)
            except _EarlyExit:
                pass
        self._act_store["active"] = False
        self._model.enable_adapter_layers()

        hidden = self._act_store["hidden"]  # [1, seq_len, D]

        pos = os.environ.get("PRISM_EVAL_ACT_WINDOW_POS", "end")
        start, n_act = _resolve_act_window(prompt_len, resp_len, max_act_tokens, pos)

        if n_act > 0:
            acts = hidden[0, start:start + n_act, :].unsqueeze(0)  # [1, n_act, D]
        else:
            acts = hidden[0, :0, :].unsqueeze(0)  # [1, 0, D]

        return acts, prompt_len, n_act

    def _generate_itm_report(
        self,
        activations: torch.Tensor,
        max_new_tokens: int = 256,
    ) -> str:
        """Project → norm match → generate the instruction list with LoRA ON."""
        from prism_eval.runners.models import norm_match

        with torch.no_grad():
            soft = activations.to(dtype=torch.bfloat16)

            # Projection (if present)
            if self._projection is not None:
                soft = self._projection(soft)

            # Norm match to decoder embedding space
            soft = norm_match(soft, self._target_norm)

            # Build decoder prompt
            skip_prompt_b = self._cfg.get("skip_prompt_b", False)
            decoder_prompt = "" if skip_prompt_b else self._cfg.get("decoder_user_prompt", "")
            prefix_ids = _apply_chat_template(
                self._tokenizer,
                self._adapt_messages([{"role": "user", "content": decoder_prompt}]),
            )

            prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=self._device)
            prefix_emb = self._model.get_input_embeddings()(prefix_tensor)

            # Prepend soft tokens to decoder prompt embeddings
            inputs_embeds = torch.cat([soft, prefix_emb], dim=1)
            full_mask = torch.ones(
                1, inputs_embeds.size(1), device=self._device, dtype=torch.long,
            )

            gen_ids = self._model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self._gen_eos_id,
                pad_token_id=self._tokenizer.eos_token_id,
                use_cache=True,
            )

        return self._tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def run_eval(self, eval_record: EvalRecord) -> EvalResult:
        """Run a single eval through the encoder-decoder ITM pipeline."""
        if self._model is None:
            raise RuntimeError("Call setup() before run_eval()")

        max_act_tokens = _env_int("PRISM_EVAL_MAX_ACT_TOKENS", self._cfg.get("max_act_tokens", 128))
        oracle_max_new_tokens = self._cfg.get("eval_generation_max_new_tokens", 256)
        system_prompt = self._cfg.get("system_prompt", "You are a helpful assistant.")

        # Determine if single-turn or multi-turn. prompt_messages records are
        # single-generation conversations and take the single-turn path.
        if eval_record.prompt_turns is not None and eval_record.prompt_messages is None:
            return self._run_multi_turn(
                eval_record, system_prompt, max_act_tokens, oracle_max_new_tokens,
            )
        else:
            return self._run_single_turn(
                eval_record, system_prompt, max_act_tokens, oracle_max_new_tokens,
            )

    def _build_messages(
        self, eval_record: EvalRecord, system_prompt: str
    ) -> list[dict]:
        """Construct the chat-message list for a single-turn / split record.

        A record may carry its own structured conversation (``prompt_messages``
        — e.g. XPIA rows that land untrusted text in a ``tool``/document turn,
        separated from the legit user task). When present we use it verbatim:
        it already includes its own system turn, and ``apply_chat_template``
        emits the correct role tokens (plus tool tags) for it. Otherwise fall
        back to the classic ``[system, user]`` pair. Activation extraction is
        token-diff based and role-agnostic, so a tool/user final turn is fine.
        """
        if eval_record.prompt_messages:
            return self._adapt_messages([dict(m) for m in eval_record.prompt_messages])
        return self._adapt_messages([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": eval_record.prompt or ""},
        ])

    def _adapt_messages(self, messages: list[dict]) -> list[dict]:
        """Adapt a message list to the target model's chat-template constraints.

        gemma-2's template raises on a ``system`` role, so for profiles with
        ``no_system_role`` each system message is folded into the next user
        message (``"{system}\\n\\n{user}"``) — content is preserved, only the
        role framing is lost. All other profiles pass through unchanged.
        """
        if self._profile is None:
            return messages
        # Ministral: training wrapped every template call in an EMPTY system
        # message (suppresses the tokenizer's ~530-token default system
        # prompt). Mirror it whenever no system turn is present.
        ov = self._profile.get("system_message_override")
        if ov is not None and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": ov}] + list(messages)
        if not self._profile.get("no_system_role"):
            return messages
        out: list[dict] = []
        pending_system: list[str] = []
        for m in messages:
            if m.get("role") == "system":
                if m.get("content"):
                    pending_system.append(m["content"])
                continue
            if pending_system and m.get("role") == "user":
                m = dict(m)
                m["content"] = "\n\n".join(pending_system + [m.get("content") or ""])
                pending_system = []
            out.append(m)
        if pending_system:  # system-only prompt: degrade to a user turn
            out.append({"role": "user", "content": "\n\n".join(pending_system)})
        return out

    def prompt_token_len(self, eval_record: EvalRecord) -> int:
        """Chat-template-applied prompt length in tokens for a single record.

        Used by the annotate pre-pass for length-aware (token-budget) batching:
        peak memory of the base-generation forward scales with
        ``batch_records * max_prompt_len_in_batch`` (left-padding), so the
        scheduler needs each record's real templated length to bound it. Mirrors
        exactly what ``_batched_generate_base`` tokenizes (``_build_messages`` →
        ``_apply_chat_template``) so the estimate matches what actually runs.
        """
        system_prompt = self._cfg.get("system_prompt", "You are a helpful assistant.")
        messages = self._build_messages(eval_record, system_prompt)
        return len(_apply_chat_template(self._tokenizer, messages))

    def _run_single_turn(
        self,
        eval_record: EvalRecord,
        system_prompt: str,
        max_act_tokens: int,
        oracle_max_new_tokens: int,
    ) -> EvalResult:
        """Single-turn: generate response, extract activations, produce ITM report."""
        messages = self._build_messages(eval_record, system_prompt)

        # Step 1: Generate base response (LoRA OFF)
        model_response = self._generate_base_response(messages)

        # Step 2: Extract activations from response
        messages_with_resp = messages + [{"role": "assistant", "content": model_response}]
        activations, input_tokens, act_tokens = self._extract_activations(
            messages_with_resp, messages, max_act_tokens=max_act_tokens,
            eval_id=eval_record.eval_id,
        )

        # Step 3: Generate ITM report (LoRA ON, soft-token prepending).
        # Empty activation window (e.g. a chunk beyond the response end) →
        # empty report; see _run_batch_single_turn.
        if act_tokens > 0:
            itm_report = self._generate_itm_report(activations, max_new_tokens=oracle_max_new_tokens)
        else:
            itm_report = ""

        output_tokens = len(self._tokenizer.encode(model_response, add_special_tokens=False))

        return EvalResult(
            eval_id=eval_record.eval_id,
            runner=self.runner_name,
            itm_report=itm_report,
            model_response=model_response,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            activation_tokens=act_tokens,
        )

    def _run_multi_turn(
        self,
        eval_record: EvalRecord,
        system_prompt: str,
        max_act_tokens: int,
        oracle_max_new_tokens: int,
    ) -> EvalResult:
        """Multi-turn: build conversation incrementally, extract from final turn."""
        turns = eval_record.prompt_turns or []

        # Canonical history keeps the system turn; _adapt_messages folds it
        # into the first user turn at generation time for no-system profiles.
        messages = [{"role": "system", "content": system_prompt}]
        per_turn_reports: list[str] = []
        input_tokens = 0
        act_tokens = 0

        for turn_idx, user_msg in enumerate(turns):
            messages.append({"role": "user", "content": user_msg})
            is_last = turn_idx == len(turns) - 1

            # Generate base response for this turn
            adapted = self._adapt_messages(messages)
            model_response = self._generate_base_response(adapted)
            messages_with_resp = adapted + [{"role": "assistant", "content": model_response}]

            if is_last:
                # Extract activations from final turn's response
                activations, input_tokens, act_tokens = self._extract_activations(
                    # `adapted`, not `messages`: for no-system model profiles
                    # _adapt_messages folds the system turn into the first user
                    # turn, so the prompt-only reference must match the sequence
                    # actually tokenised or the activation window is offset.
                    # eval_id is what the `swapped` act-context ablation uses to
                    # look up this record's donor prompt.
                    messages_with_resp, adapted, max_act_tokens=max_act_tokens,
                    eval_id=eval_record.eval_id,
                )
                itm_report = self._generate_itm_report(
                    activations, max_new_tokens=oracle_max_new_tokens,
                )
                per_turn_reports.append(itm_report)
            else:
                per_turn_reports.append("")

            # Keep response in conversation history
            messages.append({"role": "assistant", "content": model_response})

        output_tokens = len(self._tokenizer.encode(model_response, add_special_tokens=False))

        return EvalResult(
            eval_id=eval_record.eval_id,
            runner=self.runner_name,
            itm_report=itm_report,
            model_response=model_response,
            per_turn_reports=per_turn_reports if len(turns) > 1 else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            activation_tokens=act_tokens,
        )

    @torch.no_grad()
    def run_batch(self, eval_records: list[EvalRecord]) -> list[EvalResult]:
        """Process a batch of evals through the ED pipeline with batched generation.

        Single-turn records are batched: the two autoregressive ``generate()``
        calls (base response + ITM report) process B samples per forward step
        instead of one. Activation extraction stays serial (one forward pass
        per sample; cheap; avoids per-sample masking gymnastics on the hook).

        Multi-turn records fall back to ``run_eval`` per record since the
        conversation has to be built incrementally; they're emitted in the
        result list in the same position the caller supplied.

        Empirically ~5-7× wall-time reduction vs sequential ``run_eval`` for
        prism on RTX 6000 Pro at B=8.
        """
        if self._model is None:
            raise RuntimeError("Call setup() before run_batch()")
        if not eval_records:
            return []

        # Split into single-turn (batchable) and multi-turn (sequential fallback).
        # We process single-turn records together for speed; multi-turn records
        # go through run_eval one at a time. Results are reassembled in order.
        single_idx: list[int] = []
        multi_idx: list[int] = []
        for i, rec in enumerate(eval_records):
            # prompt_messages records are single-generation conversations →
            # batched single-turn path, not the sequential multi-turn fallback.
            if rec.prompt_turns is not None and rec.prompt_messages is None:
                multi_idx.append(i)
            else:
                single_idx.append(i)

        results: list[EvalResult | None] = [None] * len(eval_records)

        # Multi-turn: per-record loop (rare in the shipped single-turn suites we
        # benchmark, so the cost of this fallback is negligible in practice).
        for i in multi_idx:
            results[i] = self.run_eval(eval_records[i])

        if single_idx:
            single_records = [eval_records[i] for i in single_idx]
            single_results = self._run_batch_single_turn(single_records)
            for i, res in zip(single_idx, single_results):
                results[i] = res

        assert all(r is not None for r in results)
        return [r for r in results if r is not None]  # type: ignore[misc]

    def _run_batch_single_turn(
        self, records: list[EvalRecord]
    ) -> list[EvalResult]:
        """Batched single-turn pipeline. Helpers lifted from
        scripts/run_mc_offline_batched.py (which now delegates here)."""
        from prism_eval.runners.models import norm_match

        max_act_tokens = _env_int("PRISM_EVAL_MAX_ACT_TOKENS", self._cfg.get("max_act_tokens", 128))
        oracle_max_new = self._cfg.get("eval_generation_max_new_tokens", 256)
        base_max_new = _env_int(
            "PRISM_EVAL_BASE_GEN_MAX_NEW_TOKENS",
            self._cfg.get("base_generation_max_new_tokens", 196),
        )
        system_prompt = self._cfg.get("system_prompt", "You are a helpful assistant.")

        # Build messages list per sample (honors prompt_messages split records).
        all_messages: list[list[dict]] = [
            self._build_messages(rec, system_prompt) for rec in records
        ]

        # Step 1: batched base response (LoRA OFF), through the optional cache.
        base_responses = self._cached_batched_generate_base(all_messages, base_max_new)

        # Step 2: serial activation extraction.
        softs: list[torch.Tensor] = []
        n_acts: list[int] = []
        input_tokens_list: list[int] = []
        for i, rec in enumerate(records):
            messages_with_resp = all_messages[i] + [
                {"role": "assistant", "content": base_responses[i]},
            ]
            acts, prompt_len, n_act = self._extract_activations(
                messages_with_resp, all_messages[i], max_act_tokens=max_act_tokens,
                eval_id=rec.eval_id,
            )
            with torch.no_grad():
                soft = acts.to(dtype=torch.bfloat16)
                if n_act > 0:
                    if self._projection is not None:
                        soft = self._projection(soft)
                    soft = norm_match(soft, self._target_norm)
            softs.append(soft)
            n_acts.append(n_act)
            input_tokens_list.append(prompt_len)

        # Step 3: batched ITM report generation (LoRA ON, soft tokens prepended).
        # Empty activation windows (e.g. a chunk beyond the response end) get
        # an empty report — there is nothing to read, and generating from
        # zero soft tokens would just freewheel hallucinations.
        nonempty_idx = [i for i, s in enumerate(softs) if s.shape[1] > 0]
        itm_reports = [""] * len(records)
        if nonempty_idx:
            gen_reports = self._batched_generate_itm_reports(
                [softs[i] for i in nonempty_idx], oracle_max_new,
            )
            for i, rep in zip(nonempty_idx, gen_reports):
                itm_reports[i] = rep

        # Step 4: assemble EvalResults.
        tok = self._tokenizer
        out: list[EvalResult] = []
        for i, rec in enumerate(records):
            output_tokens = len(tok.encode(base_responses[i], add_special_tokens=False))
            out.append(EvalResult(
                eval_id=rec.eval_id,
                runner=self.runner_name,
                itm_report=itm_reports[i],
                model_response=base_responses[i],
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_tokens=input_tokens_list[i],
                output_tokens=output_tokens,
                activation_tokens=n_acts[i],
            ))
        return out

    def _cached_batched_generate_base(
        self, all_messages: list[list[dict]], max_new_tokens: int
    ) -> list[str]:
        """`_batched_generate_base` behind the optional on-disk response cache.

        With PRISM_EVAL_BASE_RESPONSE_CACHE unset this is a pass-through.
        Otherwise cache hits skip generation entirely and misses are
        generated in one batch, then appended to the cache file.
        """
        if not _base_response_cache_path():
            return self._batched_generate_base(all_messages, max_new_tokens)

        cache = _load_base_response_cache()
        model_id = self._cfg["model_id"]
        keys = [_base_response_key(model_id, m, max_new_tokens) for m in all_messages]
        responses: list[str | None] = [cache.get(k) for k in keys]

        miss_idx = [i for i, r in enumerate(responses) if r is None]
        if miss_idx:
            generated = self._batched_generate_base(
                [all_messages[i] for i in miss_idx], max_new_tokens,
            )
            new_entries = []
            for i, resp in zip(miss_idx, generated):
                responses[i] = resp
                cache[keys[i]] = resp
                new_entries.append((keys[i], resp))
            _append_base_responses(new_entries)

        n_hits = len(all_messages) - len(miss_idx)
        if n_hits:
            logger.info(
                "Base-response cache: %d hits, %d generated (batch of %d)",
                n_hits, len(miss_idx), len(all_messages),
            )
        return responses  # type: ignore[return-value]

    def _batched_generate_base(
        self, all_messages: list[list[dict]], max_new_tokens: int
    ) -> list[str]:
        """Batched base-response generation (LoRA OFF), one ``generate()`` call."""
        tok = self._tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        all_prompt_ids = [_apply_chat_template(tok, m) for m in all_messages]
        max_input = self._cfg.get("max_input_tokens", 32768)
        all_prompt_ids = [
            self._truncate_middle(ids, max_input) if len(ids) > max_input else ids
            for ids in all_prompt_ids
        ]

        input_ids, attn_mask, _ = _left_pad(all_prompt_ids, pad_id)
        input_ids = input_ids.to(self._device)
        attn_mask = attn_mask.to(self._device)
        max_prompt_len = input_ids.shape[1]

        self._model.disable_adapter_layers()
        with torch.inference_mode():
            out = self._model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=self._gen_eos_id,
                use_cache=True,
            )
        self._model.enable_adapter_layers()

        # Left-padded: generated tokens start at max_prompt_len.
        return [
            tok.decode(out[i, max_prompt_len:].tolist(), skip_special_tokens=True).strip()
            for i in range(out.shape[0])
        ]

    def _batched_generate_itm_reports(
        self,
        soft_per_sample: list[torch.Tensor],  # each [1, n_act_i, D] bf16 on device
        max_new_tokens: int,
    ) -> list[str]:
        """Batched ITM-report generation (LoRA ON, soft tokens prepended).

        Left-pads soft tokens to ``max(n_act)`` with zero vectors + 0-mask,
        prepends the same decoder prefix to all samples, runs one batched
        ``generate()`` on ``inputs_embeds``.
        """
        tok = self._tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        B = len(soft_per_sample)
        D = soft_per_sample[0].shape[-1]
        n_acts = [s.shape[1] for s in soft_per_sample]
        max_n = max(n_acts) if n_acts else 0

        padded_soft = torch.zeros(B, max_n, D, dtype=torch.bfloat16, device=self._device)
        soft_mask = torch.zeros(B, max_n, dtype=torch.long, device=self._device)
        for i, s in enumerate(soft_per_sample):
            n = s.shape[1]
            if n > 0:
                padded_soft[i, max_n - n:, :] = s[0]
                soft_mask[i, max_n - n:] = 1

        skip_prompt_b = self._cfg.get("skip_prompt_b", False)
        decoder_prompt = "" if skip_prompt_b else self._cfg.get("decoder_user_prompt", "")
        prefix_ids = _apply_chat_template(
            tok, self._adapt_messages([{"role": "user", "content": decoder_prompt}])
        )
        prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=self._device)
        prefix_emb = self._model.get_input_embeddings()(prefix_tensor)  # [1, P, D]
        prefix_len = prefix_emb.shape[1]
        prefix_emb_b = prefix_emb.expand(B, -1, -1).to(dtype=torch.bfloat16)

        inputs_embeds = torch.cat([padded_soft, prefix_emb_b], dim=1)
        prefix_mask = torch.ones(B, prefix_len, dtype=torch.long, device=self._device)
        attn_mask = torch.cat([soft_mask, prefix_mask], dim=1)

        with torch.inference_mode():
            gen_ids = self._model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self._gen_eos_id,
                pad_token_id=pad_id,
                use_cache=True,
            )

        # With inputs_embeds, HF generate returns only the newly generated tokens.
        return [
            tok.decode(gen_ids[i], skip_special_tokens=True).strip()
            for i in range(gen_ids.shape[0])
        ]
