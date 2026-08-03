"""Model components for the PRISM runner.

Components:
  1. ActivationProjection — Linear(D→D), projects activations into the decoder's embedding space
  2. BottleneckMLPProjection — SwiGLU bottleneck MLP with residual (AURA Variant A)
  3. build_projection — factory for arch="linear" | "bottleneck_mlp"
  4. Target-model hook + EarlyExit — captures activations at a specific layer

Adapted from the standalone training/eval code for use within the prism-eval runner.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder (loaded from pretrain checkpoint, frozen)
# ─────────────────────────────────────────────────────────────────────────────


def _strip_prefix(state: dict, prefix: str) -> dict:
    return {
        k[len(prefix):] if k.startswith(prefix) else k: v
        for k, v in state.items()
    }


class ActivationProjection(nn.Module):
    """Linear projection from encoder output / raw activations → decoder embedding space."""

    def __init__(self, dim: int = 4096):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _RMSNorm(nn.Module):
    """RMSNorm — Qwen/Llama-style. Avoids a torch-version dependency on nn.RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight


class BottleneckMLPProjection(nn.Module):
    """Variant A from AURA-new-experiments §2: bottleneck MLP with residual.

        x → RMSNorm → SwiGLU(d → b → d) + residual

    SwiGLU = down_proj( silu(gate_proj(x)) * up_proj(x) ).
    `down_proj` is zero-initialised so the module starts at identity.
    """

    def __init__(self, dim: int = 4096, bottleneck_dim: int = 1024):
        super().__init__()
        self.norm = _RMSNorm(dim)
        self.gate_proj = nn.Linear(dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(dim, bottleneck_dim, bias=False)
        self.down_proj = nn.Linear(bottleneck_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x + h


def build_projection(arch: str, dim: int, bottleneck_dim: int = 1024) -> nn.Module:
    """Factory mirroring prism.pretrain_infoNCE.model.build_projection."""
    if arch == "linear":
        return ActivationProjection(dim=dim)
    if arch == "bottleneck_mlp":
        return BottleneckMLPProjection(dim=dim, bottleneck_dim=bottleneck_dim)
    raise NotImplementedError(
        f"projection_arch={arch!r} not implemented. "
        f"Supported: 'linear', 'bottleneck_mlp'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Norm matching
# ─────────────────────────────────────────────────────────────────────────────


def compute_target_norm(model) -> torch.Tensor:
    """Compute mean embedding norm from the decoder's embedding table."""
    with torch.no_grad():
        emb_weight = model.get_input_embeddings().weight
        return emb_weight.norm(dim=1).mean()


def norm_match(soft_tokens: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    """Scale soft tokens to match the decoder's embedding norm."""
    norms = soft_tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return soft_tokens / norms * target_norm


# ─────────────────────────────────────────────────────────────────────────────
# Qwen hook + EarlyExit
# ─────────────────────────────────────────────────────────────────────────────


class _EarlyExit(Exception):
    """Raised in the forward hook to abort Qwen's forward pass after the hooked layer."""


def _find_layers(model):
    """Find the transformer layer list in the model."""
    best = None

    def _walk(module, prefix=""):
        nonlocal best
        for name, child in module._modules.items():
            if child is None:
                continue
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.ModuleList) and len(child) > 1:
                types = {type(c).__name__ for c in child}
                if len(types) == 1:
                    if best is None or len(child) > len(best[1]):
                        best = (path, child)
            _walk(child, path)

    _walk(model)
    if best is None:
        raise RuntimeError("Cannot locate transformer layers in model.")

    return best[1]


def register_hook(model, layer_idx: int):
    """Register a forward hook that captures hidden states and raises EarlyExit.

    Returns:
        activation_store: dict with key "hidden" populated after each hooked forward
        hook_handle: call .remove() to deregister
    """
    layers = _find_layers(model)
    assert 0 <= layer_idx < len(layers), (
        f"hook_layer={layer_idx} out of range for model with {len(layers)} layers."
    )

    activation_store = {"active": False}

    def _hook(module, inp, out):
        if not activation_store["active"]:
            return
        hidden = out[0] if isinstance(out, tuple) else out
        activation_store["hidden"] = hidden.detach()
        raise _EarlyExit()

    handle = layers[layer_idx].register_forward_hook(_hook)
    logger.info("Hook registered on layer %d/%d.", layer_idx, len(layers) - 1)
    return activation_store, handle


# ─── Target-model profiles (rebuttal multi-model port) ──────────────────────
#
# Mirrors the training repo, trimmed to the
# fields the eval runners need. Selected from the checkpoint's config
# (`_subject_profile`, stamped by the training-side apply_profile_overlay),
# with a model_id fallback for checkpoints that predate the field. The
# default (Qwen) profile reproduces the original runner behaviour exactly.

SUBJECT_PROFILES: dict[str, dict] = {
    "qwen3.5-9b": dict(
        load_class="image_text_to_text",
        tokenizer_via_processor=True,
        attn_implementation=None,
        turn_end_token=None,     # eos ends the assistant turn
        no_system_role=False,
        system_message_override=None,
    ),
    "gemma2-9b": dict(
        load_class="causal_lm",
        tokenizer_via_processor=False,
        attn_implementation="eager",   # sdpa silently drops gemma-2 softcapping
        turn_end_token="<end_of_turn>",  # id 107; eos (1) never appears in turns
        no_system_role=True,           # gemma-2 chat template forbids system role
        system_message_override=None,
    ),
    "ministral3-8b": dict(
        load_class="image_text_to_text",  # Mistral3ForConditionalGeneration
        tokenizer_via_processor=False,
        attn_implementation="sdpa",
        turn_end_token=None,
        no_system_role=False,
        # Training wrapped every template call with an EMPTY system message to
        # suppress Ministral's ~530-token default system prompt (see
        # the training repo subject_models.py). Mirror that whenever
        # a message list has no system turn — most critically the decoder
        # prefix, which is soft-tokens + template(user:"").
        system_message_override="",
    ),
}


def get_subject_profile(ckpt_cfg: dict) -> dict:
    """Resolve the target-model profile for a loaded checkpoint config."""
    name = ckpt_cfg.get("_subject_profile")
    if name is None:
        model_id = ckpt_cfg.get("model_id", "")
        if "gemma-2" in model_id:
            name = "gemma2-9b"
        elif "Ministral" in model_id:
            name = "ministral3-8b"
        else:
            name = "qwen3.5-9b"
    if name not in SUBJECT_PROFILES:
        raise KeyError(f"Unknown subject profile {name!r}. Known: {sorted(SUBJECT_PROFILES)}")
    prof = dict(SUBJECT_PROFILES[name])
    prof["name"] = name
    return prof
