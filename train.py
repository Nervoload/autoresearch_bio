"""
Autoresearch training script with profile-driven runs, artifacts, and MPS guardrails.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import math
import os
import random
import shutil
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ar_runtime import (
    apply_mps_memory_fraction,
    checkpoint_path,
    final_checkpoint_path,
    get_thermal_state,
    host_telemetry_snapshot,
    latest_checkpoint_path,
    load_profile,
    memory_bytes_to_mb,
    mps_memory_snapshot,
    path_str,
    prune_checkpoints,
    read_json,
    run_checkpoints_dir,
    run_config_path,
    run_manifest_path,
    run_metrics_path,
    run_sample_path,
    run_status_path,
    run_summary_path,
    run_system_path,
    summary_text,
    system_info,
    utcnow,
    write_json,
    write_text,
)
from prepare import Tokenizer, evaluate_bpb, make_dataloader


os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

H100_BF16_PEAK_FLOPS = 989.5e12

STOP_REQUESTED = False


def handle_stop_signal(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\nReceived signal {signum}, preparing to stop...", flush=True)


signal.signal(signal.SIGTERM, handle_stop_signal)
signal.signal(signal.SIGINT, handle_stop_signal)


@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    window_pattern: str


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        bsz, steps, _ = x.size()
        q = self.c_q(x).view(bsz, steps, self.n_head, self.head_dim)
        k = self.c_k(x).view(bsz, steps, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(bsz, steps, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(bsz, steps, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        window = window_size[0]
        if window > 0 and window < steps:
            mask = torch.ones(steps, steps, dtype=torch.bool, device=q.device).tril()
            mask = mask.triu(diagonal=1 - window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, steps, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        scale = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -scale, scale)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -scale, scale)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type in {"cuda", "mps"}:
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(
        self, seq_len: int, head_dim: int, base: int = 10000, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config: GPTConfig) -> list[tuple[int, int]]:
        pattern = config.window_pattern.upper()
        assert all(char in "SL" for char in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        windows = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            windows.append(char_to_window[char])
        windows[-1] = (long_window, 0)
        return windows

    def estimate_flops(self) -> float:
        nparams = sum(param.numel() for param in self.parameters())
        value_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        heads = self.config.n_head
        qdim = self.config.n_embd // self.config.n_head
        seq_len = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = seq_len if window < 0 else min(window, seq_len)
            attn_flops += 12 * heads * qdim * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self) -> dict[str, int]:
        wte = sum(param.numel() for param in self.transformer.wte.parameters())
        value_embeds = sum(param.numel() for param in self.value_embeds.parameters())
        lm_head = sum(param.numel() for param in self.lm_head.parameters())
        transformer_matrices = sum(param.numel() for param in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(self, optim_cfg: dict[str, Any], device_type: str) -> "MuonAdamW":
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(
                kind="adamw",
                params=lm_head_params,
                lr=float(optim_cfg["unembedding_lr"]) * dmodel_lr_scale,
                betas=tuple(optim_cfg["adam_betas"]),
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=embedding_params,
                lr=float(optim_cfg["embedding_lr"]) * dmodel_lr_scale,
                betas=tuple(optim_cfg["adam_betas"]),
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=value_embeds_params,
                lr=float(optim_cfg["embedding_lr"]) * dmodel_lr_scale,
                betas=tuple(optim_cfg["adam_betas"]),
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=resid_params,
                lr=float(optim_cfg["scalar_lr"]) * 0.01,
                betas=tuple(optim_cfg["adam_betas"]),
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=x0_params,
                lr=float(optim_cfg["scalar_lr"]),
                betas=(0.96, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),
        ]
        for shape in sorted({tuple(param.shape) for param in matrix_params}):
            group_params = [param for param in matrix_params if tuple(param.shape) == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=float(optim_cfg["matrix_lr"]),
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.95,
                    weight_decay=float(optim_cfg["weight_decay"]),
                )
            )
        optimizer = MuonAdamW(param_groups, device_type=device_type)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, reduction: str = "mean"
    ) -> torch.Tensor:
        _, steps = idx.size()
        cos_sin = self.cos[:, :steps], self.sin[:, :steps]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for layer_idx, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[layer_idx] * x + self.x0_lambdas[layer_idx] * x0
            ve = self.value_embeds[str(layer_idx)](idx) if str(layer_idx) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[layer_idx])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is None:
            return logits
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=reduction,
        )


polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(
    p,
    grad,
    exp_avg,
    exp_avg_sq,
    step_t,
    lr_t,
    beta1_t,
    beta2_t,
    eps_t,
    wd_t,
):
    step_t = step_t.to(device=p.device, dtype=p.dtype)
    lr_t = lr_t.to(device=p.device, dtype=p.dtype)
    beta1_t = beta1_t.to(device=p.device, dtype=p.dtype)
    beta2_t = beta2_t.to(device=p.device, dtype=p.dtype)
    eps_t = eps_t.to(device=p.device, dtype=p.dtype)
    wd_t = wd_t.to(device=p.device, dtype=p.dtype)
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


def muon_step_fused(
    stacked_grads,
    stacked_params,
    momentum_buffer,
    second_momentum_buffer,
    momentum_t,
    lr_t,
    wd_t,
    beta2_t,
    ns_steps,
    red_dim,
):
    momentum_t = momentum_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    lr_t = lr_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    wd_t = wd_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    beta2_t = beta2_t.to(device=stacked_params.device, dtype=stacked_params.dtype)

    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    x = g.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            a_mat = x.mT @ x
            b_mat = b * a_mat + c * (a_mat @ a_mat)
            x = a * x + x @ b_mat
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            a_mat = x @ x.mT
            b_mat = b * a_mat + c * (a_mat @ a_mat)
            x = a * x + b_mat @ x
    g = x
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    beta2_cast = beta2_t.to(second_momentum_buffer.dtype)
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2_cast)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict[str, Any]], device_type: str):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        compiler_kwargs = {"dynamic": False, "fullgraph": True}
        if device_type in {"cuda", "cpu"}:
            self.adamw_step_fused = torch.compile(adamw_step_fused, **compiler_kwargs)
            self.muon_step_fused = torch.compile(muon_step_fused, **compiler_kwargs)
        else:
            self.adamw_step_fused = adamw_step_fused
            self.muon_step_fused = muon_step_fused

    def _step_adamw(self, group: dict[str, Any]) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            self.adamw_step_fused(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_muon(self, group: dict[str, Any]) -> None:
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_params, shape[-2], 1)
                if shape[-2] >= shape[-1]
                else (num_params, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([param.grad for param in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        self.muon_step_fused(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)


def build_model_config(profile: dict[str, Any], vocab_size: int) -> tuple[GPTConfig, dict[str, Any]]:
    model_cfg = profile["model"]
    if model_cfg.get("legacy_scaling"):
        depth = int(model_cfg["depth"])
        aspect_ratio = int(model_cfg["aspect_ratio"])
        head_dim = int(model_cfg["head_dim"])
        base_dim = depth * aspect_ratio
        model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
        n_head = model_dim // head_dim
        n_kv_head = int(model_cfg.get("n_kv_head", n_head))
        resolved = {
            "legacy_scaling": True,
            "depth": depth,
            "aspect_ratio": aspect_ratio,
            "head_dim": head_dim,
            "n_layer": depth,
            "n_embd": model_dim,
            "n_head": n_head,
            "n_kv_head": n_kv_head,
            "window_pattern": model_cfg["window_pattern"],
        }
        return (
            GPTConfig(
                sequence_len=int(profile["max_seq_len"]),
                vocab_size=vocab_size,
                n_layer=depth,
                n_head=n_head,
                n_kv_head=n_kv_head,
                n_embd=model_dim,
                window_pattern=str(model_cfg["window_pattern"]),
            ),
            resolved,
        )

    resolved = {
        "legacy_scaling": False,
        "n_layer": int(model_cfg["n_layer"]),
        "n_embd": int(model_cfg["n_embd"]),
        "n_head": int(model_cfg["n_head"]),
        "n_kv_head": int(model_cfg.get("n_kv_head", model_cfg["n_head"])),
        "window_pattern": str(model_cfg["window_pattern"]),
    }
    return (
        GPTConfig(
            sequence_len=int(profile["max_seq_len"]),
            vocab_size=vocab_size,
            n_layer=resolved["n_layer"],
            n_head=resolved["n_head"],
            n_kv_head=resolved["n_kv_head"],
            n_embd=resolved["n_embd"],
            window_pattern=resolved["window_pattern"],
        ),
        resolved,
    )


def detect_device() -> tuple[str, torch.device]:
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.device("mps")
    return "cpu", torch.device("cpu")


def autocast_context(device_type: str):
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device_type == "cpu":
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def sync_device(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def get_lr_multiplier(progress: float, optim_cfg: dict[str, Any]) -> float:
    warmup_ratio = float(optim_cfg["warmup_ratio"])
    warmdown_ratio = float(optim_cfg["warmdown_ratio"])
    final_lr_frac = float(optim_cfg["final_lr_frac"])
    if progress < warmup_ratio:
        return progress / warmup_ratio if warmup_ratio > 0 else 1.0
    if progress < 1.0 - warmdown_ratio:
        return 1.0
    cooldown = (1.0 - progress) / warmdown_ratio if warmdown_ratio > 0 else final_lr_frac
    return cooldown * 1.0 + (1 - cooldown) * final_lr_frac


def get_muon_momentum(step: int) -> float:
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress: float, optim_cfg: dict[str, Any]) -> float:
    return float(optim_cfg["weight_decay"]) * (1 - progress)


def update_run_status(run_dir: Path, payload: dict[str, Any]) -> None:
    payload["heartbeat_at"] = utcnow()
    write_json(run_status_path(run_dir), payload)


def fast_forward_dataloader(
    loader,
    batches_to_skip: int,
    log_every: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    last_batch = None
    if batches_to_skip <= 0:
        return last_batch
    print(f"Restoring dataloader position ({batches_to_skip} batch(es))...", flush=True)
    for batch_idx in range(1, batches_to_skip + 1):
        last_batch = next(loader)
        if batch_idx % log_every == 0 or batch_idx == batches_to_skip:
            print(f"  restored {batch_idx}/{batches_to_skip} batches", flush=True)
    return last_batch


def save_checkpoint(
    run_dir: Path,
    tag: str,
    model: GPT,
    optimizer: MuonAdamW,
    step: int,
    epoch: int,
    loader_batches_seen: int,
    next_inputs: torch.Tensor,
    next_targets: torch.Tensor,
    total_training_time: float,
    smooth_train_loss: float,
    config: GPTConfig,
    profile: dict[str, Any],
) -> Path:
    target = checkpoint_path(run_dir, tag)
    ensure_rng = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        ensure_rng["cuda"] = torch.cuda.get_rng_state_all()
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "loader_batches_seen": loader_batches_seen,
        "next_inputs": next_inputs.detach().cpu().clone(),
        "next_targets": next_targets.detach().cpu().clone(),
        "train_seconds": total_training_time,
        "smooth_train_loss": smooth_train_loss,
        "config": asdict(config),
        "profile_id": profile["profile_id"],
        "rng_state": ensure_rng,
        "saved_at": utcnow(),
    }
    torch.save(payload, target)
    shutil.copyfile(target, latest_checkpoint_path(run_dir))
    prune_checkpoints(run_dir, int(profile["checkpoint"].get("keep_last", 2)))
    return target


def load_checkpoint(
    run_dir: Path,
    model: GPT,
    optimizer: MuonAdamW,
    device: torch.device,
) -> dict[str, Any] | None:
    path = latest_checkpoint_path(run_dir)
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    rng_state = checkpoint.get("rng_state", {})
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    return checkpoint


@torch.no_grad()
def generate_samples(
    run_dir: Path,
    profile: dict[str, Any],
    model: GPT,
    tokenizer: Tokenizer,
    device: torch.device,
    filename: str,
) -> Path:
    prompts = profile["sample"]["prompts"]
    max_new_tokens = int(profile["sample"]["max_new_tokens"])
    temperature = float(profile["sample"]["temperature"])
    top_k = int(profile["sample"]["top_k"])
    outputs = []
    model.eval()

    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
        idx = torch.tensor([token_ids[-model.config.sequence_len :]], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.config.sequence_len :]
            with autocast_context(device.type):
                logits = model(idx_cond)[:, -1, :]
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        decoded = tokenizer.decode(idx[0].tolist()[1:])
        outputs.append(f"=== Prompt ===\n{prompt}\n=== Sample ===\n{decoded}\n")

    path = run_sample_path(run_dir, filename)
    write_text(path, "\n".join(outputs))
    return path


def telemetry_snapshot(device_type: str, peak_memory_bytes: int) -> tuple[dict[str, Any], int]:
    memory = {}
    if device_type == "mps":
        snapshot = mps_memory_snapshot()
        peak_memory_bytes = max(
            peak_memory_bytes,
            int(snapshot.get("current_allocated") or 0),
            int(snapshot.get("driver_allocated") or 0),
        )
        memory = {
            "current_allocated_mb": memory_bytes_to_mb(snapshot.get("current_allocated")),
            "driver_allocated_mb": memory_bytes_to_mb(snapshot.get("driver_allocated")),
            "recommended_max_mb": memory_bytes_to_mb(snapshot.get("recommended_max")),
        }
    elif device_type == "cuda":
        peak_memory_bytes = max(peak_memory_bytes, int(torch.cuda.max_memory_allocated()))
        memory = {
            "current_allocated_mb": memory_bytes_to_mb(torch.cuda.memory_allocated()),
            "driver_allocated_mb": memory_bytes_to_mb(torch.cuda.memory_reserved()),
            "recommended_max_mb": None,
        }
    return memory, peak_memory_bytes


def update_host_telemetry(
    pid: int,
    process_cpu_percent: float | None,
    peak_process_rss_mb: float,
    peak_system_memory_used_mb: float,
) -> tuple[dict[str, Any], float, float]:
    host = host_telemetry_snapshot(pid)
    if process_cpu_percent is not None:
        host["process_cpu_percent"] = round(process_cpu_percent, 1)
    rss_mb = float(host.get("process_rss_mb") or 0.0)
    used_mb = float(host.get("memory_used_mb") or 0.0)
    peak_process_rss_mb = max(peak_process_rss_mb, rss_mb)
    peak_system_memory_used_mb = max(peak_system_memory_used_mb, used_mb)
    host["peak_process_rss_mb"] = round(peak_process_rss_mb, 1)
    host["peak_system_memory_used_mb"] = round(peak_system_memory_used_mb, 1)
    return host, peak_process_rss_mb, peak_system_memory_used_mb


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an autoresearch training experiment")
    parser.add_argument("--profile", default="climbmix_legacy")
    parser.add_argument("--mode", choices=["search", "soak"])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    profile = load_profile(args.profile, mode_override=args.mode)
    run_dir = args.run_dir or Path("results") / "runs" / f"manual-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_checkpoints_dir(run_dir)
    existing_manifest = read_json(run_manifest_path(run_dir), {})
    resume_candidate = latest_checkpoint_path(run_dir).exists()

    source_metadata_path = run_dir / "source_metadata.json"
    source_metadata = existing_manifest.get("source_metadata", {})
    if source_metadata_path.exists():
        payload = read_json(source_metadata_path, {})
        source_metadata = payload if isinstance(payload, dict) else {}

    write_json(
        run_manifest_path(run_dir),
        {
            "run_id": run_dir.name,
            "profile_id": profile["profile_id"],
            "mode": profile["mode"],
            "dataset_id": profile["dataset_id"],
            "tokenizer_id": profile["tokenizer_id"],
            "state": "initializing",
            "started_at": existing_manifest.get("started_at", utcnow()),
            "resume_count": int(existing_manifest.get("resume_count", 0)) + int(resume_candidate),
            "last_resumed_at": utcnow() if resume_candidate else existing_manifest.get("last_resumed_at"),
            "source_metadata": source_metadata,
            "artifacts": {
                "config": path_str(run_config_path(run_dir)),
                "system": path_str(run_system_path(run_dir)),
                "metrics": path_str(run_metrics_path(run_dir)),
                "summary": path_str(run_summary_path(run_dir)),
                "samples": path_str(run_sample_path(run_dir)),
                "latest_checkpoint": path_str(latest_checkpoint_path(run_dir)),
                "final_checkpoint": path_str(final_checkpoint_path(run_dir)),
            },
        },
    )

    system = system_info(profile)
    write_json(run_system_path(run_dir), system)
    device_type, device = detect_device()
    print(f"Device: {device_type}")
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    if device_type == "mps":
        apply_mps_memory_fraction(float(profile["mps"]["memory_fraction"]))

    tokenizer = Tokenizer.from_profile(profile)
    vocab_size = tokenizer.get_vocab_size()
    config, resolved_model = build_model_config(profile, vocab_size)
    resolved_config = {
        "profile_id": profile["profile_id"],
        "mode": profile["mode"],
        "time_budget_s": int(profile["time_budget_s"]),
        "max_seq_len": int(profile["max_seq_len"]),
        "eval_tokens": int(profile["eval_tokens"]),
        "dataset_id": profile["dataset_id"],
        "tokenizer_id": profile["tokenizer_id"],
        "model": resolved_model,
        "optim": profile["optim"],
        "mps": profile["mps"],
        "checkpoint": profile["checkpoint"],
    }
    write_json(run_config_path(run_dir), resolved_config)

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    param_counts = model.num_scaling_params()
    num_params = param_counts["total"]
    num_flops_per_token = model.estimate_flops()

    optim_cfg = profile["optim"]
    tokens_per_fwdbwd = int(optim_cfg["device_batch_size"]) * int(profile["max_seq_len"])
    total_batch_size = int(optim_cfg["total_batch_size"])
    assert total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd

    optimizer = model.setup_optimizer(optim_cfg, device_type=device_type)
    if device_type == "cuda":
        model = torch.compile(model, dynamic=False)

    checkpoint = None
    if args.resume or profile["mode"] == "soak":
        checkpoint = load_checkpoint(run_dir, model, optimizer, device=device)

    step = int(checkpoint["step"]) if checkpoint else 0
    train_loader = make_dataloader(
        profile,
        tokenizer,
        int(optim_cfg["device_batch_size"]),
        int(profile["max_seq_len"]),
        "train",
    )
    if (
        checkpoint
        and "loader_batches_seen" in checkpoint
        and "next_inputs" in checkpoint
        and "next_targets" in checkpoint
    ):
        loader_batches_seen = int(checkpoint["loader_batches_seen"])
        fast_forward_dataloader(train_loader, loader_batches_seen)
        x = checkpoint["next_inputs"].to(device=device, dtype=torch.long)
        y = checkpoint["next_targets"].to(device=device, dtype=torch.long)
        epoch = int(checkpoint.get("epoch", 1))
    elif checkpoint:
        # Legacy checkpoints did not store loader position or the prefetched next batch.
        print("Legacy checkpoint detected; reconstructing dataloader state.", flush=True)
        step += 1
        loader_batches_seen = step * grad_accum_steps + 1
        restored_batch = fast_forward_dataloader(train_loader, loader_batches_seen)
        if restored_batch is None:
            raise RuntimeError("Could not reconstruct dataloader state from legacy checkpoint.")
        x, y, epoch = restored_batch
    else:
        x, y, epoch = next(train_loader)
        loader_batches_seen = 1

    total_training_time = float(checkpoint["train_seconds"]) if checkpoint else 0.0
    smooth_train_loss = float(checkpoint["smooth_train_loss"]) if checkpoint else 0.0
    peak_memory_bytes = 0
    last_checkpoint = path_str(latest_checkpoint_path(run_dir)) if checkpoint else None
    latest_tok_per_sec = None
    thermal_state = get_thermal_state()
    warnings: list[str] = []
    thermal_events: list[dict[str, Any]] = []
    sustained_memory_samples = 0
    termination_state = "completed"
    last_error = None
    last_status_write = 0.0
    last_telemetry_check = 0.0
    last_checkpoint_save = time.time()
    started_at = time.time()
    time_budget = int(profile["time_budget_s"])
    last_process_cpu_time = time.process_time()
    process_cpu_percent = None
    host, peak_process_rss_mb, peak_system_memory_used_mb = update_host_telemetry(
        os.getpid(),
        process_cpu_percent=None,
        peak_process_rss_mb=0.0,
        peak_system_memory_used_mb=0.0,
    )

    print(f"Run dir: {run_dir}")
    print(f"Profile: {profile['profile_id']} ({profile['mode']})")
    print(f"Model config: {asdict(config)}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")
    print(f"Parameter counts: {param_counts}")

    update_run_status(
        run_dir,
        {
            "run_id": run_dir.name,
            "profile_id": profile["profile_id"],
            "mode": profile["mode"],
            "state": "running",
            "step": step,
            "tokens_total": step * total_batch_size,
            "train_seconds": total_training_time,
            "tok_per_sec": latest_tok_per_sec,
            "memory": {},
            "host": host,
            "thermal_state": thermal_state,
            "last_checkpoint": last_checkpoint,
            "last_error": None,
        },
    )

    def maybe_checkpoint(reason: str) -> None:
        nonlocal last_checkpoint, last_checkpoint_save
        saved = save_checkpoint(
            run_dir,
            reason,
            model,
            optimizer,
            step,
            epoch,
            loader_batches_seen,
            x,
            y,
            total_training_time,
            smooth_train_loss,
            config,
            profile,
        )
        last_checkpoint = str(saved)
        last_checkpoint_save = time.time()

    while True:
        sync_device(device_type)
        t0 = time.time()
        for _ in range(grad_accum_steps):
            with autocast_context(device_type):
                loss = model(x, y)
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, epoch = next(train_loader)
            loader_batches_seen += 1

        progress = min(total_training_time / time_budget, 1.0)
        lrm = get_lr_multiplier(progress, optim_cfg)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress, optim_cfg)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay

        optimizer.step()
        model.zero_grad(set_to_none=True)
        train_loss_f = train_loss.item()
        if math.isnan(train_loss_f) or train_loss_f > 100:
            termination_state = "failed"
            last_error = f"Training diverged with loss={train_loss_f}"
            break
        step += 1

        sync_device(device_type)
        dt = time.time() - t0
        process_cpu_time = time.process_time()
        if dt > 0:
            process_cpu_percent = 100.0 * (process_cpu_time - last_process_cpu_time) / dt
            host["process_cpu_percent"] = round(process_cpu_percent, 1)
        last_process_cpu_time = process_cpu_time
        if step > 10:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**step)
        latest_tok_per_sec = int(total_batch_size / dt)
        remaining = max(0, time_budget - total_training_time)

        if device_type == "cuda":
            mfu = 100 * num_flops_per_token * total_batch_size / dt / H100_BF16_PEAK_FLOPS
            perf_fragment = f" | mfu: {mfu:.1f}%"
        else:
            perf_fragment = ""

        memory, peak_memory_bytes = telemetry_snapshot(device_type, peak_memory_bytes)
        thermal_state = get_thermal_state()
        memory_fragment = ""
        host_fragment = ""
        if memory.get("current_allocated_mb") is not None:
            memory_fragment = (
                f" | memory: {memory['current_allocated_mb']:.1f}MB"
                f"/{memory.get('recommended_max_mb') or 0:.1f}MB"
            )
        if host.get("process_cpu_percent") is not None:
            host_fragment += f" | cpu: {host['process_cpu_percent']:.1f}%"
        if host.get("process_rss_mb") is not None:
            host_fragment += f" | rss: {host['process_rss_mb']:.0f}MB"
        if host.get("memory_used_mb") is not None and host.get("memory_total_mb") is not None:
            host_fragment += f" | sysmem: {host['memory_used_mb']:.0f}/{host['memory_total_mb']:.0f}MB"
        if host.get("load_1m") is not None:
            host_fragment += f" | load1: {host['load_1m']:.2f}"
        print(
            f"\rstep {step:05d} ({100 * progress:.1f}%) | loss: {debiased_smooth_loss:.6f}"
            f" | lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {latest_tok_per_sec:,}"
            f"{perf_fragment}{memory_fragment}{host_fragment} | thermal: {thermal_state}"
            f" | epoch: {epoch} | remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

        if step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 5000 == 0:
            gc.collect()

        now = time.time()
        if now - last_telemetry_check >= 5:
            last_telemetry_check = now
            host, peak_process_rss_mb, peak_system_memory_used_mb = update_host_telemetry(
                os.getpid(),
                process_cpu_percent=process_cpu_percent,
                peak_process_rss_mb=peak_process_rss_mb,
                peak_system_memory_used_mb=peak_system_memory_used_mb,
            )
            if device_type == "mps":
                cap = (mps_memory_snapshot().get("recommended_max") or 0) * float(
                    profile["mps"]["memory_fraction"]
                )
                used = max(
                    int((mps_memory_snapshot().get("current_allocated") or 0)),
                    int((mps_memory_snapshot().get("driver_allocated") or 0)),
                )
                warn_threshold = cap * float(profile["mps"]["warn_ratio"])
                abort_threshold = cap * float(profile["mps"]["abort_ratio"])
                if cap and used >= warn_threshold:
                    warning = f"MPS memory warning: used={used} cap={cap}"
                    if warning not in warnings:
                        warnings.append(warning)
                if cap and used >= abort_threshold:
                    sustained_memory_samples += 1
                else:
                    sustained_memory_samples = 0
                if sustained_memory_samples >= int(profile["mps"]["abort_samples"]):
                    termination_state = "memory_guard"
                    warnings.append("Memory guard triggered after sustained allocator pressure.")
                    maybe_checkpoint("memory-guard")
                    break

            if thermal_state == "fair":
                warning = "Thermal state is fair; monitoring."
                if warning not in warnings:
                    warnings.append(warning)
            elif thermal_state == "serious":
                thermal_events.append({"state": thermal_state, "at": utcnow()})
                termination_state = "thermal_guard"
                maybe_checkpoint("thermal-serious")
                break
            elif thermal_state == "critical":
                thermal_events.append({"state": thermal_state, "at": utcnow()})
                termination_state = "thermal_guard"
                maybe_checkpoint("thermal-critical")
                break

        if STOP_REQUESTED:
            termination_state = "stopped"
            if profile["mode"] == "soak":
                maybe_checkpoint("stopped")
            break

        if profile["mode"] == "soak" and now - last_checkpoint_save >= int(
            profile["checkpoint"]["interval_s"]
        ):
            saved = save_checkpoint(
                run_dir,
                f"step-{step:06d}",
                model,
                optimizer,
                step,
                epoch,
                loader_batches_seen,
                x,
                y,
                total_training_time,
                smooth_train_loss,
                config,
                profile,
            )
            last_checkpoint = str(saved)
            last_checkpoint_save = now
            generate_samples(run_dir, profile, model, tokenizer, device, f"step-{step:06d}.txt")

        if now - last_status_write >= 5:
            last_status_write = now
            update_run_status(
                run_dir,
                {
                    "run_id": run_dir.name,
                    "profile_id": profile["profile_id"],
                    "mode": profile["mode"],
                    "state": "running",
                    "step": step,
                    "tokens_total": step * total_batch_size,
                    "train_seconds": round(total_training_time, 2),
                    "tok_per_sec": latest_tok_per_sec,
                    "memory": memory,
                    "host": host,
                    "thermal_state": thermal_state,
                    "last_checkpoint": last_checkpoint,
                    "last_error": last_error,
                },
            )

        if step > 10 and total_training_time >= time_budget:
            break

    print()

    total_seconds = time.time() - started_at
    total_tokens = step * total_batch_size
    val_bpb = None
    sample_path = None

    if termination_state in {"completed", "stopped"} or (
        termination_state == "memory_guard" and profile["mode"] == "search"
    ):
        model.eval()
        if termination_state == "completed":
            with autocast_context(device_type):
                val_bpb = evaluate_bpb(
                    profile,
                    model,
                    tokenizer,
                    int(optim_cfg["device_batch_size"]),
                )
        maybe_checkpoint("final")
        shutil.copyfile(latest_checkpoint_path(run_dir), final_checkpoint_path(run_dir))
        last_checkpoint = str(final_checkpoint_path(run_dir))
        sample_path = generate_samples(run_dir, profile, model, tokenizer, device, "final.txt")
    elif last_checkpoint is None and profile["mode"] == "soak":
        maybe_checkpoint("final")
        last_checkpoint = str(latest_checkpoint_path(run_dir))

    if device_type == "cuda":
        peak_memory_bytes = max(peak_memory_bytes, int(torch.cuda.max_memory_allocated()))

    host, peak_process_rss_mb, peak_system_memory_used_mb = update_host_telemetry(
        os.getpid(),
        process_cpu_percent=process_cpu_percent,
        peak_process_rss_mb=peak_process_rss_mb,
        peak_system_memory_used_mb=peak_system_memory_used_mb,
    )

    metrics = {
        "run_id": run_dir.name,
        "profile_id": profile["profile_id"],
        "mode": profile["mode"],
        "state": termination_state,
        "val_bpb": round(float(val_bpb), 6) if val_bpb is not None else None,
        "training_seconds": round(total_training_time, 1),
        "total_seconds": round(total_seconds, 1),
        "peak_memory_mb": round(memory_bytes_to_mb(peak_memory_bytes) or 0.0, 1),
        "tok_per_sec": latest_tok_per_sec,
        "total_tokens_M": round(total_tokens / 1e6, 1),
        "num_steps": step,
        "num_params_M": round(num_params / 1e6, 1),
        "depth": config.n_layer,
        "thermal_state": thermal_state,
        "load_1m": host.get("load_1m"),
        "peak_process_rss_mb": round(peak_process_rss_mb, 1),
        "peak_system_memory_used_mb": round(peak_system_memory_used_mb, 1),
        "host": host,
        "warnings": warnings,
        "thermal_events": thermal_events,
        "last_checkpoint": last_checkpoint,
        "last_error": last_error,
        "sample_path": path_str(sample_path),
        "mfu_percent": None,
    }
    if device_type == "cuda" and total_training_time > 0:
        metrics["mfu_percent"] = round(
            100
            * num_flops_per_token
            * total_batch_size
            * max(step - 10, 0)
            / total_training_time
            / H100_BF16_PEAK_FLOPS,
            2,
        )

    write_json(run_metrics_path(run_dir), metrics)
    write_text(run_summary_path(run_dir), summary_text(metrics))
    manifest = read_json(run_manifest_path(run_dir), {})
    manifest["state"] = termination_state
    manifest["finished_at"] = utcnow()
    manifest["warnings"] = warnings
    manifest["thermal_events"] = thermal_events
    manifest["last_error"] = last_error
    write_json(run_manifest_path(run_dir), manifest)

    update_run_status(
        run_dir,
        {
            "run_id": run_dir.name,
            "profile_id": profile["profile_id"],
            "mode": profile["mode"],
            "state": termination_state,
            "step": step,
            "tokens_total": total_tokens,
            "train_seconds": round(total_training_time, 2),
            "tok_per_sec": latest_tok_per_sec,
            "memory": telemetry_snapshot(device_type, peak_memory_bytes)[0],
            "host": host,
            "thermal_state": thermal_state,
            "last_checkpoint": last_checkpoint,
            "last_error": last_error,
        },
    )

    print("---")
    print(f"run_id:            {run_dir.name}")
    print(f"val_bpb:           {metrics['val_bpb']}")
    print(f"training_seconds:  {metrics['training_seconds']}")
    print(f"total_seconds:     {metrics['total_seconds']}")
    print(f"peak_memory_mb:    {metrics['peak_memory_mb']}")
    print(f"tok_per_sec:       {metrics['tok_per_sec']}")
    print(f"total_tokens_M:    {metrics['total_tokens_M']}")
    print(f"num_steps:         {metrics['num_steps']}")
    print(f"num_params_M:      {metrics['num_params_M']}")
    print(f"depth:             {metrics['depth']}")
    print(f"thermal_state:     {metrics['thermal_state']}")
    print(f"state:             {metrics['state']}")
    return 0 if termination_state in {"completed", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
