"""Checkpoint-faithful RetViM inference with a portable Mamba-1 recurrence.

The released checkpoint contains native one-dimensional Mamba weights. It does
not contain the optional four-direction SS2D branch described in the notebooks.
This implementation retains every trained tensor and executes the full selective
state recurrence on ordinary PyTorch, without requiring CUDA extensions.

Reference equations: https://github.com/state-spaces/mamba/blob/main/
mamba_ssm/ops/selective_scan_interface.py (selective_scan_ref).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import timm
import torch
from safetensors.torch import load_file
from timm.layers import DropPath
from torch import Tensor, nn
from torch.nn import functional as F

CLASS_NAMES = ("CNV", "DME", "DRUSEN", "NORMAL")
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKBONE = "vit_base_patch16_224.augreg_in21k_ft_in1k"


def selective_scan(
    u: Tensor, delta: Tensor, a: Tensor, b: Tensor, c: Tensor, d: Tensor, z: Tensor
) -> Tensor:
    """Evaluate the real-valued Mamba-1 recurrence in float32.

    ``u``, positive ``delta`` and ``z`` have shape [batch, length, channels];
    ``a`` is [channels, state], ``b``/``c`` are [batch, length, state].
    Chunking only bounds temporary memory; state is carried across every token.
    """
    output_dtype = u.dtype
    u, delta, b, c = u.float(), delta.float(), b.float(), c.float()
    a, d = a.float(), d.float()
    batch, length, channels = u.shape
    state = u.new_zeros(batch, channels, a.shape[-1])
    outputs = []
    for start in range(0, length, 32):
        end = min(start + 32, length)
        delta_chunk = delta[:, start:end]
        transition = torch.exp(delta_chunk.unsqueeze(-1) * a[None, None])
        drive = (
            delta_chunk.unsqueeze(-1)
            * b[:, start:end, None, :]
            * u[:, start:end, :, None]
        )
        for offset in range(end - start):
            state = transition[:, offset] * state + drive[:, offset]
            value = (state * c[:, start + offset, None, :]).sum(dim=-1)
            outputs.append(value)
    values = torch.stack(outputs, dim=1) + u * d[None, None]
    return (values * F.silu(z.float())).to(output_dtype)


class NativeMamba(nn.Module):
    """Portable Mamba-1 layer preserving the native checkpoint key structure."""

    def __init__(self, dim: int = 768, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        inner = dim * expand
        self.d_state = d_state
        self.dt_rank = math.ceil(dim / 16)
        self.in_proj = nn.Linear(dim, inner * 2, bias=False)
        self.conv1d = nn.Conv1d(inner, inner, d_conv, padding=d_conv - 1,
                                groups=inner, bias=True)
        self.x_proj = nn.Linear(inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, inner, bias=True)
        initial_a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(initial_a.log().repeat(inner, 1))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        x, gate = self.in_proj(tokens).chunk(2, dim=-1)
        length = tokens.shape[1]
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :length]).transpose(1, 2)
        delta, b, c = self.x_proj(x).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # Native Mamba applies the delta bias and softplus inside its scan kernel.
        delta = F.linear(delta, self.dt_proj.weight).float()
        delta = F.softplus(delta + self.dt_proj.bias.float())
        values = selective_scan(x, delta, -self.A_log.float().exp(), b, c,
                                self.D, gate)
        return self.out_proj(values)


class SSConvSSMBlock(nn.Module):
    """The trained parallel convolution/Mamba block, applied to patch tokens."""

    def __init__(self, dim: int = 768, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False),
            nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False),
            nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim),
        )
        self.ssm = NativeMamba(dim)
        self.fusion = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(),
                                    nn.Linear(dim, dim))
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, tokens: Tensor) -> Tensor:
        batch, count, dim = tokens.shape
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError("The hybrid blocks require a square grid of patch tokens.")
        normalized = self.norm(tokens)
        local = normalized.transpose(1, 2).reshape(batch, dim, side, side)
        local = self.conv_branch(local).flatten(2).transpose(1, 2)
        global_features = self.ssm(normalized)
        fused = self.fusion(torch.cat((local, global_features), dim=-1))
        return tokens + self.drop_path(fused)


class RetViM(nn.Module):
    """Architecture recorded in the improved notebook; load weights before use."""

    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(BACKBONE, pretrained=False, num_classes=0,
                                     drop_path_rate=0.1)
        dim = self.vit.num_features
        self.feature_dim = dim
        self.num_classes = len(CLASS_NAMES)
        self.medmamba_blocks = nn.ModuleList([
            SSConvSSMBlock(dim, 0.0), SSConvSSMBlock(dim, 0.1)
        ])
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.attn_pool = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(),
                                       nn.Linear(dim // 4, 1))
        self.pool_fusion = nn.Sequential(
            nn.Linear(dim * 4, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.125),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Expected normalized RGB images shaped [batch, 3, 224, 224].")
        tokens = self.vit.forward_features(images)
        cls_token, patches = tokens[:, 0], tokens[:, 1:]
        for block in self.medmamba_blocks:
            patches = block(patches)
        spatial = patches.transpose(1, 2)
        average = self.pool_avg(spatial).squeeze(-1)
        maximum = self.pool_max(spatial).squeeze(-1)
        attention = F.softmax(self.attn_pool(patches), dim=1)
        attended = (patches * attention).sum(dim=1)
        pooled = self.pool_fusion(torch.cat((cls_token, average, maximum, attended), dim=-1))
        return self.classifier(pooled)


def build_model(weights_path: str | Path, *, device: str | torch.device = "cpu") -> RetViM:
    """Load every trained tensor strictly and return an evaluation-mode model.

    Safetensors is the public release format. Plain tensor state dictionaries are
    accepted with PyTorch's restricted loader. Original Lightning checkpoints
    containing a notebook Config object must first be converted offline.
    """
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")
    if path.suffix == ".safetensors":
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()
    ):
        raise ValueError("Expected a non-empty tensor state dictionary; convert training checkpoints first.")
    model = RetViM()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
