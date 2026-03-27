"""
Voxtral Codec encoder.

This version follows the research notes more closely:
  - waveform is first split into non-overlapping 240-sample patches
  - a causal kernel-7 projection maps 240-dim patches to hidden states
  - each encoder block applies transformer layers before a causal CNN layer
  - transformer attention uses sliding windows, ALiBi, QK norm and LayerScale
  - the final block projects to the 292-dim latent with stride 1
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _expand_block_param(value: int | Sequence[int], n_blocks: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * n_blocks
    values = tuple(value)
    if len(values) != n_blocks:
        raise ValueError(f"{name} must have length {n_blocks}, got {len(values)}")
    return values


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.padding, 0)))


class Patchify(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time = x.shape
        if channels != 1:
            raise ValueError(f"Expected mono waveform input, got {channels} channels")
        pad = (-time) % self.patch_size
        if pad:
            x = F.pad(x, (0, pad))
        frames = x.shape[-1] // self.patch_size
        return x.view(batch, 1, frames, self.patch_size).squeeze(1).transpose(1, 2).contiguous()


class ResidualCausalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            CausalConv1d(channels, channels, kernel_size, dilation=dilation),
            nn.ELU(),
            CausalConv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 0.01) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _alibi_slopes(n_heads: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = torch.linspace(0.0, 1.0, n_heads, device=device, dtype=dtype)
    return torch.pow(torch.tensor(2.0, device=device, dtype=dtype), -(8.0 * base))


class CausalSlidingWindowAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, window_size: int, qk_norm_eps: float = 1e-6) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size
        self.qk_norm_eps = qk_norm_eps

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, dim = x.shape
        q = self.q_proj(x).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)

        q = F.normalize(q, dim=-1, eps=self.qk_norm_eps)
        k = F.normalize(k, dim=-1, eps=self.qk_norm_eps)
        attn = torch.matmul(q, k.transpose(-2, -1))

        positions = torch.arange(time, device=x.device)
        relative = positions[None, :] - positions[:, None]
        mask = relative < 0
        if self.window_size > 0:
            mask |= relative >= self.window_size

        slopes = _alibi_slopes(self.n_heads, x.device, x.dtype).view(1, self.n_heads, 1, 1)
        alibi = -relative.abs().to(dtype=x.dtype).view(1, 1, time, time) * slopes
        attn = attn + alibi.masked_fill(mask.view(1, 1, time, time), float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, time, dim)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim: int,
        window_size: int,
        layer_scale_init: float = 0.01,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSlidingWindowAttention(dim, n_heads, window_size, qk_norm_eps=qk_norm_eps)
        self.scale1 = LayerScale(dim, init_value=layer_scale_init)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.scale2 = LayerScale(dim, init_value=layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scale1(self.attn(self.norm1(x)))
        x = x + self.scale2(self.ffn(self.norm2(x)))
        return x


class SlidingWindowTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim: int,
        n_layers: int,
        window_size: int,
        layer_scale_init: float = 0.01,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    dim,
                    n_heads,
                    ffn_dim,
                    window_size,
                    layer_scale_init=layer_scale_init,
                    qk_norm_eps=qk_norm_eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x).transpose(1, 2)


class CausalDownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        n_residual: int,
        dilations: Sequence[int],
    ) -> None:
        super().__init__()
        if len(dilations) != n_residual:
            raise ValueError("len(dilations) must equal n_residual")
        self.residuals = nn.Sequential(
            *[ResidualCausalBlock(in_channels, kernel_size=7, dilation=d) for d in dilations]
        )
        self.downsample = nn.Sequential(
            nn.ELU(),
            CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.residuals(x))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        n_residual: int,
        dilations: Sequence[int],
        n_transformer_layers: int,
        n_heads: int,
        ffn_dim: int,
        window_size: int,
        layer_scale_init: float = 0.01,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.transformer = SlidingWindowTransformer(
            dim=in_channels,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            n_layers=n_transformer_layers,
            window_size=window_size,
            layer_scale_init=layer_scale_init,
            qk_norm_eps=qk_norm_eps,
        )
        self.cnn = CausalDownsampleBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            n_residual=n_residual,
            dilations=dilations,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(self.transformer(x))


class VoxtralEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 1024,
        latent_dim: int = 292,
        patch_stride: int = 240,
        patch_kernel_size: int = 7,
        block_strides: Sequence[int] = (2, 2, 2, 1),
        block_kernel_sizes: Sequence[int] = (4, 4, 4, 3),
        n_residual: int = 3,
        dilations: Sequence[int] = (1, 3, 9),
        n_transformer_layers: int | Sequence[int] = (2, 2, 2, 2),
        n_heads: int = 16,
        ffn_dim: int = 4096,
        window_size: int | Sequence[int] = (16, 8, 4, 2),
        layer_scale_init: float = 0.01,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("VoxtralEncoder expects mono waveform input")

        block_strides = tuple(block_strides)
        n_blocks = len(block_strides)
        block_kernel_sizes = _expand_block_param(block_kernel_sizes, n_blocks, "block_kernel_sizes")
        window_sizes = _expand_block_param(window_size, n_blocks, "window_size")
        transformer_layers = _expand_block_param(n_transformer_layers, n_blocks, "n_transformer_layers")

        self.patch_stride = patch_stride
        self.patchify = Patchify(patch_stride)
        self.patch_projection = nn.Sequential(
            CausalConv1d(patch_stride, hidden_dim, kernel_size=patch_kernel_size),
            nn.ELU(),
        )

        block_out_dims = (hidden_dim, hidden_dim, hidden_dim, latent_dim)
        if len(block_out_dims) != n_blocks:
            raise ValueError("Expected exactly four encoder blocks")

        self.blocks = nn.ModuleList()
        in_dim = hidden_dim
        for out_dim, stride, kernel_size, layers, win in zip(
            block_out_dims,
            block_strides,
            block_kernel_sizes,
            transformer_layers,
            window_sizes,
        ):
            self.blocks.append(
                EncoderBlock(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    stride=stride,
                    kernel_size=kernel_size,
                    n_residual=n_residual,
                    dilations=dilations,
                    n_transformer_layers=layers,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    window_size=win,
                    layer_scale_init=layer_scale_init,
                    qk_norm_eps=qk_norm_eps,
                )
            )
            in_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patchify(x)
        x = self.patch_projection(x)
        for block in self.blocks:
            x = block(x)
        return x
