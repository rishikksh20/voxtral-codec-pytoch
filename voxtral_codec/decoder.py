"""Voxtral Codec decoder."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .encoder import CausalConv1d, ResidualCausalBlock, SlidingWindowTransformer, _expand_block_param


# ---------------------------------------------------------------------------
# Causal upsampling block
# ---------------------------------------------------------------------------

class CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.trim = max(kernel_size - stride, 0)
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        if self.trim:
            x = x[..., :-self.trim]
        return x


class CausalUpsampleBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        n_residual: int = 3,
        dilations: Sequence[int] = (1, 3, 9),
    ) -> None:
        super().__init__()
        if len(dilations) != n_residual:
            raise ValueError("len(dilations) must equal n_residual")
        self.upsample = nn.Sequential(
            nn.ELU(),
            CausalConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        )
        self.residuals = nn.Sequential(
            *[ResidualCausalBlock(out_channels, kernel_size, d) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residuals(self.upsample(x))


# ---------------------------------------------------------------------------
# Decoder block (Transformer + upsampling CNN)
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    One decoder stage: SlidingWindowTransformer → CausalUpsampleBlock.
    The transformer runs at the lower (compressed) time resolution,
    then the CNN upsamples to the next level.
    """

    def __init__(
        self,
        channels: int,
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
        self.cnn = CausalUpsampleBlock(
            in_channels=channels,
            out_channels=channels,
            stride=stride,
            kernel_size=kernel_size,
            n_residual=n_residual,
            dilations=dilations,
        )
        self.transformer = SlidingWindowTransformer(
            dim=channels,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            n_layers=n_transformer_layers,
            window_size=window_size,
            layer_scale_init=layer_scale_init,
            qk_norm_eps=qk_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.cnn(x))


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------

class VoxtralDecoder(nn.Module):
    """
    Voxtral Decoder.

    Converts the 292-dim latent sequence at 12.5 Hz back to a 24 kHz
    raw waveform (B, 1, T) via:
      1. Linear projection: latent_dim → hidden_dim
      2. Four decoder blocks (strides [1, 2, 2, 2]) = 8× upsampling
      3. De-patchification: ConvTranspose1d with stride=patch_stride → 24 kHz

    Overall expansion: 8 × 240 = 1920×  (12.5 → 24 000 Hz) ✓
    """

    def __init__(
        self,
        out_channels: int = 1,
        hidden_dim: int = 1024,
        latent_dim: int = 292,
        patch_stride: int = 240,
        patch_kernel_size: int = 7,
        block_strides: Sequence[int] = (1, 2, 2, 2),
        block_kernel_sizes: Sequence[int] = (3, 4, 4, 4),
        n_residual: int = 3,
        dilations: Sequence[int] = (1, 3, 9),
        n_transformer_layers: int | Sequence[int] = (2, 2, 2, 2),
        n_heads: int = 16,
        ffn_dim: int = 4096,
        window_size: int | Sequence[int] = (2, 4, 8, 16),
        layer_scale_init: float = 0.01,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        block_strides = tuple(block_strides)
        n_blocks = len(block_strides)
        block_kernel_sizes = _expand_block_param(block_kernel_sizes, n_blocks, "block_kernel_sizes")
        window_sizes = _expand_block_param(window_size, n_blocks, "window_size")
        transformer_layers = _expand_block_param(n_transformer_layers, n_blocks, "n_transformer_layers")

        self.patch_stride = patch_stride
        self.input_projection = nn.Sequential(
            CausalConv1d(latent_dim, hidden_dim, kernel_size=1),
            nn.ELU(),
        )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    channels=hidden_dim,
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
                for stride, kernel_size, layers, win in zip(
                    block_strides,
                    block_kernel_sizes,
                    transformer_layers,
                    window_sizes,
                )
            ]
        )
        self.output_projection = nn.Sequential(
            nn.ELU(),
            CausalConv1d(hidden_dim, patch_stride, kernel_size=patch_kernel_size),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim, T_latent) quantized latent at 12.5 Hz

        Returns:
            x_hat: (B, 1, T) reconstructed waveform at 24 kHz
        """
        x = self.input_projection(z)
        for block in self.blocks:
            x = block(x)
        x = self.output_projection(x)
        batch, patch_dim, frames = x.shape
        return x.transpose(1, 2).reshape(batch, 1, frames * patch_dim)
