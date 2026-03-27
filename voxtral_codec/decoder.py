"""
Voxtral Codec Decoder

Architecture (mirror of the encoder):
  1. Linear projection from latent_dim (292) → hidden_dim
  2. Four Decoder Blocks (in reverse order of the encoder strides [1, 2, 2, 2]):
       - Sliding-Window Causal Self-Attention Transformer
       - Causal upsampling CNN (nearest-neighbour upsample + causal conv)
  3. De-patchification: ConvTranspose1d with stride=patch_stride → 24 kHz output

Overall expansion: 8 (CNN) × 240 (depatch) = 1920×  (12.5 Hz → 24 000 Hz) ✓
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import (
    CausalConv1d,
    ResidualCausalBlock,
    SlidingWindowTransformer,
)


# ---------------------------------------------------------------------------
# Causal upsampling block
# ---------------------------------------------------------------------------

class CausalUpsampleBlock(nn.Module):
    """
    Causal upsampling via nearest-neighbour interpolation followed by
    a causal convolution (avoids artefacts of transposed convolutions).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        n_residual: int = 3,
        kernel_size: int = 7,
        dilations: tuple = (1, 3, 9),
    ) -> None:
        super().__init__()
        assert len(dilations) == n_residual

        self.stride = stride

        if stride > 1:
            # Nearest-neighbour upsample + causal conv to smooth
            self.upsample = nn.Sequential(
                nn.ELU(),
                nn.Upsample(scale_factor=stride, mode="nearest"),
                CausalConv1d(in_channels, out_channels, kernel_size=2 * stride),
            )
        else:
            self.upsample = nn.Sequential(
                nn.ELU(),
                CausalConv1d(in_channels, out_channels, 1),
            )

        # Residual dilated conv blocks after upsampling
        self.residuals = nn.Sequential(
            *[ResidualCausalBlock(out_channels, kernel_size, d) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.residuals(x)
        return x


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
        n_residual: int,
        dilations: tuple,
        n_transformer_layers: int,
        n_heads: int,
        ffn_dim: int,
        window_size: int,
    ) -> None:
        super().__init__()
        self.transformer = SlidingWindowTransformer(
            dim=channels,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            n_layers=n_transformer_layers,
            window_size=window_size,
        )
        self.cnn = CausalUpsampleBlock(
            in_channels=channels,
            out_channels=channels,
            stride=stride,
            n_residual=n_residual,
            dilations=dilations,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer(x)
        x = self.cnn(x)
        return x


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
        hidden_dim: int = 768,
        latent_dim: int = 292,
        patch_stride: int = 240,
        block_strides: tuple = (1, 2, 2, 2),
        n_residual: int = 3,
        dilations: tuple = (1, 3, 9),
        n_transformer_layers: int = 3,
        n_heads: int = 12,
        ffn_dim: int = 3072,
        window_size: int = 32,
    ) -> None:
        super().__init__()

        # --- Projection from latent space --------------------------------
        self.proj = nn.Conv1d(latent_dim, hidden_dim, 1)

        # --- Four decoder blocks -----------------------------------------
        self.blocks = nn.ModuleList()
        for stride in block_strides:
            self.blocks.append(
                DecoderBlock(
                    channels=hidden_dim,
                    stride=stride,
                    n_residual=n_residual,
                    dilations=dilations,
                    n_transformer_layers=n_transformer_layers,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    window_size=window_size,
                )
            )

        # --- De-patchification -------------------------------------------
        # Mirrors the patchify Conv1d: expands (B, hidden_dim, T_latent × 8)
        # back to (B, out_channels, T) at 24 kHz.
        self.depatchify = nn.Sequential(
            nn.ELU(),
            nn.ConvTranspose1d(
                hidden_dim, out_channels,
                kernel_size=patch_stride, stride=patch_stride,
            ),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim, T_latent) quantized latent at 12.5 Hz

        Returns:
            x_hat: (B, 1, T) reconstructed waveform at 24 kHz
        """
        x = self.proj(z)              # (B, hidden_dim, T_latent)
        for block in self.blocks:
            x = block(x)              # (B, hidden_dim, T_latent × 8)
        x_hat = self.depatchify(x)    # (B, 1, T)
        return x_hat
