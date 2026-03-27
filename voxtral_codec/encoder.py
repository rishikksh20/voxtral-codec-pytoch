"""
Voxtral Codec Encoder

Architecture:
  1. Patchification: strided Conv1d (patch_stride=240) converts 24 kHz → 100 Hz
  2. 4 Encoder Blocks, each consisting of:
       - Residual Causal CNN sub-blocks (with dilation)
       - Strided Causal CNN for downsampling (total 8x across all blocks)
       - Sliding-Window Causal Self-Attention Transformer
  3. Final linear projection to latent_dim (292)

Overall compression: 240 (patch) × 8 (CNN) = 1920x → 24 000 Hz / 1920 = 12.5 Hz output.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Causal convolution primitives
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """1-D convolution that pads only on the left so output never looks ahead."""

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
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualCausalBlock(nn.Module):
    """Residual block with two dilated causal convolutions and ELU activation."""

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


class CausalDownsampleBlock(nn.Module):
    """
    Causal CNN block that:
      - applies a stack of residual dilated conv sub-blocks
      - then optionally downsamples via a strided causal convolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        n_residual: int = 3,
        kernel_size: int = 7,
        dilations: tuple = (1, 3, 9),
    ) -> None:
        super().__init__()
        assert len(dilations) == n_residual, "len(dilations) must equal n_residual"

        # Residual sub-blocks at full time resolution
        residuals = [ResidualCausalBlock(in_channels, kernel_size, d) for d in dilations]
        self.residuals = nn.Sequential(*residuals)

        # Strided causal conv for temporal downsampling (or identity if stride=1)
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.ELU(),
                CausalConv1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride),
            )
        else:
            self.downsample = nn.Sequential(
                nn.ELU(),
                CausalConv1d(in_channels, out_channels, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residuals(x)
        x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# Causal sliding-window self-attention transformer
# ---------------------------------------------------------------------------

class CausalSlidingWindowAttention(nn.Module):
    """
    Multi-head self-attention where each token attends only to the W most
    recent tokens (causal, no future look-ahead).
    """

    def __init__(self, dim: int, n_heads: int, window_size: int) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, W = self.n_heads, self.window_size

        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, D)

        # Reshape to (B, H, T, head_dim)
        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, H, self.head_dim).transpose(1, 2)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)

        # Build causal sliding-window mask: -inf where j < i-W+1 or j > i
        mask = torch.zeros(T, T, device=x.device)
        for i in range(T):
            left = max(0, i - W + 1)
            mask[i, :left] = float("-inf")
            mask[i, i + 1 :] = float("-inf")

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    """Single transformer layer: sliding-window attention + FFN, both pre-norm."""

    def __init__(self, dim: int, n_heads: int, ffn_dim: int, window_size: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSlidingWindowAttention(dim, n_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SlidingWindowTransformer(nn.Module):
    """Stack of causal sliding-window transformer layers."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim: int,
        n_layers: int,
        window_size: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(dim, n_heads, ffn_dim, window_size) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) – convert to (B, T, C) for transformer
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x.transpose(1, 2)  # back to (B, C, T)


# ---------------------------------------------------------------------------
# Encoder block
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    One encoder stage:
      CausalDownsampleBlock (CNN, optional stride) → SlidingWindowTransformer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        n_residual: int,
        dilations: tuple,
        n_transformer_layers: int,
        n_heads: int,
        ffn_dim: int,
        window_size: int,
    ) -> None:
        super().__init__()
        self.cnn = CausalDownsampleBlock(
            in_channels, out_channels, stride=stride,
            n_residual=n_residual, dilations=dilations,
        )
        self.transformer = SlidingWindowTransformer(
            dim=out_channels,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            n_layers=n_transformer_layers,
            window_size=window_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.transformer(x)
        return x


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------

class VoxtralEncoder(nn.Module):
    """
    Voxtral Encoder.

    Converts a 24 kHz raw waveform (B, 1, T) to a 292-dim latent sequence
    at 12.5 Hz via:
      1. Patchification: Conv1d with stride=patch_stride (default 240) → 100 Hz
      2. Four encoder blocks with strides [2, 2, 2, 1] = 8x total → 12.5 Hz
      3. Linear projection to latent_dim

    Overall compression: 240 × 8 = 1920×  (24000 / 1920 = 12.5 Hz) ✓
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 768,
        latent_dim: int = 292,
        patch_stride: int = 240,
        block_strides: tuple = (2, 2, 2, 1),
        n_residual: int = 3,
        dilations: tuple = (1, 3, 9),
        n_transformer_layers: int = 3,
        n_heads: int = 12,
        ffn_dim: int = 3072,
        window_size: int = 32,
    ) -> None:
        super().__init__()

        # --- Patchification -------------------------------------------
        # Converts (B, 1, T) → (B, hidden_dim, T/patch_stride)
        self.patchify = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=patch_stride, stride=patch_stride),
            nn.ELU(),
        )

        # --- Four encoder blocks ---------------------------------------
        self.blocks = nn.ModuleList()
        for stride in block_strides:
            self.blocks.append(
                EncoderBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    stride=stride,
                    n_residual=n_residual,
                    dilations=dilations,
                    n_transformer_layers=n_transformer_layers,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    window_size=window_size,
                )
            )

        # --- Projection to latent space --------------------------------
        self.proj = nn.Conv1d(hidden_dim, latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T) raw waveform at 24 kHz

        Returns:
            z: (B, latent_dim, T_latent) latent at 12.5 Hz
        """
        x = self.patchify(x)          # (B, hidden_dim, T/240)
        for block in self.blocks:
            x = block(x)              # (B, hidden_dim, T/240/8)
        z = self.proj(x)              # (B, latent_dim, T_latent)
        return z
