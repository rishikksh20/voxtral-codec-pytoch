"""
VoxtralCodec – Main Autoencoder Model

Combines:
  * VoxtralEncoder   – waveform → 292-dim latent at 12.5 Hz
  * DualQuantizer    – semantic VQ (256-dim) + acoustic FSQ (36-dim)
  * VoxtralDecoder   – 292-dim latent → reconstructed waveform

Default configuration targets ~300 M trainable parameters (encoder + decoder).
The discriminator is a separate module and is not included here.

Bitrate summary:
  Semantic  : log2(8192) = 13 bits/frame
  Acoustic  : 36 × log2(21) ≈ 158.1 bits/frame
  Total     : ≈ 171.1 bits/frame × 12.5 fps ≈ 2.14 kbps  ✓
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .encoder import VoxtralEncoder
from .decoder import VoxtralDecoder
from .quantizer import DualQuantizer


class VoxtralCodec(nn.Module):
    """
    End-to-end speech codec.

    Args:
        in_channels:          waveform channels (1 for mono)
        hidden_dim:           feature dimension in encoder/decoder (default 768)
        latent_dim:           total latent dimension = semantic_dim + acoustic_dim (292)
        semantic_dim:         semantic VQ dimension (256)
        acoustic_dim:         acoustic FSQ dimension (36)
        patch_stride:         patchification stride in samples at 24 kHz (240)
        encoder_strides:      per-block downsampling strides, product = 8
        decoder_strides:      per-block upsampling strides  (reverse of encoder)
        n_residual:           number of residual dilated conv sub-blocks per CNN block
        dilations:            dilation schedule for residual blocks
        n_transformer_layers: transformer layers per encoder/decoder block
        n_heads:              attention heads
        ffn_dim:              feed-forward hidden dim
        window_size:          causal sliding window size (tokens)
        codebook_size:        VQ codebook size (8192)
        fsq_levels:           FSQ levels per acoustic dimension (21)
        commitment_cost:      VQ commitment loss weight
        sample_rate:          audio sample rate (24000)
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 768,
        latent_dim: int = 292,
        semantic_dim: int = 256,
        acoustic_dim: int = 36,
        patch_stride: int = 240,
        encoder_strides: Tuple[int, ...] = (2, 2, 2, 1),
        decoder_strides: Tuple[int, ...] = (1, 2, 2, 2),
        n_residual: int = 3,
        dilations: Tuple[int, ...] = (1, 3, 9),
        n_transformer_layers: int = 3,
        n_heads: int = 12,
        ffn_dim: int = 3072,
        window_size: int = 32,
        codebook_size: int = 8192,
        fsq_levels: int = 21,
        commitment_cost: float = 0.25,
        sample_rate: int = 24_000,
    ) -> None:
        super().__init__()

        assert latent_dim == semantic_dim + acoustic_dim, (
            f"latent_dim ({latent_dim}) must equal "
            f"semantic_dim + acoustic_dim ({semantic_dim} + {acoustic_dim})"
        )

        self.sample_rate = sample_rate
        self.patch_stride = patch_stride
        self.encoder_strides = encoder_strides
        self.latent_dim = latent_dim
        self.semantic_dim = semantic_dim
        self.acoustic_dim = acoustic_dim

        # Effective temporal compression: patch_stride × product(encoder_strides)
        import math
        _cnn_stride = 1
        for s in encoder_strides:
            _cnn_stride *= s
        self.total_stride = patch_stride * _cnn_stride
        self.frame_rate = sample_rate / self.total_stride  # 12.5 Hz

        # ------------------------------------------------------------------
        self.encoder = VoxtralEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            patch_stride=patch_stride,
            block_strides=encoder_strides,
            n_residual=n_residual,
            dilations=dilations,
            n_transformer_layers=n_transformer_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            window_size=window_size,
        )

        self.quantizer = DualQuantizer(
            latent_dim=latent_dim,
            semantic_dim=semantic_dim,
            acoustic_dim=acoustic_dim,
            codebook_size=codebook_size,
            fsq_levels=fsq_levels,
            commitment_cost=commitment_cost,
        )

        self.decoder = VoxtralDecoder(
            out_channels=in_channels,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            patch_stride=patch_stride,
            block_strides=decoder_strides,
            n_residual=n_residual,
            dilations=dilations,
            n_transformer_layers=n_transformer_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            window_size=window_size,
        )

    # ------------------------------------------------------------------
    # Encode / decode API
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode raw waveform to continuous latent.

        Args:
            x: (B, 1, T) waveform at self.sample_rate

        Returns:
            z: (B, latent_dim, T_latent)
        """
        return self.encoder(x)

    def quantize(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize latent with dual VQ+FSQ.

        Args:
            z: (B, latent_dim, T_latent) continuous latent

        Returns:
            z_q:             (B, latent_dim, T_latent)
            semantic_indices:(B, T_latent) VQ indices
            acoustic_codes:  (B, acoustic_dim, T_latent) FSQ codes
            vq_loss:         scalar
        """
        z_q, sem_idx, ac_codes, vq_loss, _ = self.quantizer(z)
        return z_q, sem_idx, ac_codes, vq_loss

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latent to waveform.

        Args:
            z_q: (B, latent_dim, T_latent)

        Returns:
            x_hat: (B, 1, T)
        """
        return self.decoder(z_q)

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full encode-quantize-decode pass.

        Args:
            x: (B, 1, T) raw waveform

        Returns:
            x_hat:           (B, 1, T)         reconstructed waveform
            z:               (B, latent_dim, T_latent) pre-quantization latent
            semantic_indices:(B, T_latent)     VQ codebook indices
            acoustic_codes:  (B, acoustic_dim, T_latent) FSQ integer codes
            vq_loss:         scalar             VQ commitment + codebook loss
        """
        z = self.encoder(x)                                          # encode
        z_q, sem_idx, ac_codes, vq_loss, _ = self.quantizer(z)      # quantize
        x_hat = self.decoder(z_q)                                    # decode

        # Trim or pad to match input length (strided convolutions may shift length)
        T_in = x.shape[-1]
        T_out = x_hat.shape[-1]
        if T_out > T_in:
            x_hat = x_hat[..., :T_in]
        elif T_out < T_in:
            x_hat = torch.nn.functional.pad(x_hat, (0, T_in - T_out))

        return x_hat, z, sem_idx, ac_codes, vq_loss

    # ------------------------------------------------------------------
    # Convenience: decode from discrete codes
    # ------------------------------------------------------------------

    def decode_from_codes(
        self,
        semantic_indices: torch.Tensor,
        acoustic_codes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct waveform directly from discrete token sequences.

        Args:
            semantic_indices: (B, T_latent)       VQ indices (int64)
            acoustic_codes:   (B, acoustic_dim, T_latent) FSQ codes (int64)

        Returns:
            x_hat: (B, 1, T)
        """
        vq = self.quantizer.vq
        fsq = self.quantizer.fsq

        # Dequantize semantic
        B, T = semantic_indices.shape
        z_sem = vq.codebook(semantic_indices.view(-1)).view(B, T, -1)
        z_sem = z_sem.permute(0, 2, 1)  # (B, 256, T)

        # Dequantize acoustic: codes [0, n_levels-1] → continuous values
        half = fsq._half_levels.float()
        z_ac = acoustic_codes.float() - half  # (B, 36, T)

        z_q = torch.cat([z_sem, z_ac], dim=1)  # (B, 292, T)
        return self.decoder(z_q)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def num_parameters(self) -> Dict[str, int]:
        """Return parameter counts for each sub-module."""

        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        return {
            "encoder":    _count(self.encoder),
            "decoder":    _count(self.decoder),
            "quantizer":  _count(self.quantizer),
            "total":      _count(self),
        }

    def info(self) -> str:
        params = self.num_parameters()
        return (
            f"VoxtralCodec\n"
            f"  Sample rate  : {self.sample_rate} Hz\n"
            f"  Frame rate   : {self.frame_rate:.1f} Hz\n"
            f"  Total stride : {self.total_stride}×\n"
            f"  Latent dim   : {self.latent_dim} "
            f"(semantic={self.semantic_dim}, acoustic={self.acoustic_dim})\n"
            f"  Encoder      : {params['encoder'] / 1e6:.1f}M params\n"
            f"  Decoder      : {params['decoder'] / 1e6:.1f}M params\n"
            f"  Quantizer    : {params['quantizer'] / 1e6:.2f}M params\n"
            f"  Total        : {params['total'] / 1e6:.1f}M params\n"
        )
