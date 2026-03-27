"""Dual semantic/acoustic quantization for Voxtral Codec."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vector Quantizer  (semantic)
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """
    Straight-through Vector Quantization with EMA codebook updates option.

    Args:
        codebook_size: number of codebook entries (K)
        dim:           embedding dimension (D)
        commitment_cost: weight for commitment loss (β)
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        dim: int = 256,
        commitment_cost: float = 0.1,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / dim, 1.0 / dim)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        batch, time = indices.shape
        z_q = self.codebook(indices.reshape(-1)).view(batch, time, self.dim)
        return z_q.permute(0, 2, 1).contiguous()

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, D, T)  – continuous latent from encoder projection

        Returns:
            z_q:    (B, D, T)  – quantized latent (straight-through)
            indices:(B, T)     – codebook indices
            loss:   scalar     – VQ + commitment loss
        """
        B, D, T = z.shape
        z_perm = z.permute(0, 2, 1).reshape(B * T, D)
        distances = (
            z_perm.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_perm @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        z_q = self.lookup(indices.view(B, T))
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        return z_q, indices.view(B, T), codebook_loss + commitment_loss


# ---------------------------------------------------------------------------
# Finite Scalar Quantizer  (acoustic)
# ---------------------------------------------------------------------------

class FSQ(nn.Module):
    """
    Finite Scalar Quantization (FSQ) — Lee et al. 2023.

    Each of the `dim` dimensions is independently quantized to `n_levels`
    uniform levels in the range (-L, L) where L = (n_levels - 1) / 2.

    Uses tanh bounding + round-with-straight-through-estimator.

    Args:
        n_levels: number of quantization levels per dimension (default 21)
        dim:      number of acoustic dimensions (default 36)
    """

    def __init__(self, n_levels: int = 21, dim: int = 36) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.dim = dim
        self.register_buffer("_half_levels", torch.tensor((n_levels - 1) / 2.0))
        self.noise_scale = 1.0 / float(n_levels)

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(z) * self._half_levels

    def _codes_from_values(self, z_q: torch.Tensor) -> torch.Tensor:
        return (z_q + self._half_levels).round().long().clamp(0, self.n_levels - 1)

    @staticmethod
    def _round_ste(z: torch.Tensor) -> torch.Tensor:
        """Round with straight-through estimator for gradients."""
        return z + (z.round() - z).detach()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, dim, T)

        Returns:
            z_q:    (B, dim, T)  – quantized values in (-half, half)
            codes:  (B, dim, T)  – integer codes in [0, n_levels-1]
        """
        return self.quantize(z)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_b = self._bound(z)
        z_q = self._round_ste(z_b)
        return z_q, self._codes_from_values(z_q)

    def dither(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_b = self._bound(z)
        noise = torch.empty_like(z_b).uniform_(-self.noise_scale, self.noise_scale)
        z_noisy = (z_b + noise).clamp(-self._half_levels, self._half_levels)
        return z_noisy, self._codes_from_values(z_noisy)

    def passthrough(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_b = self._bound(z)
        return z_b, self._codes_from_values(z_b)


# ---------------------------------------------------------------------------
# Dual Quantizer (semantic VQ + acoustic FSQ)
# ---------------------------------------------------------------------------

class DualQuantizer(nn.Module):
    """
    Splits the 292-dim latent into semantic (256-dim) and acoustic (36-dim),
    then applies VQ and FSQ respectively.

    Args:
        latent_dim:     total latent dimension (292)
        semantic_dim:   dimension for VQ (256)
        acoustic_dim:   dimension for FSQ (36)
        codebook_size:  VQ codebook entries (8192)
        fsq_levels:     FSQ levels per acoustic dimension (21)
        commitment_cost: VQ commitment loss weight
    """

    def __init__(
        self,
        latent_dim: int = 292,
        semantic_dim: int = 256,
        acoustic_dim: int = 36,
        codebook_size: int = 8192,
        fsq_levels: int = 21,
        commitment_cost: float = 0.1,
        semantic_quantize_prob: float = 0.5,
    ) -> None:
        super().__init__()
        assert latent_dim == semantic_dim + acoustic_dim
        self.semantic_dim = semantic_dim
        self.acoustic_dim = acoustic_dim
        self.semantic_quantize_prob = semantic_quantize_prob

        self.vq = VectorQuantizer(
            codebook_size=codebook_size, dim=semantic_dim,
            commitment_cost=commitment_cost,
        )
        self.fsq = FSQ(n_levels=fsq_levels, dim=acoustic_dim)

    def _semantic_mix(
        self,
        z_sem: torch.Tensor,
        z_sem_q: torch.Tensor,
        vq_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return z_sem_q, vq_loss
        batch = z_sem.shape[0]
        mask = (torch.rand(batch, device=z_sem.device) < self.semantic_quantize_prob).view(batch, 1, 1)
        z_out = torch.where(mask, z_sem_q, z_sem)
        return z_out, vq_loss if mask.any() else z_sem.new_zeros(())

    def _acoustic_mix(self, z_ac: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_quant, codes_quant = self.fsq.quantize(z_ac)
        if not self.training:
            return z_quant, codes_quant
        z_dither, codes_dither = self.fsq.dither(z_ac)
        z_pass, codes_pass = self.fsq.passthrough(z_ac)

        batch = z_ac.shape[0]
        probs = torch.rand(batch, device=z_ac.device)
        quant_mask = (probs < 0.50).view(batch, 1, 1)
        dither_mask = ((probs >= 0.50) & (probs < 0.75)).view(batch, 1, 1)
        z_out = torch.where(quant_mask, z_quant, torch.where(dither_mask, z_dither, z_pass))
        codes = torch.where(quant_mask, codes_quant, torch.where(dither_mask, codes_dither, codes_pass))
        return z_out, codes

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, latent_dim=292, T)

        Returns:
            z_q:            (B, 292, T)  – combined quantized latent
            semantic_indices:(B, T)      – VQ codebook indices (1 token/frame)
            acoustic_codes: (B, 36, T)  – FSQ integer codes (36 tokens/frame)
            vq_loss:        scalar       – VQ codebook + commitment loss
            acoustic_q:     (B, 36, T)  – continuous quantized acoustic latent
        """
        z_sem = z[:, : self.semantic_dim, :]   # (B, 256, T)
        z_ac  = z[:, self.semantic_dim :, :]   # (B, 36, T)

        z_sem_q, sem_indices, vq_loss = self.vq(z_sem)
        z_sem_out, vq_loss = self._semantic_mix(z_sem, z_sem_q, vq_loss)
        z_ac_out, ac_codes = self._acoustic_mix(z_ac)
        z_q = torch.cat([z_sem_out, z_ac_out], dim=1)
        return z_q, sem_indices, ac_codes, vq_loss, z_ac_out
