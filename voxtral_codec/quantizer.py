"""
Dual Quantization for Voxtral Codec:

  * Semantic quantizer – Vector Quantization (VQ)
      dim=256, codebook_size=8192 → 1 semantic token per frame
      → log2(8192) = 13 bits/frame

  * Acoustic quantizer – Finite Scalar Quantization (FSQ)
      dim=36, levels=21 per dimension → 36 acoustic tokens per frame
      → 36 × log2(21) ≈ 158.1 bits/frame

  Total per frame: ≈ 171.1 bits/frame
  At 12.5 fps: ≈ 2139 bits/s ≈ 2.14 kbps ✓

The DualQuantizer splits the 292-dim latent (B, 292, T) into:
  - semantic part: first 256 dims
  - acoustic part: last 36 dims
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


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
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

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
        # (B, T, D) for distance computation
        z_perm = z.permute(0, 2, 1).contiguous().view(B * T, D)

        # Squared Euclidean distances to all codebook entries
        # ||z - e||² = ||z||² - 2 z·e + ||e||²
        distances = (
            z_perm.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * torch.mm(z_perm, self.codebook.weight.t())
            + self.codebook.weight.pow(2).sum(dim=1, keepdim=True).t()
        )  # (B*T, K)

        indices = distances.argmin(dim=1)          # (B*T,)
        z_q_flat = self.codebook(indices)          # (B*T, D)

        z_q = z_q_flat.view(B, T, D).permute(0, 2, 1).contiguous()  # (B, D, T)

        # Codebook loss + commitment loss
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
        loss = codebook_loss + commitment_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        indices = indices.view(B, T)  # (B, T)
        return z_q, indices, loss


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
        half = (n_levels - 1) / 2.0
        # half-range for bounding; kept as a buffer so it moves with .to(device)
        # NOTE: named _half_levels to avoid clashing with nn.Module.half()
        self.register_buffer("_half_levels", torch.tensor(half))

    def _bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Map z ∈ ℝ → (-half-ε, half+ε) via scaled tanh."""
        return torch.tanh(z) * (self._half_levels + eps)

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
        z_b = self._bound(z)          # bound to valid range
        z_q = self._round_ste(z_b)    # quantize with straight-through

        # Shift from (-half, half) to integer codes [0, n_levels-1]
        codes = (z_q + self._half_levels).long().clamp(0, self.n_levels - 1)
        return z_q, codes


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
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        assert latent_dim == semantic_dim + acoustic_dim, (
            f"latent_dim ({latent_dim}) must equal "
            f"semantic_dim ({semantic_dim}) + acoustic_dim ({acoustic_dim})"
        )
        self.semantic_dim = semantic_dim
        self.acoustic_dim = acoustic_dim

        self.vq = VectorQuantizer(
            codebook_size=codebook_size, dim=semantic_dim,
            commitment_cost=commitment_cost,
        )
        self.fsq = FSQ(n_levels=fsq_levels, dim=acoustic_dim)

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
        z_ac_q, ac_codes = self.fsq(z_ac)

        z_q = torch.cat([z_sem_q, z_ac_q], dim=1)  # (B, 292, T)

        return z_q, sem_indices, ac_codes, vq_loss, z_ac_q
