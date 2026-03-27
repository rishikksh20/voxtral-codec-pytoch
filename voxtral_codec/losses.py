from __future__ import annotations

"""Training losses for Voxtral Codec."""

from typing import List, Tuple

import torch
import torch.nn.functional as F

from .discriminator import DEFAULT_STFT_SIZES


# ---------------------------------------------------------------------------
# Reconstruction loss  (exponentially decaying weight)
# ---------------------------------------------------------------------------

def reconstruction_loss(
    x_real: torch.Tensor,
    x_hat: torch.Tensor,
    step: int,
    initial_weight: float = 1.0,
    decay_steps: float | None = None,
    decay_base: float = 0.9999,
) -> Tuple[torch.Tensor, float]:
    """
    L1 waveform reconstruction loss with an exponentially decaying weight.

    λ(t) = initial_weight · exp(−t / decay_steps)

    This high weight during early training helps bootstrap learning before
    the discriminator is strong enough to provide meaningful gradients.

    Args:
        x_real:        (B, 1, T) real waveform
        x_hat:         (B, 1, T) reconstructed waveform
        step:          current training step
        initial_weight: λ₀  (weight at step 0)
        decay_steps:   τ    (e-folding decay constant, in steps)

    Returns:
        loss:   weighted reconstruction loss (scalar tensor)
        weight: current λ(t)  (Python float, for logging)
    """
    if decay_steps is not None:
        weight = initial_weight * torch.exp(
            torch.tensor(-step / decay_steps, device=x_real.device, dtype=x_real.dtype)
        ).item()
    else:
        weight = initial_weight * (decay_base ** step)
    loss = F.l1_loss(x_hat, x_real)
    return weight * loss, weight


def stft_magnitude_loss(
    x_real: torch.Tensor,
    x_hat: torch.Tensor,
    fft_sizes: Tuple[int, ...] = DEFAULT_STFT_SIZES,
) -> torch.Tensor:
    total = x_real.new_zeros(())
    x_real_1d = x_real.squeeze(1)
    x_hat_1d = x_hat.squeeze(1)
    for size in fft_sizes:
        window = torch.hann_window(size, device=x_real.device)
        real_spec = torch.stft(
            x_real_1d,
            n_fft=size,
            hop_length=max(size // 4, 1),
            win_length=size,
            window=window,
            return_complex=True,
            normalized=False,
            onesided=True,
        )
        fake_spec = torch.stft(
            x_hat_1d,
            n_fft=size,
            hop_length=max(size // 4, 1),
            win_length=size,
            window=window,
            return_complex=True,
            normalized=False,
            onesided=True,
        )
        total = total + F.l1_loss(fake_spec.abs(), real_spec.abs())
    return total / max(len(fft_sizes), 1)


# ---------------------------------------------------------------------------
# Feature-matching loss  (L1-based, from discriminator feature maps)
# ---------------------------------------------------------------------------

def feature_matching_loss(
    fmaps_real: List[List[torch.Tensor]],
    fmaps_fake: List[List[torch.Tensor]],
) -> torch.Tensor:
    """
    L1 feature-matching loss across all discriminator layers and scales.

    For each STFT discriminator and each convolutional feature map, we
    measure the L1 distance between the feature activations produced by
    the real and generated audio.

    Args:
        fmaps_real: list-of-lists of tensors from discriminator(x_real)
                    outer: per discriminator, inner: per layer
        fmaps_fake: same structure but for discriminator(x_hat)

    Returns:
        loss: scalar tensor
    """
    total = torch.tensor(0.0, device=fmaps_real[0][0].device)
    n_terms = 0
    for fmap_r, fmap_f in zip(fmaps_real, fmaps_fake):
        for fr, ff in zip(fmap_r, fmap_f):
            total = total + F.l1_loss(ff, fr.detach())
            n_terms += 1

    return total / max(n_terms, 1)


# ---------------------------------------------------------------------------
# Discriminator losses  (least-squares GAN formulation, used for disc update)
# ---------------------------------------------------------------------------

def discriminator_loss(
    logits_real: List[torch.Tensor],
    logits_fake: List[torch.Tensor],
) -> torch.Tensor:
    """
    Least-squares discriminator loss:
        L_D = mean( (D(x) - 1)² ) + mean( D(G(z))² )

    Args:
        logits_real: list of score maps for real audio (one per discriminator)
        logits_fake: list of score maps for generated audio

    Returns:
        loss: scalar tensor
    """
    loss = logits_real[0].new_zeros(())
    for lr, lf in zip(logits_real, logits_fake):
        loss = loss + F.relu(1.0 - lr).mean() + F.relu(1.0 + lf).mean()
    return loss / max(len(logits_real), 1)


def generator_adversarial_loss(logits_fake: List[torch.Tensor]) -> torch.Tensor:
    """
    Least-squares generator loss:
        L_G_adv = mean( (D(G(z)) - 1)² )

    Args:
        logits_fake: list of score maps for generated audio

    Returns:
        loss: scalar tensor
    """
    loss = logits_fake[0].new_zeros(())
    for lf in logits_fake:
        loss = loss - lf.mean()
    return loss / max(len(logits_fake), 1)


# ---------------------------------------------------------------------------
# Combined codec loss  (generator side only)
# ---------------------------------------------------------------------------

def codec_loss(
    x_real: torch.Tensor,
    x_hat: torch.Tensor,
    fmaps_real: List[List[torch.Tensor]],
    fmaps_fake: List[List[torch.Tensor]],
    logits_fake: List[torch.Tensor],
    vq_loss: torch.Tensor,
    asr_loss: torch.Tensor,
    step: int,
    w_feat: float = 1.0,
    w_adv: float = 0.0,
    w_vq: float = 0.1,
    w_asr: float = 1.0,
    rec_initial_weight: float = 1.0,
    rec_decay_steps: float | None = None,
    rec_decay_base: float = 0.9999,
) -> Tuple[torch.Tensor, dict]:
    """
    Total generator-side loss for one training step.

    Args:
        x_real:        (B, 1, T) real waveform
        x_hat:         (B, 1, T) reconstructed waveform
        fmaps_real:    feature maps from discriminator applied to real audio
        fmaps_fake:    feature maps from discriminator applied to generated audio
        logits_fake:   discriminator scores for generated audio
        vq_loss:       VQ codebook + commitment loss (from DualQuantizer)
        asr_loss:      ASR distillation loss
        step:          current training step  (for decay schedule)
        w_feat:        weight for feature-matching loss
        w_adv:         weight for generator adversarial loss
        w_vq:          weight for VQ loss
        w_asr:         weight for ASR distillation loss
        rec_initial_weight: λ₀ for reconstruction loss decay
        rec_decay_steps:    τ  for reconstruction loss decay

    Returns:
        total_loss: scalar tensor
        log_dict:   dict with individual loss components for logging
    """
    l_rec, rec_weight = reconstruction_loss(
        x_real, x_hat, step,
        initial_weight=rec_initial_weight,
        decay_steps=rec_decay_steps,
        decay_base=rec_decay_base,
    )
    l_stft = stft_magnitude_loss(x_real, x_hat) * rec_weight
    l_feat = w_feat * feature_matching_loss(fmaps_real, fmaps_fake)
    l_adv  = w_adv * generator_adversarial_loss(logits_fake) if w_adv else x_real.new_zeros(())
    l_vq   = w_vq   * vq_loss
    l_asr  = w_asr  * asr_loss

    total = l_rec + l_stft + l_feat + l_adv + l_vq + l_asr

    log_dict = {
        "loss/total":          total.item(),
        "loss/l1":             l_rec.item(),
        "loss/stft":           l_stft.item(),
        "loss/rec_weight":     rec_weight,
        "loss/feat_match":     l_feat.item(),
        "loss/adv":            l_adv.item(),
        "loss/vq":             l_vq.item(),
        "loss/asr":            l_asr.item(),
    }
    return total, log_dict
