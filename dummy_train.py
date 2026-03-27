"""
Voxtral Codec – Self-Contained Dummy Training Loop
====================================================
Demonstrates the full training pipeline end-to-end using **random tensors**
(no real audio required).  Run this script to verify the architecture and
training logic work correctly on your machine.

Usage:
    python dummy_train.py           # CPU, 5 steps, tiny model
    python dummy_train.py --steps 20 --device cuda

What this demonstrates:
  1. 🏗️  Architecture: Encoder → DualQuantizer → Decoder + Discriminator
  2. 🧠  Training pipeline: how every component connects
  3. ⚙️  Full training step: all losses in one pass (rec + feat_match + vq + asr)
  4. 🔄  Alternating updates: generator step → discriminator step
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Tiny model configuration  (runs in seconds on CPU)
# ---------------------------------------------------------------------------
TINY_CFG = dict(
    hidden_dim=64,
    latent_dim=292,        # must stay 256 + 36
    semantic_dim=256,
    acoustic_dim=36,
    patch_stride=240,
    encoder_strides=(2, 2, 2, 1),
    decoder_strides=(1, 2, 2, 2),
    n_residual=1,
    dilations=(1,),
    n_transformer_layers=1,
    n_heads=4,
    ffn_dim=128,
    window_size=8,
    codebook_size=256,     # small codebook for demo
    fsq_levels=21,
    sample_rate=24_000,
)

# Audio parameters (must be consistent with model strides)
# T = patch_stride × cnn_stride × n_latent_frames  =  240 × 8 × 10  =  19 200
BATCH        = 2
N_LAT_FRAMES = 10
SAMPLE_RATE  = 24_000
T_AUDIO      = 240 * 8 * N_LAT_FRAMES          # 19 200 samples ≈ 0.8 s at 24 kHz

# STFT configs sized to fit T_AUDIO (8 discriminators)
SMALL_STFT_CONFIGS = [
    {"n_fft": 256,  "hop_length": 64,  "win_length": 256},
    {"n_fft": 128,  "hop_length": 32,  "win_length": 128},
    {"n_fft": 512,  "hop_length": 128, "win_length": 512},
    {"n_fft": 64,   "hop_length": 16,  "win_length": 64},
    {"n_fft": 1024, "hop_length": 256, "win_length": 1024},
    {"n_fft": 128,  "hop_length": 64,  "win_length": 128},
    {"n_fft": 256,  "hop_length": 128, "win_length": 256},
    {"n_fft": 512,  "hop_length": 256, "win_length": 512},
]


# ---------------------------------------------------------------------------
# One complete training step (generator + discriminator)
# ---------------------------------------------------------------------------

def run_training_step(
    step: int,
    x_real: torch.Tensor,            # (B, 1, T) simulated audio batch
    model: nn.Module,                # VoxtralCodec
    disc: nn.Module,                 # MultiResolutionDiscriminator
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    disc_start_step: int = 0,        # 0 → discriminator active from step 0
    w_feat: float = 1.0,
    w_adv: float = 0.1,
    w_vq: float = 1.0,
    rec_initial_weight: float = 1.0,
    rec_decay_steps: float = 10.0,   # fast decay for demo
) -> dict:
    """
    ┌─ GENERATOR STEP ──────────────────────────────────────────────────┐
    │                                                                    │
    │  x_real ──► Encoder ──► z (292-dim)                               │
    │                         ├─ z_sem (256) ──► VQ  ──► z_sem_q        │
    │                         └─ z_ac  ( 36) ──► FSQ ──► z_ac_q         │
    │                                  z_q = concat[z_sem_q, z_ac_q]    │
    │  z_q ──► Decoder ──► x_hat                                        │
    │                                                                    │
    │  x_real }                                                          │
    │  x_hat  }──► Discriminator (frozen weights) ──► feature maps      │
    │                                                                    │
    │  Loss = λ(t)·L1(x_hat, x_real)  [decaying reconstruction]        │
    │       + L1(fmaps_fake, fmaps_real) [feature matching]             │
    │       + VQ commitment loss                                         │
    │       (+ ASR distillation, omitted here for simplicity)           │
    │                                                                    │
    │  Backprop → update codec params                                    │
    └────────────────────────────────────────────────────────────────────┘

    ┌─ DISCRIMINATOR STEP ──────────────────────────────────────────────┐
    │                                                                    │
    │  x_real     }──► Discriminator (trainable)                        │
    │  x_hat.detach() }                       ──► LS-GAN loss           │
    │  (Reuses x_hat from G step — no second codec forward pass!)       │
    │                                                                    │
    │  Backprop → update discriminator params                           │
    └────────────────────────────────────────────────────────────────────┘
    """
    import math
    import torch.nn.functional as F
    from voxtral_codec.losses import (
        reconstruction_loss,
        feature_matching_loss,
        generator_adversarial_loss,
        discriminator_loss,
    )

    # ── GENERATOR STEP ──────────────────────────────────────────────────────
    model.train()
    disc.eval()       # freeze discriminator batch-norm / dropout during G step

    # 1. Encode → Quantize → Decode
    x_hat, z, _, _, vq_loss = model(x_real)

    # 2. Run discriminator on real + fake (detached disc weights for gradient flow)
    with torch.no_grad():
        disc_real_out = disc(x_real)
    disc_fake_out = disc(x_hat)

    fmaps_real  = [fmaps  for _, fmaps  in disc_real_out]
    fmaps_fake  = [fmaps  for _, fmaps  in disc_fake_out]
    logits_fake = [logits for logits, _ in disc_fake_out]

    # 3. Compute all generator losses
    l_rec, rec_w = reconstruction_loss(
        x_real, x_hat, step,
        initial_weight=rec_initial_weight,
        decay_steps=rec_decay_steps,
    )
    l_feat = w_feat * feature_matching_loss(fmaps_real, fmaps_fake)
    l_vq   = w_vq   * vq_loss
    l_adv  = (
        w_adv * generator_adversarial_loss(logits_fake)
        if step >= disc_start_step
        else torch.zeros(1, device=x_real.device)
    )

    g_loss = l_rec + l_feat + l_vq + l_adv

    # 4. Update codec
    opt_g.zero_grad()
    g_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_g.step()

    # ── DISCRIMINATOR STEP ──────────────────────────────────────────────────
    disc.train()

    # Reuse x_hat from above — detach to stop gradients flowing into the codec.
    # After g_loss.backward() the compute graph is freed, but explicit detach
    # is the correct practice to ensure the discriminator update only optimises D.
    disc_real_out_d = disc(x_real)
    disc_fake_out_d = disc(x_hat.detach())

    logits_real_d = [logits for logits, _ in disc_real_out_d]
    logits_fake_d = [logits for logits, _ in disc_fake_out_d]

    d_loss = discriminator_loss(logits_real_d, logits_fake_d)

    opt_d.zero_grad()
    d_loss.backward()
    nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
    opt_d.step()

    return {
        "g_total":   g_loss.item(),
        "rec":       l_rec.item(),
        "rec_w":     rec_w,
        "feat":      l_feat.item(),
        "vq":        l_vq.item(),
        "adv_g":     l_adv.item(),
        "disc":      d_loss.item(),
    }


# ---------------------------------------------------------------------------
# Dummy training loop
# ---------------------------------------------------------------------------

def dummy_train(n_steps: int = 5, device_str: str = "cpu") -> None:
    """
    Run `n_steps` training iterations with random (fake) audio data,
    printing loss values at every step.

    This proves the full pipeline is wired up correctly without requiring
    any real audio files.
    """
    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print("  Voxtral Codec – Dummy Training Loop")
    print(f"{'='*60}")
    print(f"  Device       : {device}")
    print(f"  Batch size   : {BATCH}")
    print(f"  Audio length : {T_AUDIO} samples ({T_AUDIO/SAMPLE_RATE:.2f}s at {SAMPLE_RATE} Hz)")
    print(f"  Steps        : {n_steps}")
    print(f"{'='*60}\n")

    # ── Build tiny models ────────────────────────────────────────────────────
    from voxtral_codec import VoxtralCodec, MultiResolutionDiscriminator

    print("Building codec (tiny config)...")
    model = VoxtralCodec(**TINY_CFG).to(device)
    print(model.info())

    print("Building discriminator (8 STFT scales, small FFT sizes)...")
    disc = MultiResolutionDiscriminator(stft_configs=SMALL_STFT_CONFIGS).to(device)
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"  Discriminator: {disc_params / 1e6:.2f}M params\n")

    # ── Optimisers ───────────────────────────────────────────────────────────
    opt_g = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.8, 0.99))
    opt_d = optim.AdamW(disc.parameters(),  lr=1e-4, betas=(0.8, 0.99))

    # ── Training loop ────────────────────────────────────────────────────────
    print("Step │ g_total │   rec   │ feat    │   vq    │ adv_g   │  disc")
    print("─────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────")

    for step in range(n_steps):
        # Simulate a batch of raw audio waveforms in [-1, 1]
        x_real = torch.randn(BATCH, 1, T_AUDIO, device=device).clamp(-1, 1)

        t0 = time.time()
        logs = run_training_step(
            step=step,
            x_real=x_real,
            model=model,
            disc=disc,
            opt_g=opt_g,
            opt_d=opt_d,
            disc_start_step=0,      # enable adversarial loss from step 0 in demo
            w_feat=1.0,
            w_adv=0.1,
            w_vq=1.0,
            rec_initial_weight=1.0,
            rec_decay_steps=10.0,   # fast decay so we see it change
        )
        elapsed = time.time() - t0

        print(
            f"  {step:2d} │ {logs['g_total']:7.4f} │ {logs['rec']:7.4f} │ "
            f"{logs['feat']:7.4f} │ {logs['vq']:7.4f} │ "
            f"{logs['adv_g']:7.4f} │ {logs['disc']:7.4f}  "
            f"({elapsed:.2f}s)"
        )

    print("\n✅ Dummy training loop completed successfully.")
    print("\nToken breakdown per frame (at 12.5 Hz):")
    print(f"  Semantic tokens : 1  (VQ codebook size = {TINY_CFG['codebook_size']})")
    print(f"  Acoustic tokens : {TINY_CFG['acoustic_dim']}  "
          f"(FSQ levels = {TINY_CFG['fsq_levels']} per dim)")
    print(f"  Total tokens    : {1 + TINY_CFG['acoustic_dim']} per frame\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Voxtral Codec dummy training loop")
    p.add_argument("--steps",  type=int, default=5,
                   help="Number of training steps to run (default: 5)")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device to run on: cpu or cuda (default: cpu)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dummy_train(n_steps=args.steps, device_str=args.device)
