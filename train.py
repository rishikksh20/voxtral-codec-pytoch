"""
Voxtral Codec – End-to-End Training Script

Training philosophy:
  The codec (encoder + quantizer + decoder) is the "generator" G.
  The multi-resolution STFT discriminator is D.

  Each training step has two phases:

  ┌─ GENERATOR STEP ──────────────────────────────────────────────────┐
  │  x_real → Encoder → z (292-dim pre-quant latent)                  │
  │                         ├─ z_sem (256-dim) → VQ  → z_sem_q        │
  │                         └─ z_ac  ( 36-dim) → FSQ → z_ac_q         │
  │                                              z_q = [z_sem_q|z_ac_q]│
  │  z_q → Decoder → x_hat                                            │
  │                                                                    │
  │  x_real, x_hat → D (frozen) → fmaps_real, fmaps_fake             │
  │                                                                    │
  │  L_total = λ(t)·L1(x_hat, x_real)   ← decaying reconstruction    │
  │          + L1(fmaps_fake, fmaps_real) ← feature-matching          │
  │          + VQ commitment loss                                      │
  │          + MSE(proj(z_sem), Whisper(x_real)) ← ASR distillation   │
  └────────────────────────────────────────────────────────────────────┘

  ┌─ DISCRIMINATOR STEP ──────────────────────────────────────────────┐
  │  x_real, x_hat.detach() → D (trainable)                           │
  │  L_D = LS-GAN(real→1, fake→0)                                     │
  └────────────────────────────────────────────────────────────────────┘

Usage example:
  python train.py \\
      --data_dir /path/to/wav24k \\
      --batch_size 8 \\
      --max_steps 400000 \\
      --save_dir ./checkpoints
"""

import argparse
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Simple audio dataset
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """
    Loads .wav/.flac files from a directory tree, crops/pads to a fixed length,
    and returns mono 24 kHz waveforms as (1, T) tensors in [-1, 1].
    """

    def __init__(
        self,
        data_dir: str,
        segment_samples: int = 24_000 * 4,  # 4-second clips
        sample_rate: int = 24_000,
    ) -> None:
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError("torchaudio is required for the dataset loader.") from exc

        import glob

        self.files = sorted(
            glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
            + glob.glob(os.path.join(data_dir, "**/*.flac"), recursive=True)
        )
        if not self.files:
            raise FileNotFoundError(f"No .wav/.flac files found in {data_dir}")

        self.segment = segment_samples
        self.sample_rate = sample_rate
        self._torchaudio = torchaudio

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import torchaudio.functional as TAF

        path = self.files[idx]
        waveform, sr = self._torchaudio.load(path)

        if sr != self.sample_rate:
            waveform = TAF.resample(waveform, sr, self.sample_rate)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        T = waveform.shape[-1]
        if T >= self.segment:
            start = random.randint(0, T - self.segment)
            waveform = waveform[:, start : start + self.segment]
        else:
            waveform = nn.functional.pad(waveform, (0, self.segment - T))

        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        return waveform  # (1, segment)


# ---------------------------------------------------------------------------
# Core training step  (generator + discriminator, in that order)
# ---------------------------------------------------------------------------

def generator_step(
    x_real: torch.Tensor,
    model: nn.Module,
    disc: nn.Module,
    asr_loss_fn: nn.Module,
    step: int,
    w_feat: float,
    w_adv: float,
    w_vq: float,
    w_asr: float,
    rec_initial_weight: float,
    rec_decay_steps: float,
    disc_start_step: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Full generator-side training step.

    Pipeline:
      x_real → Encoder → z → DualQuantizer → z_q → Decoder → x_hat
      x_real, x_hat → Discriminator (frozen) → feature maps
      Compute: L_rec + L_feat + L_adv + L_vq + L_asr

    Returns:
        g_loss:  total generator loss (scalar)
        x_hat:   reconstructed waveform (for reuse in discriminator step)
        log_dict: per-component loss values for logging
    """
    from voxtral_codec.losses import (
        reconstruction_loss,
        stft_magnitude_loss,
        feature_matching_loss,
        generator_adversarial_loss,
    )

    # ── Encode → Quantize → Decode ──────────────────────────────────────────
    model_out = model.forward_with_details(x_real)
    x_hat = model_out["x_hat"]
    z = model_out["z"]
    z_sem = model_out["semantic_q"]
    vq_loss = model_out["vq_loss"]

    # ── Discriminator feature maps (weights frozen during G step) ───────────
    disc.eval()
    with torch.no_grad():
        disc_real_out = disc(x_real)
    disc_fake_out = disc(x_hat)

    fmaps_real  = [fmaps  for _, fmaps  in disc_real_out]
    fmaps_fake  = [fmaps  for _, fmaps  in disc_fake_out]
    logits_fake = [logits for logits, _ in disc_fake_out]

    # ── ASR distillation loss (on pre-quantization semantic latent) ─────────
    l_asr = w_asr * asr_loss_fn(x_real, z_sem)

    # ── Reconstruction losses (paper uses the same decay for L1 and STFT) ──
    l_rec, rec_weight = reconstruction_loss(
        x_real, x_hat, step,
        initial_weight=rec_initial_weight,
        decay_steps=rec_decay_steps,
    )
    l_stft = rec_weight * stft_magnitude_loss(x_real, x_hat)

    # ── Feature-matching loss (L1 on discriminator intermediate features) ───
    l_feat = w_feat * feature_matching_loss(fmaps_real, fmaps_fake)

    # ── Generator adversarial loss (active only after disc warm-up) ─────────
    l_adv = (
        w_adv * generator_adversarial_loss(logits_fake)
        if step >= disc_start_step
        else torch.zeros(1, device=x_real.device)
    )

    # ── VQ commitment + codebook loss ───────────────────────────────────────
    l_vq = w_vq * vq_loss

    g_loss = l_rec + l_stft + l_feat + l_adv + l_vq + l_asr

    log_dict = {
        "loss/total":          g_loss.item(),
        "loss/l1":            l_rec.item(),
        "loss/stft":          l_stft.item(),
        "loss/rec_weight":     rec_weight,
        "loss/feat_match":     l_feat.item(),
        "loss/adv_g":          l_adv.item(),
        "loss/vq":             l_vq.item(),
        "loss/asr":            l_asr.item(),
    }
    return g_loss, x_hat, log_dict


def discriminator_step(
    x_real: torch.Tensor,
    x_hat: torch.Tensor,
    disc: nn.Module,
) -> Tuple[torch.Tensor, float]:
    """
    Discriminator training step.

    Reuses the already-generated x_hat from the generator step (detached).
    Uses LS-GAN: L_D = E[(D(real)-1)²] + E[D(fake)²]

    Returns:
        d_loss: discriminator loss (scalar)
        d_loss_val: Python float for logging
    """
    from voxtral_codec.losses import discriminator_loss

    disc.train()
    disc_real_out = disc(x_real)
    disc_fake_out = disc(x_hat.detach())      # detach: stop grads into G

    logits_real = [logits for logits, _ in disc_real_out]
    logits_fake = [logits for logits, _ in disc_fake_out]

    d_loss = discriminator_loss(logits_real, logits_fake)
    return d_loss, d_loss.item()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    step: int,
    model: nn.Module,
    disc: nn.Module,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_step{step:07d}.pt")
    torch.save(
        {
            "step":             step,
            "model_state_dict": model.state_dict(),
            "disc_state_dict":  disc.state_dict(),
            "opt_g_state_dict": opt_g.state_dict(),
            "opt_d_state_dict": opt_d.state_dict(),
        },
        path,
    )
    print(f"  Saved checkpoint → {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    disc: nn.Module,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    disc.load_state_dict(ckpt["disc_state_dict"])
    opt_g.load_state_dict(ckpt["opt_g_state_dict"])
    opt_d.load_state_dict(ckpt["opt_d_state_dict"])
    step = ckpt["step"]
    print(f"  Resumed from {path}  (step {step})")
    return step


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Voxtral Codec")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing .wav/.flac files (searched recursively)")
    p.add_argument("--save_dir",   type=str, default="./checkpoints")
    p.add_argument("--resume",     type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--segment_sec", type=float, default=4.0,
                   help="Audio segment length in seconds")
    p.add_argument("--sample_rate", type=int, default=24_000)
    p.add_argument("--max_steps",  type=int, default=400_000)
    # Optimiser
    p.add_argument("--lr_g", type=float, default=3e-4,
                   help="Generator (codec) learning rate")
    p.add_argument("--lr_d", type=float, default=1e-4,
                   help="Discriminator learning rate")
    # Loss weights
    p.add_argument("--w_feat", type=float, default=1.0,
                   help="Weight for feature-matching loss")
    p.add_argument("--w_adv",  type=float, default=0.0,
                   help="Optional weight for generator adversarial loss (paper uses feature matching instead)")
    p.add_argument("--w_vq",   type=float, default=0.1,
                   help="Weight for VQ commitment loss")
    p.add_argument("--w_asr",  type=float, default=1.0,
                   help="Weight for ASR distillation loss")
    p.add_argument("--rec_initial_weight", type=float, default=1.0,
                   help="Initial weight λ₀ for exponentially decaying reconstruction loss")
    p.add_argument("--rec_decay_steps", type=float, default=50_000.0,
                   help="Decay constant τ (steps) for reconstruction loss")
    p.add_argument("--disc_start_step", type=int, default=50_000,
                   help="Delay discriminator + adversarial losses by this many steps")
    # ASR distillation
    p.add_argument("--use_asr", action="store_true",
                   help="Enable Whisper ASR distillation loss")
    p.add_argument("--whisper_model", type=str, default="openai/whisper-base")
    # Model arch
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--n_transformer_layers", type=int, default=2,
                   help="Number of transformer layers per encoder/decoder block")
    # Misc
    p.add_argument("--log_every",  type=int, default=100)
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------------
    segment_samples = int(args.segment_sec * args.sample_rate)
    dataset = AudioDataset(
        data_dir=args.data_dir,
        segment_samples=segment_samples,
        sample_rate=args.sample_rate,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} files, batch_size={args.batch_size}")

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    from voxtral_codec import VoxtralCodec, MultiResolutionDiscriminator
    from voxtral_codec.asr_distillation import ASRDistillationLoss, NoOpASRLoss

    model = VoxtralCodec(
        hidden_dim=args.hidden_dim,
        n_transformer_layers=args.n_transformer_layers,
        sample_rate=args.sample_rate,
    ).to(device)
    print(model.info())

    disc = MultiResolutionDiscriminator().to(device)
    print(
        f"Discriminator: "
        f"{sum(p.numel() for p in disc.parameters()) / 1e6:.1f}M params"
    )

    asr_loss_fn: nn.Module
    if args.use_asr:
        asr_loss_fn = ASRDistillationLoss(
            whisper_model_name=args.whisper_model,
            codec_sample_rate=args.sample_rate,
        ).to(device)
        print(f"ASR distillation enabled (whisper={args.whisper_model})")
    else:
        asr_loss_fn = NoOpASRLoss().to(device)

    # ------------------------------------------------------------------
    # Optimisers
    #   G optimiser covers codec + (optional) ASR projection head.
    #   D optimiser covers only the discriminator.
    # ------------------------------------------------------------------
    opt_g = optim.AdamW(
        list(model.parameters()) + list(asr_loss_fn.parameters()),
        lr=args.lr_g, betas=(0.8, 0.99), weight_decay=1e-4,
    )
    opt_d = optim.AdamW(
        disc.parameters(),
        lr=args.lr_d, betas=(0.8, 0.99), weight_decay=1e-4,
    )

    # ------------------------------------------------------------------
    # Optionally resume from checkpoint
    # ------------------------------------------------------------------
    step = 0
    if args.resume:
        step = load_checkpoint(args.resume, model, disc, opt_g, opt_d, device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    running_log: Dict[str, float] = {}
    t0 = time.time()

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            x_real = batch.to(device)  # (B, 1, T)

            # ── Generator step ─────────────────────────────────────────────
            model.train()
            g_loss, x_hat, log_dict = generator_step(
                x_real=x_real,
                model=model,
                disc=disc,
                asr_loss_fn=asr_loss_fn,
                step=step,
                w_feat=args.w_feat,
                w_adv=args.w_adv,
                w_vq=args.w_vq,
                w_asr=args.w_asr,
                rec_initial_weight=args.rec_initial_weight,
                rec_decay_steps=args.rec_decay_steps,
                disc_start_step=args.disc_start_step,
            )
            opt_g.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_g.step()

            # ── Discriminator step (after warm-up) ─────────────────────────
            if step >= args.disc_start_step:
                # Reuse x_hat from generator step (detached) — no extra fwd pass
                d_loss, d_loss_val = discriminator_step(x_real, x_hat, disc)
                opt_d.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                opt_d.step()
                log_dict["loss/disc"] = d_loss_val

            # ── Logging ────────────────────────────────────────────────────
            for k, v in log_dict.items():
                running_log[k] = running_log.get(k, 0.0) + v

            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                avg = {k: v / args.log_every for k, v in running_log.items()}
                log_str = "  ".join(
                    f"{k.split('/')[-1]}={v:.4f}" for k, v in avg.items()
                )
                print(
                    f"[step {step:7d}] {log_str}  "
                    f"({args.log_every / elapsed:.1f} steps/s)"
                )
                running_log = {}
                t0 = time.time()

            if step % args.save_every == 0:
                save_checkpoint(step, model, disc, opt_g, opt_d, args.save_dir)

    save_checkpoint(step, model, disc, opt_g, opt_d, args.save_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
