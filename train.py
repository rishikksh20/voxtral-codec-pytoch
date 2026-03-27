"""
Voxtral Codec – End-to-End Training Script

Training objective:
  L_total = λ_rec(t) · L_rec          (exponentially decaying reconstruction)
           + L_feat_match              (L1 feature-matching from discriminator)
           + L_adv                     (generator adversarial, LS-GAN)
           + L_vq                      (VQ codebook + commitment)
           + L_asr                     (Whisper ASR distillation)

The discriminator is updated with an alternating optimiser schedule
(one discriminator step per generator step).

Usage example:
  python train.py \
      --data_dir /path/to/wav24k \
      --batch_size 8 \
      --max_steps 400000 \
      --save_dir ./checkpoints
"""

import argparse
import os
import random
import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Simple audio dataset
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """
    Loads .wav files from a directory, crops/pads to a fixed length, and
    returns mono 24 kHz waveforms as (1, T) tensors in [-1, 1].
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

        # Resample if needed
        if sr != self.sample_rate:
            waveform = TAF.resample(waveform, sr, self.sample_rate)

        # To mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Crop or pad to fixed length
        T = waveform.shape[-1]
        if T >= self.segment:
            start = random.randint(0, T - self.segment)
            waveform = waveform[:, start : start + self.segment]
        else:
            waveform = nn.functional.pad(waveform, (0, self.segment - T))

        # Normalise to [-1, 1]
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        return waveform  # (1, segment)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Voxtral Codec")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing .wav/.flac files (searched recursively)")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--segment_sec", type=float, default=4.0,
                   help="Audio segment length in seconds")
    p.add_argument("--sample_rate", type=int, default=24_000)
    p.add_argument("--max_steps", type=int, default=400_000)
    p.add_argument("--lr_g", type=float, default=3e-4, help="Generator learning rate")
    p.add_argument("--lr_d", type=float, default=1e-4, help="Discriminator learning rate")
    p.add_argument("--disc_start_step", type=int, default=50_000,
                   help="Delay discriminator training by this many steps")
    p.add_argument("--w_feat", type=float, default=1.0)
    p.add_argument("--w_adv",  type=float, default=0.1)
    p.add_argument("--w_vq",   type=float, default=1.0)
    p.add_argument("--w_asr",  type=float, default=1.0)
    p.add_argument("--rec_initial_weight", type=float, default=1.0,
                   help="Initial weight λ₀ for reconstruction loss")
    p.add_argument("--rec_decay_steps", type=float, default=50_000.0,
                   help="Decay constant τ for reconstruction loss (steps)")
    p.add_argument("--use_asr", action="store_true",
                   help="Enable Whisper ASR distillation loss")
    p.add_argument("--whisper_model", type=str, default="openai/whisper-base")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--n_transformer_layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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
            "step": step,
            "model_state_dict": model.state_dict(),
            "disc_state_dict": disc.state_dict(),
            "opt_g_state_dict": opt_g.state_dict(),
            "opt_d_state_dict": opt_d.state_dict(),
        },
        path,
    )
    print(f"  Saved checkpoint → {path}")


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
    from voxtral_codec.losses import (
        codec_loss, discriminator_loss, feature_matching_loss,
    )
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
    # Training loop
    # ------------------------------------------------------------------
    step = 0
    running_log: dict = {}
    t0 = time.time()

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            x_real = batch.to(device)  # (B, 1, T)

            # --------------------------------------------------------------
            # 1. Generator / codec step
            # --------------------------------------------------------------
            model.train()
            disc.eval()  # disc in eval during generator update

            x_hat, z, sem_idx, ac_codes, vq_loss = model(x_real)

            # Discriminator feature maps
            with torch.no_grad():
                disc_real_out = disc(x_real)
            disc_fake_out = disc(x_hat)

            fmaps_real = [fmaps for _, fmaps in disc_real_out]
            fmaps_fake = [fmaps for _, fmaps in disc_fake_out]
            logits_fake = [logits for logits, _ in disc_fake_out]

            # ASR distillation (uses pre-quantization semantic latent)
            z_semantic = z[:, : model.semantic_dim, :]
            asr_loss = asr_loss_fn(x_real, z_semantic)

            g_loss, log_dict = codec_loss(
                x_real=x_real,
                x_hat=x_hat,
                fmaps_real=fmaps_real,
                fmaps_fake=fmaps_fake,
                logits_fake=logits_fake,
                vq_loss=vq_loss,
                asr_loss=asr_loss,
                step=step,
                w_feat=args.w_feat,
                w_adv=args.w_adv if step >= args.disc_start_step else 0.0,
                w_vq=args.w_vq,
                w_asr=args.w_asr,
                rec_initial_weight=args.rec_initial_weight,
                rec_decay_steps=args.rec_decay_steps,
            )

            opt_g.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_g.step()

            # --------------------------------------------------------------
            # 2. Discriminator step  (after warm-up)
            # --------------------------------------------------------------
            if step >= args.disc_start_step:
                disc.train()

                with torch.no_grad():
                    x_hat_d = model(x_real)[0]  # detached generated audio

                disc_real_out_d = disc(x_real)
                disc_fake_out_d = disc(x_hat_d.detach())

                logits_real_d = [l for l, _ in disc_real_out_d]
                logits_fake_d = [l for l, _ in disc_fake_out_d]

                d_loss = discriminator_loss(logits_real_d, logits_fake_d)

                opt_d.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                opt_d.step()

                log_dict["loss/discriminator"] = d_loss.item()

            # --------------------------------------------------------------
            # Logging
            # --------------------------------------------------------------
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

    # Final checkpoint
    save_checkpoint(step, model, disc, opt_g, opt_d, args.save_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
