"""
ASR Distillation Loss for Voxtral Codec.

Voxtral uses a frozen Whisper encoder to provide soft supervision for the
semantic VQ codebook.  Instead of computing a hard transcript alignment,
we distil directly from Whisper's continuous hidden states, which capture
rich phonetic and prosodic information.

Architecture:
  1. A *frozen* Whisper encoder processes the real (target) audio and
     produces a sequence of hidden-state vectors.
  2. A lightweight *trainable* projection head maps the codec's semantic
     latent to the same dimensionality as the Whisper hidden states.
  3. The distillation loss is the MSE between the projected semantic latent
     and the (detached) Whisper hidden states.

The Whisper model is only used in evaluation mode and its parameters are
never updated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASRDistillationLoss(nn.Module):
    """
    Whisper-based ASR distillation loss.

    Args:
        whisper_model_name: HuggingFace model identifier for a Whisper encoder
                            (e.g. "openai/whisper-base").
        semantic_dim:       Dimensionality of the codec's semantic latent (256).
        projection_hidden:  Hidden dim of the projection MLP. Set to 0 to use
                            a single linear layer.
        sample_rate:        Audio sample rate expected by Whisper (16 000 Hz).
                            If the codec runs at 24 kHz, we resample on the fly.
        codec_sample_rate:  Sample rate of the codec waveforms (24 000 Hz).
    """

    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-base",
        semantic_dim: int = 256,
        projection_hidden: int = 512,
        sample_rate: int = 16_000,
        codec_sample_rate: int = 24_000,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.codec_sample_rate = codec_sample_rate

        # ------------------------------------------------------------------
        # Load frozen Whisper encoder + feature extractor
        # ------------------------------------------------------------------
        try:
            from transformers import WhisperModel, WhisperFeatureExtractor
        except ImportError as exc:
            raise ImportError(
                "The `transformers` package is required for ASR distillation. "
                "Install it with: pip install transformers"
            ) from exc

        whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.whisper_encoder = whisper.encoder
        # Freeze all Whisper parameters
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False
        self.whisper_encoder.eval()

        # Feature extractor converts raw waveform → log-mel spectrogram
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            whisper_model_name
        )

        whisper_dim = self.whisper_encoder.config.d_model

        # ------------------------------------------------------------------
        # Trainable projection head  semantic_dim → whisper_dim
        # ------------------------------------------------------------------
        if projection_hidden > 0:
            self.projection = nn.Sequential(
                nn.Linear(semantic_dim, projection_hidden),
                nn.GELU(),
                nn.Linear(projection_hidden, whisper_dim),
            )
        else:
            self.projection = nn.Linear(semantic_dim, whisper_dim)

    # ------------------------------------------------------------------
    # Resampling helper (no external dependency)
    # ------------------------------------------------------------------

    def _resample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resample waveform from codec_sample_rate to Whisper's sample_rate.
        Uses linear interpolation to avoid requiring torchaudio.

        Args:
            x: (B, 1, T) at codec_sample_rate

        Returns:
            (B, 1, T') at sample_rate
        """
        if self.codec_sample_rate == self.sample_rate:
            return x

        scale = self.sample_rate / self.codec_sample_rate
        T_out = int(round(x.shape[-1] * scale))
        # F.interpolate expects (B, C, T)
        x_resampled = F.interpolate(x, size=T_out, mode="linear", align_corners=False)
        return x_resampled

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_real: torch.Tensor,
        z_semantic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the ASR distillation loss.

        Args:
            x_real:     (B, 1, T)    real waveform at codec_sample_rate
            z_semantic: (B, 256, T_latent) semantic latent from encoder
                        (before or after VQ, teacher-forced on real audio)

        Returns:
            loss: scalar MSE between projected semantic latent and Whisper
                  hidden states (detached).
        """
        # 1. Resample to Whisper's expected sample rate (16 kHz)
        x_16k = self._resample(x_real).squeeze(1)  # (B, T')

        # 2. Compute log-mel spectrogram via WhisperFeatureExtractor
        #    The extractor expects a list of numpy arrays or CPU tensors.
        x_16k_cpu = x_16k.detach().cpu().float()
        inputs = self.feature_extractor(
            [x_16k_cpu[i].numpy() for i in range(x_16k_cpu.shape[0])],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="longest",
        )
        # log-mel features: (B, n_mels=80, time_frames)
        input_features = inputs.input_features.to(x_real.device)

        # 3. Get Whisper hidden states (no grad)
        with torch.no_grad():
            whisper_out = self.whisper_encoder(
                input_features=input_features,
                return_dict=True,
            )
            whisper_hidden = whisper_out.last_hidden_state  # (B, T_w, whisper_dim)
            whisper_hidden = whisper_hidden.detach()

        # 4. Project semantic latent: (B, 256, T_lat) → (B, T_lat, whisper_dim)
        z_perm = z_semantic.permute(0, 2, 1)  # (B, T_lat, 256)
        z_proj = self.projection(z_perm)      # (B, T_lat, whisper_dim)

        # 5. Align time dimensions via adaptive average pooling
        T_lat = z_proj.shape[1]
        T_w   = whisper_hidden.shape[1]

        if T_lat != T_w:
            # Pool whichever is longer to the shorter length
            if T_lat > T_w:
                z_proj = F.adaptive_avg_pool1d(
                    z_proj.transpose(1, 2), T_w
                ).transpose(1, 2)
            else:
                whisper_hidden = F.adaptive_avg_pool1d(
                    whisper_hidden.transpose(1, 2), T_lat
                ).transpose(1, 2)

        loss = F.mse_loss(z_proj, whisper_hidden)
        return loss


# ---------------------------------------------------------------------------
# Dummy no-op distillation loss  (used when Whisper is unavailable)
# ---------------------------------------------------------------------------

class NoOpASRLoss(nn.Module):
    """Returns a zero loss tensor; useful for ablations or when running without
    the `transformers` dependency."""

    def forward(
        self,
        x_real: torch.Tensor,
        z_semantic: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(0.0, device=x_real.device, requires_grad=False)
