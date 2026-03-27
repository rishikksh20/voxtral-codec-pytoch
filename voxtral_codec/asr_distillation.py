"""Whisper-based semantic distillation for Voxtral Codec."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASRDistillationLoss(nn.Module):
    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-base",
        semantic_dim: int = 256,
        sample_rate: int = 16_000,
        codec_sample_rate: int = 24_000,
        selected_heads: Sequence[int] | None = None,
        max_new_tokens: int = 64,
        median_filter_size: int = 3,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.codec_sample_rate = codec_sample_rate
        self.selected_heads = tuple(selected_heads) if selected_heads is not None else None
        self.max_new_tokens = max_new_tokens
        self.median_filter_size = median_filter_size

        # ------------------------------------------------------------------
        # Load frozen Whisper encoder + feature extractor
        # ------------------------------------------------------------------
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as exc:
            raise ImportError(
                "The transformers package is required for ASR distillation."
            ) from exc

        self.processor = WhisperProcessor.from_pretrained(whisper_model_name)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        for param in self.whisper.parameters():
            param.requires_grad = False
        self.whisper.eval()
        whisper_dim = self.whisper.config.d_model
        self.projection = nn.Linear(semantic_dim, whisper_dim)

    def _resample(self, x: torch.Tensor) -> torch.Tensor:
        if self.codec_sample_rate == self.sample_rate:
            return x
        target = int(round(x.shape[-1] * self.sample_rate / self.codec_sample_rate))
        return F.interpolate(x, size=target, mode="linear", align_corners=False)

    def _median_filter(self, attn: torch.Tensor) -> torch.Tensor:
        if self.median_filter_size <= 1:
            return attn
        pad = self.median_filter_size // 2
        padded = F.pad(attn, (pad, pad), mode="replicate")
        windows = padded.unfold(-1, self.median_filter_size, 1)
        return windows.median(dim=-1).values

    def _compute_alignment(self, cross_attn: torch.Tensor, codec_frames: int) -> torch.Tensor:
        if self.selected_heads is not None:
            cross_attn = cross_attn[:, self.selected_heads, :, :]
        attn = cross_attn.mean(dim=1)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attn = self._median_filter(attn)
        attn = F.interpolate(attn, size=codec_frames, mode="linear", align_corners=False)
        return attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(
        self,
        x_real: torch.Tensor,
        z_semantic: torch.Tensor,
    ) -> torch.Tensor:
        """Distill quantized semantic codec states against Whisper decoder states."""
        x_16k = self._resample(x_real).squeeze(1)
        inputs = self.processor.feature_extractor(
            [sample.detach().cpu().float().numpy() for sample in x_16k],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="longest",
        )
        input_features = inputs.input_features.to(x_real.device)

        with torch.no_grad():
            generated = self.whisper.generate(
                input_features=input_features,
                max_new_tokens=self.max_new_tokens,
            )
            if generated.shape[1] < 2:
                bos = self.whisper.config.decoder_start_token_id
                generated = torch.tensor([[bos, bos]], device=x_real.device).repeat(input_features.shape[0], 1)
            outputs = self.whisper.model(
                input_features=input_features,
                decoder_input_ids=generated[:, :-1],
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
                use_cache=False,
            )
            decoder_hidden = outputs.decoder_hidden_states[-1].detach()
            cross_attn = outputs.cross_attentions[-1].detach()

        z_proj = self.projection(z_semantic.transpose(1, 2))
        alignment = self._compute_alignment(cross_attn, codec_frames=z_proj.shape[1])
        aligned_codec = alignment @ z_proj
        aligned_codec = F.normalize(aligned_codec, dim=-1)
        decoder_hidden = F.normalize(decoder_hidden, dim=-1)
        return 1.0 - (aligned_codec * decoder_hidden).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Dummy no-op distillation loss  (used when Whisper is unavailable)
# ---------------------------------------------------------------------------

class NoOpASRLoss(nn.Module):
    def forward(self, x_real: torch.Tensor, z_semantic: torch.Tensor) -> torch.Tensor:
        return x_real.new_zeros(())
