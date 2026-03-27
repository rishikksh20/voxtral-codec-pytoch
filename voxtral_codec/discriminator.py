"""
Multi-Resolution STFT Discriminator for Voxtral Codec.

Uses 8 different STFT configurations (n_fft sizes). For each configuration:
  1. Compute STFT of the waveform → complex spectrogram
  2. Stack real and imaginary parts → 2-channel input
  3. Pass through a stack of strided Conv2d layers
  4. Output a score map (logits) and intermediate feature maps

The feature maps are used for the L1 feature-matching adversarial loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Utility: STFT helper
# ---------------------------------------------------------------------------

def _stft(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    """
    Compute STFT and return stacked (real, imag) as a 2-channel tensor.

    Args:
        x: (B, T) waveform

    Returns:
        spec: (B, 2, freq, time) – channels are [real, imag]
    """
    window = torch.hann_window(win_length, device=x.device)
    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        normalized=False,
        onesided=True,
    )  # (B, freq, time) complex
    # Stack real and imaginary parts
    spec = torch.stack([stft.real, stft.imag], dim=1)  # (B, 2, freq, time)
    return spec


# ---------------------------------------------------------------------------
# Single STFT Discriminator
# ---------------------------------------------------------------------------

class STFTDiscriminator(nn.Module):
    """
    A single STFT-based discriminator.

    Architecture: strided Conv2d blocks on the magnitude/phase spectrogram,
    followed by a final 1×1 conv producing per-patch scores.

    Args:
        n_fft:       FFT size
        hop_length:  hop size (defaults to n_fft // 4)
        win_length:  window length (defaults to n_fft)
        channels:    base number of channels
        n_layers:    number of strided conv blocks
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int | None = None,
        win_length: int | None = None,
        channels: int = 32,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft

        # Input: 2 channels (real + imag)
        in_ch = 2
        layers = []
        for i in range(n_layers):
            out_ch = min(channels * (2 ** i), 512)
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )
            in_ch = out_ch

        # One more conv without striding
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), padding=(1, 1)),
                nn.LeakyReLU(0.1, inplace=True),
            )
        )
        self.convs = nn.ModuleList(layers)

        # Final projection to 1 channel (score map)
        self.conv_post = nn.Conv2d(in_ch, 1, kernel_size=(3, 3), padding=(1, 1))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, 1, T) waveform

        Returns:
            logits:  (B, 1, freq', time') score map
            fmaps:   list of intermediate feature maps for feature-matching loss
        """
        x = x.squeeze(1)  # (B, T)
        spec = _stft(x, self.n_fft, self.hop_length, self.win_length)  # (B, 2, F, T')

        fmaps: List[torch.Tensor] = []
        h = spec
        for conv in self.convs:
            h = conv(h)
            fmaps.append(h)

        logits = self.conv_post(h)  # (B, 1, F'', T'')
        return logits, fmaps


# ---------------------------------------------------------------------------
# Multi-Resolution Discriminator (8 STFT sizes)
# ---------------------------------------------------------------------------

# Eight STFT configurations covering a wide range of time-frequency resolutions
_DEFAULT_STFT_CONFIGS: List[dict] = [
    {"n_fft": 2048, "hop_length": 512,  "win_length": 2048},
    {"n_fft": 1024, "hop_length": 256,  "win_length": 1024},
    {"n_fft": 512,  "hop_length": 128,  "win_length": 512},
    {"n_fft": 256,  "hop_length": 64,   "win_length": 256},
    {"n_fft": 128,  "hop_length": 32,   "win_length": 128},
    {"n_fft": 4096, "hop_length": 1024, "win_length": 4096},
    {"n_fft": 8192, "hop_length": 2048, "win_length": 8192},
    {"n_fft": 300,  "hop_length": 75,   "win_length": 300},
]


class MultiResolutionDiscriminator(nn.Module):
    """
    Multi-resolution STFT discriminator with 8 different STFT sizes.

    Returns a list of (logits, fmaps) pairs – one per STFT configuration.
    """

    def __init__(
        self,
        stft_configs: List[dict] | None = None,
        channels: int = 32,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        if stft_configs is None:
            stft_configs = _DEFAULT_STFT_CONFIGS

        assert len(stft_configs) == 8, "Expected exactly 8 STFT configurations"

        self.discriminators = nn.ModuleList(
            [
                STFTDiscriminator(
                    n_fft=cfg["n_fft"],
                    hop_length=cfg.get("hop_length"),
                    win_length=cfg.get("win_length"),
                    channels=channels,
                    n_layers=n_layers,
                )
                for cfg in stft_configs
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            x: (B, 1, T) waveform (real or generated)

        Returns:
            results: list of (logits, fmaps) – one entry per discriminator
        """
        return [disc(x) for disc in self.discriminators]
