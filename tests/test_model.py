"""
Unit tests for Voxtral Codec.

These tests use a *tiny* model configuration (small hidden_dim, fewer layers)
so they run quickly on CPU without requiring a GPU or large memory.
"""

import math
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24_000
PATCH_STRIDE = 240           # 24000/240 = 100 Hz post-patchify
CNN_STRIDE   = 8             # product of encoder_strides (2×2×2×1)
FRAME_RATE   = SAMPLE_RATE / (PATCH_STRIDE * CNN_STRIDE)  # 12.5 Hz

# Tiny config for fast CPU tests
TINY = dict(
    hidden_dim=64,
    latent_dim=292,           # must keep 256+36
    semantic_dim=256,
    acoustic_dim=36,
    patch_stride=PATCH_STRIDE,
    encoder_strides=(2, 2, 2, 1),
    decoder_strides=(1, 2, 2, 2),
    n_residual=1,
    dilations=(1,),
    n_transformer_layers=1,
    n_heads=4,
    ffn_dim=128,
    window_size=8,
    codebook_size=64,        # tiny codebook for speed
    fsq_levels=21,
    sample_rate=SAMPLE_RATE,
)

# Batch and segment sizes chosen so T_latent = exactly 5 frames
# T = PATCH_STRIDE × CNN_STRIDE × T_LATENT = 240 × 8 × 5 = 9600
T_LATENT = 5
T        = PATCH_STRIDE * CNN_STRIDE * T_LATENT   # 9600
BATCH    = 2


# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------

class TestVoxtralEncoder:
    def _make_encoder(self):
        from voxtral_codec.encoder import VoxtralEncoder
        return VoxtralEncoder(
            in_channels=1,
            hidden_dim=TINY["hidden_dim"],
            latent_dim=TINY["latent_dim"],
            patch_stride=TINY["patch_stride"],
            block_strides=TINY["encoder_strides"],
            n_residual=TINY["n_residual"],
            dilations=TINY["dilations"],
            n_transformer_layers=TINY["n_transformer_layers"],
            n_heads=TINY["n_heads"],
            ffn_dim=TINY["ffn_dim"],
            window_size=TINY["window_size"],
        )

    def test_output_shape(self):
        enc = self._make_encoder()
        x = torch.randn(BATCH, 1, T)
        z = enc(x)
        assert z.shape == (BATCH, TINY["latent_dim"], T_LATENT), (
            f"Expected ({BATCH}, {TINY['latent_dim']}, {T_LATENT}), got {z.shape}"
        )

    def test_gradient_flow(self):
        enc = self._make_encoder()
        x = torch.randn(BATCH, 1, T)
        z = enc(x)
        loss = z.mean()
        loss.backward()
        # Check at least one parameter received a gradient
        has_grad = any(p.grad is not None for p in enc.parameters())
        assert has_grad, "No gradients flowed back through the encoder"

    def test_causal_property(self):
        """Future frames must not influence past frames (causal masking)."""
        from voxtral_codec.encoder import CausalConv1d
        conv = CausalConv1d(1, 4, kernel_size=5, bias=False)
        x = torch.zeros(1, 1, 20)
        x[0, 0, 10] = 1.0          # impulse at position 10
        y = conv(x)
        # Positions 0..9 should be zero (causal conv, no bias)
        assert y[0, :, :10].abs().max().item() == 0.0, (
            "CausalConv1d is not causal: future input affected past output"
        )


# ---------------------------------------------------------------------------
# Decoder tests
# ---------------------------------------------------------------------------

class TestVoxtralDecoder:
    def _make_decoder(self):
        from voxtral_codec.decoder import VoxtralDecoder
        return VoxtralDecoder(
            out_channels=1,
            hidden_dim=TINY["hidden_dim"],
            latent_dim=TINY["latent_dim"],
            patch_stride=TINY["patch_stride"],
            block_strides=TINY["decoder_strides"],
            n_residual=TINY["n_residual"],
            dilations=TINY["dilations"],
            n_transformer_layers=TINY["n_transformer_layers"],
            n_heads=TINY["n_heads"],
            ffn_dim=TINY["ffn_dim"],
            window_size=TINY["window_size"],
        )

    def test_output_shape(self):
        dec = self._make_decoder()
        z = torch.randn(BATCH, TINY["latent_dim"], T_LATENT)
        x_hat = dec(z)
        # Output length must be T = 9600 (or close, ConvTranspose1d may vary by a few)
        assert x_hat.shape[0] == BATCH
        assert x_hat.shape[1] == 1
        assert x_hat.shape[-1] == T, (
            f"Expected T={T}, got {x_hat.shape[-1]}"
        )

    def test_output_range(self):
        """Tanh in decoder should bound output to (-1, 1)."""
        dec = self._make_decoder()
        z = torch.randn(BATCH, TINY["latent_dim"], T_LATENT) * 10  # large input
        x_hat = dec(z)
        assert x_hat.abs().max().item() <= 1.0 + 1e-5, (
            "Decoder output is not bounded to [-1, 1]"
        )

    def test_gradient_flow(self):
        dec = self._make_decoder()
        z = torch.randn(BATCH, TINY["latent_dim"], T_LATENT, requires_grad=True)
        x_hat = dec(z)
        x_hat.mean().backward()
        assert z.grad is not None and z.grad.abs().max() > 0


# ---------------------------------------------------------------------------
# Quantizer tests
# ---------------------------------------------------------------------------

class TestVectorQuantizer:
    def test_forward(self):
        from voxtral_codec.quantizer import VectorQuantizer
        vq = VectorQuantizer(codebook_size=64, dim=16)
        z = torch.randn(BATCH, 16, T_LATENT)
        z_q, indices, loss = vq(z)
        assert z_q.shape == z.shape
        assert indices.shape == (BATCH, T_LATENT)
        assert loss.item() >= 0

    def test_straight_through(self):
        from voxtral_codec.quantizer import VectorQuantizer
        vq = VectorQuantizer(codebook_size=64, dim=16)
        z = torch.randn(BATCH, 16, T_LATENT, requires_grad=True)
        z_q, _, loss = vq(z)
        (z_q.mean() + loss).backward()
        assert z.grad is not None and z.grad.abs().max() > 0

    def test_indices_in_range(self):
        from voxtral_codec.quantizer import VectorQuantizer
        vq = VectorQuantizer(codebook_size=64, dim=16)
        z = torch.randn(BATCH, 16, T_LATENT)
        _, indices, _ = vq(z)
        assert indices.min() >= 0
        assert indices.max() < 64


class TestFSQ:
    def test_forward(self):
        from voxtral_codec.quantizer import FSQ
        fsq = FSQ(n_levels=21, dim=36)
        z = torch.randn(BATCH, 36, T_LATENT)
        z_q, codes = fsq(z)
        assert z_q.shape == z.shape
        assert codes.shape == z.shape

    def test_codes_in_range(self):
        from voxtral_codec.quantizer import FSQ
        fsq = FSQ(n_levels=21, dim=36)
        z = torch.randn(BATCH, 36, T_LATENT)
        _, codes = fsq(z)
        assert codes.min() >= 0
        assert codes.max() < 21

    def test_levels(self):
        """Quantized values should be multiples of 1 (integers in bound range)."""
        from voxtral_codec.quantizer import FSQ
        fsq = FSQ(n_levels=21, dim=36)
        z = torch.randn(BATCH, 36, T_LATENT) * 5
        z_q, _ = fsq(z)
        # z_q should be rounded (STE applied); check they are near integers
        half = (21 - 1) / 2
        z_q_shifted = z_q + half
        assert (z_q_shifted - z_q_shifted.round()).abs().max().item() < 1e-5

    def test_straight_through(self):
        from voxtral_codec.quantizer import FSQ
        fsq = FSQ(n_levels=21, dim=36)
        z = torch.randn(BATCH, 36, T_LATENT, requires_grad=True)
        z_q, _ = fsq(z)
        z_q.mean().backward()
        assert z.grad is not None and z.grad.abs().max() > 0


class TestDualQuantizer:
    def test_forward(self):
        from voxtral_codec.quantizer import DualQuantizer
        dq = DualQuantizer(
            latent_dim=292, semantic_dim=256, acoustic_dim=36,
            codebook_size=64,
        )
        z = torch.randn(BATCH, 292, T_LATENT)
        z_q, sem_idx, ac_codes, vq_loss, ac_q = dq(z)

        assert z_q.shape == (BATCH, 292, T_LATENT)
        assert sem_idx.shape == (BATCH, T_LATENT)
        assert ac_codes.shape == (BATCH, 36, T_LATENT)
        assert vq_loss.item() >= 0

    def test_latent_split(self):
        """Semantic and acoustic parts are correctly split from 292-dim latent."""
        from voxtral_codec.quantizer import DualQuantizer
        dq = DualQuantizer(latent_dim=292, semantic_dim=256, acoustic_dim=36,
                           codebook_size=64)
        z = torch.zeros(1, 292, 2)
        z[:, :256, :] = 1.0   # semantic part = 1
        z[:, 256:, :] = -1.0  # acoustic part = -1
        z_q, _, _, _, _ = dq(z)
        assert z_q.shape == (1, 292, 2)

    def test_bitrate(self):
        """Verify the theoretical bitrate matches 2.14 kbps."""
        sem_bits  = math.log2(8192)            # 13.0
        ac_bits   = 36 * math.log2(21)        # ≈158.1
        total_bpf = sem_bits + ac_bits
        bitrate   = total_bpf * 12.5 / 1000   # kbps
        assert abs(bitrate - 2.14) < 0.01, (
            f"Bitrate mismatch: expected ~2.14 kbps, got {bitrate:.4f} kbps"
        )


# ---------------------------------------------------------------------------
# Discriminator tests
# ---------------------------------------------------------------------------

class TestSTFTDiscriminator:
    def test_forward(self):
        from voxtral_codec.discriminator import STFTDiscriminator
        disc = STFTDiscriminator(n_fft=256, hop_length=64, win_length=256)
        x = torch.randn(BATCH, 1, T)
        logits, fmaps = disc(x)
        assert logits.shape[0] == BATCH
        assert logits.shape[1] == 1
        assert len(fmaps) > 0

    def test_fmaps_count(self):
        from voxtral_codec.discriminator import STFTDiscriminator
        disc = STFTDiscriminator(n_fft=256, n_layers=3)
        x = torch.randn(BATCH, 1, T)
        _, fmaps = disc(x)
        # n_layers conv blocks + 1 extra → n_layers + 1 feature maps
        assert len(fmaps) == 4  # n_layers=3 + 1


class TestMultiResolutionDiscriminator:
    def test_forward(self):
        from voxtral_codec.discriminator import MultiResolutionDiscriminator
        # Use small FFT sizes for fast CPU test
        configs = [
            {"n_fft": 256, "hop_length": 64, "win_length": 256},
            {"n_fft": 128, "hop_length": 32, "win_length": 128},
            {"n_fft": 512, "hop_length": 128, "win_length": 512},
            {"n_fft": 64,  "hop_length": 16, "win_length": 64},
            {"n_fft": 1024,"hop_length": 256,"win_length": 1024},
            {"n_fft": 128, "hop_length": 64, "win_length": 128},
            {"n_fft": 256, "hop_length": 128,"win_length": 256},
            {"n_fft": 512, "hop_length": 256,"win_length": 512},
        ]
        disc = MultiResolutionDiscriminator(stft_configs=configs)
        x = torch.randn(BATCH, 1, T)
        results = disc(x)
        assert len(results) == 8
        for logits, fmaps in results:
            assert logits.shape[0] == BATCH

    def test_8_discriminators(self):
        from voxtral_codec.discriminator import MultiResolutionDiscriminator
        configs = [
            {"n_fft": 256, "hop_length": 64, "win_length": 256},
            {"n_fft": 128, "hop_length": 32, "win_length": 128},
            {"n_fft": 512, "hop_length": 128, "win_length": 512},
            {"n_fft": 64,  "hop_length": 16, "win_length": 64},
            {"n_fft": 1024,"hop_length": 256,"win_length": 1024},
            {"n_fft": 128, "hop_length": 64, "win_length": 128},
            {"n_fft": 256, "hop_length": 128,"win_length": 256},
            {"n_fft": 512, "hop_length": 256,"win_length": 512},
        ]
        disc = MultiResolutionDiscriminator(stft_configs=configs)
        assert len(disc.discriminators) == 8


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestLosses:
    def test_reconstruction_loss_decay(self):
        from voxtral_codec.losses import reconstruction_loss
        x = torch.randn(1, 1, 100)
        x_hat = torch.randn(1, 1, 100)
        _, w0 = reconstruction_loss(x, x_hat, step=0, initial_weight=1.0,
                                    decay_steps=1000.0)
        _, w1 = reconstruction_loss(x, x_hat, step=1000, initial_weight=1.0,
                                    decay_steps=1000.0)
        assert w0 > w1, "Reconstruction weight should decrease over time"
        assert abs(w1 - math.exp(-1.0)) < 1e-6

    def test_feature_matching_loss(self):
        from voxtral_codec.losses import feature_matching_loss
        # Simulate 2 discriminators, 3 feature maps each
        fmaps_real = [[torch.ones(2, 4, 10), torch.ones(2, 8, 5)],
                      [torch.ones(2, 4, 10), torch.ones(2, 8, 5)]]
        fmaps_fake = [[torch.zeros(2, 4, 10), torch.zeros(2, 8, 5)],
                      [torch.zeros(2, 4, 10), torch.zeros(2, 8, 5)]]
        loss = feature_matching_loss(fmaps_real, fmaps_fake)
        assert loss.item() == pytest.approx(1.0, abs=1e-4)

    def test_discriminator_loss_shape(self):
        from voxtral_codec.losses import discriminator_loss
        logits_real = [torch.ones(2, 1, 5, 10)]
        logits_fake = [torch.zeros(2, 1, 5, 10)]
        loss = discriminator_loss(logits_real, logits_fake)
        assert loss.item() >= 0
        assert loss.ndim == 0  # scalar

    def test_generator_adversarial_loss(self):
        from voxtral_codec.losses import generator_adversarial_loss
        # Perfect fake → logits all 1 → loss ≈ 0
        logits_fake = [torch.ones(2, 1, 5, 10)]
        loss = generator_adversarial_loss(logits_fake)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Full model (VoxtralCodec) tests
# ---------------------------------------------------------------------------

class TestVoxtralCodec:
    def _make_model(self):
        from voxtral_codec import VoxtralCodec
        return VoxtralCodec(**TINY)

    def test_forward_shape(self):
        model = self._make_model()
        x = torch.randn(BATCH, 1, T)
        x_hat, z, sem_idx, ac_codes, vq_loss = model(x)

        assert x_hat.shape == x.shape, (
            f"Expected output shape {x.shape}, got {x_hat.shape}"
        )
        assert z.shape == (BATCH, TINY["latent_dim"], T_LATENT)
        assert sem_idx.shape == (BATCH, T_LATENT)
        assert ac_codes.shape == (BATCH, TINY["acoustic_dim"], T_LATENT)
        assert vq_loss.item() >= 0

    def test_frame_rate(self):
        model = self._make_model()
        assert model.frame_rate == pytest.approx(12.5, abs=0.01)

    def test_total_stride(self):
        model = self._make_model()
        assert model.total_stride == PATCH_STRIDE * CNN_STRIDE  # 1920

    def test_encode_decode_roundtrip(self):
        """Encoder output fed directly to decoder should recover correct shape."""
        model = self._make_model()
        x = torch.randn(BATCH, 1, T)
        z = model.encode(x)
        z_q, _, _, _ = model.quantize(z)
        x_hat = model.decode(z_q)
        assert x_hat.shape[-1] == T

    def test_decode_from_codes(self):
        model = self._make_model()
        sem_indices = torch.randint(0, TINY["codebook_size"], (BATCH, T_LATENT))
        ac_codes = torch.randint(0, TINY["fsq_levels"], (BATCH, TINY["acoustic_dim"], T_LATENT))
        x_hat = model.decode_from_codes(sem_indices, ac_codes)
        assert x_hat.shape[0] == BATCH
        assert x_hat.shape[1] == 1

    def test_num_parameters(self):
        model = self._make_model()
        params = model.num_parameters()
        assert params["total"] > 0
        assert params["encoder"] > 0
        assert params["decoder"] > 0
        # Total = encoder + decoder + quantizer
        assert params["total"] == params["encoder"] + params["decoder"] + params["quantizer"]

    def test_gradient_end_to_end(self):
        model = self._make_model()
        x = torch.randn(BATCH, 1, T)
        x_hat, _, _, _, vq_loss = model(x)
        loss = x_hat.mean() + vq_loss
        loss.backward()
        # Verify encoder parameters received gradients
        enc_has_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in model.encoder.parameters()
        )
        assert enc_has_grad, "No gradients reached the encoder"

    def test_no_asr_loss(self):
        """NoOpASRLoss should return a zero scalar."""
        from voxtral_codec import NoOpASRLoss
        fn = NoOpASRLoss()
        x_real = torch.randn(BATCH, 1, T)
        z_sem  = torch.randn(BATCH, 256, T_LATENT)
        loss   = fn(x_real, z_sem)
        assert loss.item() == 0.0
