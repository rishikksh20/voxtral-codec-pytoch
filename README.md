# Voxtral Codec PyTorch

This repository contains a PyTorch implementation of the Voxtral Codec component used to convert 24 kHz mono speech waveforms into discrete codes for TTS training.

The implementation is aligned with the research notes in this repository and includes:

- A causal convolution-transformer autoencoder
- A 292-dimensional latent split into 256 semantic and 36 acoustic dimensions
- Semantic vector quantization with an 8192-entry codebook
- Acoustic finite scalar quantization with 21 levels per dimension
- A multi-resolution STFT discriminator
- Whisper-based ASR distillation support

## What The Model Does

Pointwise summary:

- Input audio is expected at 24 kHz, mono.
- The waveform is split into non-overlapping patches of 240 samples.
- Patch frames are projected into a hidden representation with a causal kernel-7 convolution.
- Four encoder stages reduce the frame rate from 100 Hz to 12.5 Hz.
- The latent is split into:
	- 256 semantic dimensions, quantized with VQ
	- 36 acoustic dimensions, quantized with FSQ
- Each 12.5 Hz frame produces 37 tokens total:
	- 1 semantic token
	- 36 acoustic tokens
- The decoder reconstructs waveform patches from the quantized latent.
- Training uses feature matching, ASR distillation, L1 reconstruction, STFT magnitude reconstruction, and VQ commitment loss.

## Architecture Summary

- Sample rate: 24,000 Hz
- Patch size: 240 samples
- Encoder patch projection: kernel size 7
- Encoder blocks: 4
- Encoder transformer layers per stage: 2 -> 2 -> 2 -> 2
- Sliding attention windows: 16 -> 8 -> 4 -> 2
- Encoder CNN strides: 2 -> 2 -> 2 -> 1
- Latent size: 292
- Semantic bottleneck: 256 dims with codebook size 8192
- Acoustic bottleneck: 36 dims with 21 FSQ levels each
- Target frame rate: $24000 / (240 \times 2 \times 2 \times 2) = 12.5$ Hz
- Approximate bitrate: $12.5 \times (\log_2 8192 + 36 \times \log_2 21) \approx 2.14$ kbps

## Repository Layout

- `voxtral_codec/encoder.py`: encoder and causal sliding-window transformer blocks
- `voxtral_codec/decoder.py`: decoder and causal upsampling blocks
- `voxtral_codec/quantizer.py`: semantic VQ and acoustic FSQ
- `voxtral_codec/discriminator.py`: multi-resolution STFT discriminator
- `voxtral_codec/asr_distillation.py`: Whisper-based semantic distillation
- `voxtral_codec/losses.py`: reconstruction, STFT, feature matching, hinge GAN losses
- `voxtral_codec/model.py`: top-level codec model
- `train.py`: real training entrypoint for audio datasets
- `dummy_train.py`: smoke test training loop with synthetic data
- `tests/test_model.py`: unit tests

## Installation

Install dependencies:

```bash
/usr/bin/python3 -m pip install -r requirements.txt
```

If you also want to run tests locally:

```bash
/usr/bin/python3 -m pip install pytest
```

## Run A Smoke Test

The quickest way to verify the model wiring is to run the dummy training loop.

```bash
/usr/bin/python3 dummy_train.py --steps 2 --device cpu
```

What this does:

- Builds a tiny Voxtral Codec configuration
- Builds the multi-resolution discriminator
- Runs generator and discriminator updates on random audio tensors
- Prints L1, STFT, feature-matching, VQ, adversarial, and discriminator losses

## Run Unit Tests

Run the test suite with:

```bash
/usr/bin/python3 -m pytest -q
```

Current validated result in this workspace:

- `33 passed`

## Train On Real Audio

Example command:

```bash
/usr/bin/python3 train.py \
	--data_dir /path/to/wav24k \
	--batch_size 8 \
	--segment_sec 4 \
	--max_steps 400000 \
	--save_dir ./checkpoints
```

Important arguments:

- `--data_dir`: directory containing `.wav` or `.flac` files
- `--batch_size`: batch size for training
- `--segment_sec`: segment length in seconds
- `--use_asr`: enable Whisper-based ASR distillation
- `--whisper_model`: Hugging Face Whisper model name
- `--save_dir`: checkpoint output directory

## Training Objective

The generator-side objective implemented here is:

$$
L = L_{feature} + L_{ASR} + \gamma_t L_{L1} + \gamma_t L_{STFT} + 0.1 L_{commit}
$$

Where:

- `L_feature`: discriminator feature matching loss
- `L_ASR`: Whisper-based semantic distillation loss
- `L_L1`: waveform reconstruction loss
- `L_STFT`: STFT magnitude reconstruction loss
- `L_commit`: VQ codebook plus commitment loss
- `\gamma_t`: decaying reconstruction weight

The discriminator uses a hinge-style objective, while the generator primarily relies on feature matching rather than a large explicit GAN generator loss.

## Quantization Behavior During Training

The implementation includes the training-time behavior described in the notes:

- Semantic VQ:
	- 50% quantized
	- 50% passed through unquantized
- Acoustic FSQ:
	- 50% quantized
	- 25% dithered with uniform noise of magnitude $1/L$
	- 25% passed through unquantized

## Notes On Implementation Scope

This codebase now covers the main codec pieces described in the notes, but a few paper details remain approximate rather than exact.

- Whisper distillation is implemented with decoder hidden states and cross-attention-derived soft alignment.
- The exact offline DTW-based head selection procedure described in the paper is not fully reproduced as a separate preprocessing stage.
- The default training code is practical for experimentation, not a claim of full paper reproduction at final-scale training.

## Verified Commands In This Workspace

The following commands were run successfully in this workspace:

```bash
/usr/bin/python3 -m pytest -q
```

Result:

- `33 passed in 8.49s`

## Reference

Original paper:

- Mistral AI, `Voxtral TTS` research paper: https://mistral.ai/static/research/voxtral-tts.pdf
