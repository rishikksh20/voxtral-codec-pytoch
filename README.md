# Voxtral Codec : Combining Semantic VQ and Acoustic FSQ for Ultra-Low Bitrate Speech Generation


🎙️ **Meet Voxtral Codec:** A novel convolutional-transformer autoencoder that acts as the backbone of Voxtral TTS. It compresses raw 24 kHz audio into 12.5 Hz frames, achieving a highly efficient bitrate of just 2.14 kbps! 📉

🧩 **Token Breakdown:** Each audio frame is converted into 37 discrete tokens: 
* **1 Semantic Token** (for meaning/speech content)
* **36 Acoustic Tokens** (for sound quality/tone)
These tokens combine with text to feed the language model. 🧠

⚙️ **The Autoencoder Architecture:** * **Encoder:** Operates on "patchified" waveforms using 4 blocks of Causal CNNs + Self-Attention Transformers (with sliding windows). It downsamples the audio 8x into a 292-dimensional latent space. 
* **Decoder:** Mirrors the encoder in reverse to perfectly reconstruct the waveform! 🪞

🧮 **Dual Quantization Strategy:**
* **Semantic (256-dim):** Uses Vector Quantization (VQ) with a codebook size of 8192. 
* **Acoustic (36-dim):** Uses Finite Scalar Quantization (FSQ), mapping independently to 21 uniform levels per dimension. 📏

🗣️ **Smart Semantic Learning:** No forced aligners needed! Voxtral uses an auxiliary ASR distillation loss from a frozen **Whisper** model. By distilling from continuous hidden states instead of hard text transcripts, it captures richer phonetic and semantic details. ✨

🥊 **Adversarial Training:** Employs a multi-resolution discriminator (using 8 different STFT sizes). Instead of a standard GAN loss, it uses an L1-based feature-matching loss to guide highly discriminative and realistic audio reconstruction. 🎵

🎯 **End-to-End Training:** The ~300M parameter model is trained on a combined objective: feature-matching + ASR distillation + VQ commitment loss + an exponentially decaying reconstruction loss (which helps bootstrap early learning). 🚀

***
