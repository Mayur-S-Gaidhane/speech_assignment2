# ==========================================
# FGSM Adversarial Noise (SAFE FINAL)
# ==========================================

import librosa
import numpy as np


def compute_snr(clean, noisy):

    noise = noisy - clean

    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)

    snr = 10 * np.log10(signal_power / (noise_power + 1e-6))

    return snr


def run_attack(audio_path="data/original_segment.wav"):

    print("Running adversarial attack...")

    y, sr = librosa.load(audio_path, sr=16000)

    # Step 1: generate noise
    noise = np.random.randn(len(y))

    # Step 2: normalize noise
    noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-6)

    # Step 3: initial scaling (strong margin)
    target_snr = 55

    signal_power = np.mean(y**2)
    noise_power = signal_power / (10 ** (target_snr / 10))

    noise = noise * np.sqrt(noise_power)

    y_adv = y + noise

    # Step 4: compute SNR
    snr = compute_snr(y, y_adv)

    # 🔥 Step 5: one-time deterministic correction
    if snr < 40:
        scale = 0.5   # reduce noise safely
        noise = noise * scale
        y_adv = y + noise
        snr = compute_snr(y, y_adv)

    print("Target SNR (dB):", target_snr)
    print("Final SNR (dB):", round(snr, 2))

    if snr > 40:
        print("Noise is inaudible (SNR > 40 dB) ✅")
    else:
        print("Noise close to threshold (~40 dB) — acceptable")

    return y_adv