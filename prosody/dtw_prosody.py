# ==========================================
# Prosody Alignment using DTW (FINAL VERSION)
# Task 3.2: F0 + Energy + DTW
# ==========================================

import librosa
import numpy as np
from fastdtw import fastdtw


def extract_prosody(audio_path):
    """
    Extract F0 (pitch) and Energy features
    """

    y, sr = librosa.load(audio_path, sr=16000)

    # ---- F0 (Pitch using YIN) ----
    f0 = librosa.yin(y, fmin=50, fmax=300)

    # ---- Energy (RMS) ----
    energy = librosa.feature.rms(y=y)[0]

    # ---- Remove NaN values in F0 ----
    valid_idx = ~np.isnan(f0)
    f0 = f0[valid_idx]

    # Match energy length with f0
    energy = energy[:len(f0)]

    # Combine features (F0 + Energy)
    features = np.vstack((f0, energy)).T

    return features


def apply_dtw(audio1, audio2):
    """
    Apply DTW on combined prosody features
    """

    print("Extracting prosody features...")

    features1 = extract_prosody(audio1)
    features2 = extract_prosody(audio2)

    # Convert to list for fastdtw
    features1 = features1.tolist()
    features2 = features2.tolist()

    print("Applying DTW...")

    distance, path = fastdtw(features1, features2)

    print("DTW Distance:", round(distance, 2))
    print("Alignment Path Length:", len(path))

    return distance