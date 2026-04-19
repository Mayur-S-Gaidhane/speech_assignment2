# ==========================================
# Anti-Spoof Detection (FIXED VERSION)
# ==========================================

import librosa
import numpy as np


def extract_lfcc(audio_path):

    y, sr = librosa.load(audio_path, sr=16000)

    spectrogram = np.abs(librosa.stft(y))

    lfcc = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram), n_mfcc=13)

    features = np.mean(lfcc, axis=1)

    return features


def detect_spoof(audio_path):

    features = extract_lfcc(audio_path)

    # ✅ Better scoring
    score = np.mean(np.abs(features))

    print("Spoof Score:", round(score, 4))

    # ✅ Adjusted threshold (important)
    if score > 20:
        print("Bona fide speech")
        return "Bona Fide"
    else:
        print("Spoof speech")
        return "Spoof"