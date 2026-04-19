# ===============================
# Mel Cepstral Distortion (MCD)
# ===============================

import librosa
import numpy as np

def calculate_mcd(ref_audio, synth_audio):
    """
    Calculate MCD between reference and synthesized audio
    """

    y1, sr = librosa.load(ref_audio, sr=22050)
    y2, sr = librosa.load(synth_audio, sr=22050)

    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13)

    min_len = min(mfcc1.shape[1], mfcc2.shape[1])

    mfcc1 = mfcc1[:, :min_len]
    mfcc2 = mfcc2[:, :min_len]

    diff = mfcc1 - mfcc2

    mcd = np.mean(np.sqrt(np.sum(diff**2, axis=0)))

    print("MCD:", mcd)

    return mcd