# ==========================================
# Speaker Embedding Extraction (Task 3.1)
# ==========================================

import librosa
import numpy as np


def extract_speaker_embedding(audio_path):

    print("Extracting speaker embedding from:", audio_path)

    y, sr = librosa.load(audio_path, sr=16000)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Take mean across time → embedding vector
    embedding = np.mean(mfcc, axis=1)

    print("Speaker embedding shape:", embedding.shape)

    return embedding