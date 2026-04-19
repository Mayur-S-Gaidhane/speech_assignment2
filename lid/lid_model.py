# ==========================================
# Language Identification Model (FINAL)
# ==========================================

import librosa
import numpy as np
import joblib
from collections import Counter

# Load trained model
model = joblib.load("lid/lid_model.pkl")
scaler = joblib.load("lid/scaler.pkl")

labels = ["English", "Hindi"]


# ------------------------------------------
# Feature Extraction (Frame-wise)
# ------------------------------------------
def extract_frame_features(y, sr, frame_size=2.0):

    frame_length = int(frame_size * sr)
    features = []

    for i in range(0, len(y), frame_length):

        frame = y[i:i + frame_length]

        if len(frame) < frame_length:
            continue

        # Skip silence
        if np.mean(np.abs(frame)) < 0.01:
            continue

        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)

        # Mean + Std features (26 total)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        feat = np.concatenate([mfcc_mean, mfcc_std])

        features.append(feat)

    return np.array(features)


# ------------------------------------------
# Smoothing Predictions
# ------------------------------------------
def smooth_predictions(preds, window=5):

    smoothed = []

    for i in range(len(preds)):
        start = max(0, i - window // 2)
        end = min(len(preds), i + window // 2 + 1)

        majority = Counter(preds[start:end]).most_common(1)[0][0]
        smoothed.append(majority)

    return smoothed


# ------------------------------------------
# Prediction Function
# ------------------------------------------
def predict(audio_path):

    y, sr = librosa.load(audio_path, sr=16000)

    features = extract_frame_features(y, sr, frame_size=2.0)

    # ⚠️ Handle edge case (no valid frames)
    if len(features) == 0:
        print("⚠️ No valid audio frames detected!")
        return "Unknown", []

    # Scale features
    features = scaler.transform(features)

    predictions = model.predict(features)

    # Convert to labels
    results = [labels[p] for p in predictions]

    # Apply smoothing
    results = smooth_predictions(results, window=5)

    # --------------------------------------
    # Debug Output
    # --------------------------------------
    print("\nFrame-level predictions (first 10 frames):")
    for i, r in enumerate(results[:10]):
        print(f"Frame {i}: {r}")

    # Overall language
    majority_lang = Counter(results).most_common(1)[0][0]

    print("\nOverall Detected Language:", majority_lang)

    # --------------------------------------
    # Switch Detection
    # --------------------------------------
    print("\nLanguage Switch Points:")

    for i in range(1, len(results)):
        if results[i] != results[i - 1]:
            print(f"Switch at {i*2} sec → {results[i-1]} → {results[i]}")

    # --------------------------------------
    # RETURN (IMPORTANT FIX)
    # --------------------------------------
    return majority_lang, results