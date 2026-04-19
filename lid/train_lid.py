# ==========================================
# Language Identification (LID) Training
# English vs Hindi using MFCC (Mean + Std)
# ==========================================

import os
import librosa
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# ------------------------------------------
# Feature Extraction (UPDATED - 26 features)
# ------------------------------------------
def extract_features(file_path):
    """
    Extract MFCC mean + std features (26 features total)
    """

    # Load audio
    y, sr = librosa.load(file_path, sr=16000)

    # Extract MFCC (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Mean and standard deviation
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Combine → 26 features
    features = np.concatenate([mfcc_mean, mfcc_std])

    return features


# ------------------------------------------
# Load Dataset
# ------------------------------------------
def load_data(base_path):
    """
    Load audio files and assign labels
    english → 0
    hindi → 1
    """

    X = []
    y = []

    for label, lang in enumerate(["english", "hindi"]):

        folder = os.path.join(base_path, lang)

        print(f"\nLoading {lang} data from: {folder}")

        for file in os.listdir(folder):

            # Only process WAV files
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(folder, file)

            try:
                features = extract_features(file_path)

                X.append(features)
                y.append(label)

            except Exception as e:
                print(f"Error processing {file}: {e}")

    return np.array(X), np.array(y)


# ------------------------------------------
# Train Model
# ------------------------------------------
def train():

    base_path = "data/lid_dataset"

    print("\nStep 1: Loading Dataset...")
    X, y = load_data(base_path)

    print("\nTotal Samples:", len(X))

    # --------------------------------------
    # Normalize Features
    # --------------------------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save scaler
    joblib.dump(scaler, "lid/scaler.pkl")

    # --------------------------------------
    # Train-Test Split
    # --------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    # --------------------------------------
    # Train Model
    # --------------------------------------
    print("\nStep 2: Training Model...")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # --------------------------------------
    # Evaluation
    # --------------------------------------
    print("\nStep 3: Evaluating Model...")

    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["English", "Hindi"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --------------------------------------
    # Save Model
    # --------------------------------------
    joblib.dump(model, "lid/lid_model.pkl")

    print("\nModel saved successfully at: lid/lid_model.pkl")
    print("Scaler saved at: lid/scaler.pkl")


# ------------------------------------------
# Main
# ------------------------------------------
if __name__ == "__main__":
    train()