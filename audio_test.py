import librosa

y, sr = librosa.load("data/original_segment.wav", sr=None)
print("Sample Rate:", sr)
print("Duration:", len(y)/sr)