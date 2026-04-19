import librosa
import soundfile as sf

def denoise_audio(input_path):

    y, sr = librosa.load(input_path, sr=16000)

    # Normalize audio
    y = y / max(abs(y))

    output_path = "data/clean_audio.wav"
    sf.write(output_path, y, sr)

    return output_path