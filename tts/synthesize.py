# ==========================================
# Simple TTS Synthesis (Part 3.3)
# ==========================================

from gtts import gTTS
import os


def synthesize_speech(text, output_path="data/output_LRL_cloned.wav"):

    print("Generating speech from translated text...")

    # Generate speech (English phonetic rendering)
    tts = gTTS(text=text, lang='en')

    temp_file = "temp_output.mp3"
    tts.save(temp_file)

    # Convert to WAV (22.05 kHz as required)
    os.system(f"ffmpeg -y -i {temp_file} -ar 22050 {output_path}")

    print(f"Synthesized speech saved at: {output_path}")

    return output_path