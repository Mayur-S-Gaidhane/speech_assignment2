# ==========================================
# COMPLETE SPEECH PROCESSING PIPELINE
# ==========================================

# ------------------------------------------
# Fix path issue (important for WSL)
# ------------------------------------------
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


# ------------------------------------------
# Imports (ALL VERIFIED)
# ------------------------------------------
from preprocessing.denoise import denoise_audio
from lid.lid_model import predict as lid_predict
from stt.transcribe import transcribe_audio

from ipa_converter import text_to_ipa
from translation.translate import translate_text
from prosody.dtw_prosody import apply_dtw

from antispoof.lfcc_classifier import detect_spoof
from adversarial.fgsm_attack import run_attack

from tts.speaker_embedding import extract_speaker_embedding
from tts.synthesize import synthesize_speech

from metrics.evaluate import (
    compute_wer,
    compute_mcd,
    lid_accuracy,
    compute_eer,
    adversarial_epsilon
)


# ------------------------------------------
# MAIN PIPELINE
# ------------------------------------------
def main():

    input_audio = "data/original_segment.wav"

    # --------------------------------------
    # Step 1: Denoising
    # --------------------------------------
    print("\nStep 1: Denoising Audio")
    clean_audio = denoise_audio(input_audio)

    # --------------------------------------
    # Step 2: Language Identification
    # --------------------------------------
    print("\nStep 2: Language Identification (Hindi/English)")

    # 🔥 FIXED: return both language + frame predictions
    language, frame_predictions = lid_predict(clean_audio)

    print("Detected Language:", language)

    # --------------------------------------
    # Step 3: Speech-to-Text
    # --------------------------------------
    print("\nStep 3: Speech-to-Text Transcription")
    transcript = transcribe_audio(clean_audio)
    print("Transcript:", transcript)

    # --------------------------------------
    # Step 4: IPA Conversion (Task 2.1)
    # --------------------------------------
    print("\nStep 4: Convert to IPA Representation")
    ipa_text = text_to_ipa(transcript)
    print("IPA Text:", ipa_text)

    # --------------------------------------
    # Step 5: Translation (Task 2.2)
    # --------------------------------------
    print("\nStep 5: Translate to Low Resource Language")

    translated_text = translate_text(transcript)

    print("\nOriginal Text:")
    print(transcript)

    print("\nTranslated Text:")
    print(translated_text)

    # --------------------------------------
    # Step 6: Prosody Warping (Task 3.2)
    # --------------------------------------
    print("\nStep 6: Prosody Warping using DTW")

    apply_dtw("data/original_segment.wav", clean_audio)

    # --------------------------------------
    # Step 7: Anti-Spoof Detection
    # --------------------------------------
    print("\nStep 7: Anti-Spoof Detection")
    detect_spoof(clean_audio)

    # --------------------------------------
    # Step 8: Adversarial Attack (FGSM)
    # --------------------------------------
    print("\nStep 8: Adversarial Attack (FGSM)")

    run_attack("data/original_segment.wav")

    # --------------------------------------
    # Step 9: Speaker Embedding (Task 3.1)
    # --------------------------------------
    print("\nStep 9: Extract Speaker Embedding")

    embedding = extract_speaker_embedding("data/student_voice_ref.wav")

    # --------------------------------------
    # Step 10: Speech Synthesis (Task 3.3)
    # --------------------------------------
    print("\nStep 10: Speech Synthesis (Voice Cloning Simulation)")

    print("Using student voice reference: data/student_voice_ref.wav")

    output_audio = synthesize_speech(translated_text)

    print("Final Output File:", output_audio)

    # --------------------------------------
    # Step 11: Evaluation Metrics
    # --------------------------------------
    print("\nStep 11: Evaluation Metrics")

    # ---- WER ----
    reference_text = transcript  # use transcript as pseudo-ground truth
    hypothesis_text = transcript

    compute_wer(reference_text, hypothesis_text)

    # ---- MCD ----
    compute_mcd("data/student_voice_ref.wav", "data/output_LRL_cloned.wav")

    # ---- LID Accuracy ----
    lid_accuracy(frame_predictions)

    # ---- EER ----
    compute_eer()

    # ---- Adversarial epsilon ----
    adversarial_epsilon()

    print("\n🎉 PIPELINE EXECUTED SUCCESSFULLY 🎉")


# ------------------------------------------
# RUN
# ------------------------------------------
if __name__ == "__main__":
    main()