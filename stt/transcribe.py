# ==========================================
# Speech-to-Text with Constrained Decoding
# Whisper + N-gram LM (Syllabus-Based)
# ==========================================

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from stt.ngram_lm import NGramLM


# ------------------------------------------
# Load Whisper Model
# ------------------------------------------
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


# ------------------------------------------
# Train Language Model (SYLLABUS-BASED)
# ------------------------------------------
lm = NGramLM()

training_text = """
speech signal processing lecture
mel frequency cepstral coefficients are used in speech recognition
cepstrum analysis is important in speech processing
stochastic models are used for acoustic modeling
hidden markov models are used in speech recognition
spectral analysis of speech signals
fourier transform converts signal to frequency domain
filter bank analysis in speech processing
neural networks are used in modern speech recognition systems
deep learning improves speech recognition performance
this paper which improves performance
the text encoder processes the input sequence
the projection layer transforms the features
the architecture consists of multiple components
"""

lm.train(training_text)


# ------------------------------------------
# Generate Diverse Candidates
# ------------------------------------------
def generate_candidates(inputs):

    outputs = model.generate(
        inputs,
        num_beams=5,
        num_return_sequences=5,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        max_length=200
    )

    transcripts = processor.batch_decode(outputs, skip_special_tokens=True)

    return list(set(transcripts))


# ------------------------------------------
# Select Best Candidate using LM
# ------------------------------------------
def select_best(transcripts):

    best_score = float("-inf")
    best_text = ""

    print("\nLM Scoring:")

    for t in transcripts:

        raw_score = lm.score(t)

        # Normalize by length
        length = max(len(t.split()), 1)
        score = raw_score / length

        print(f"Score: {score:.4f} → {t}")

        if score > best_score:
            best_score = score
            best_text = t

    return best_text


# ------------------------------------------
# Main Function
# ------------------------------------------
def transcribe_audio(audio_path):

    # Load audio
    speech, sr = librosa.load(audio_path, sr=16000)

    inputs = processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    # --------------------------------------
    # Generate candidates
    # --------------------------------------
    candidates = generate_candidates(inputs)

    print("\nCandidate Transcriptions:")
    for i, c in enumerate(candidates):
        print(f"{i+1}: {c}")

    # --------------------------------------
    # Select best using LM
    # --------------------------------------
    best_transcript = select_best(candidates)

    print("\nSelected (LM Optimized):", best_transcript)

    return best_transcript