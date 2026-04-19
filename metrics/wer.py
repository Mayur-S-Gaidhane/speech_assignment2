# ===============================
# Word Error Rate (WER)
# ===============================

from jiwer import wer

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate between reference and predicted text

    WER = (Substitutions + Deletions + Insertions) / Total words
    """

    error = wer(reference, hypothesis)

    print("WER:", error)

    return error