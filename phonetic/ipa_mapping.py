# ===============================
# Convert text to IPA representation
# ===============================

def convert_to_ipa(text):
    """
    Basic Hinglish to IPA mapping
    """

    mapping = {
        "a": "ɑ",
        "e": "e",
        "i": "i",
        "o": "o",
        "u": "u",
        "th": "tʰ",
        "ph": "pʰ"
    }

    ipa_text = text.lower()

    for key, value in mapping.items():
        ipa_text = ipa_text.replace(key, value)

    return ipa_text