import json
import re

def load_dictionary():
    with open("translation/dictionary.json", "r") as f:
        return json.load(f)


def translate_text(text):

    dictionary = load_dictionary()

    # Extract clean words
    words = re.findall(r"\b\w+\b", text.lower())

    translated_words = []

    for word in words:

        if word in dictionary:
            translated_words.append(dictionary[word])
        else:
            # Mark untranslated words (IMPORTANT for visibility)
            translated_words.append(f"[{word}]")

    return " ".join(translated_words)