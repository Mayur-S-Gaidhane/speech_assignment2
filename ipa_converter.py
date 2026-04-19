from g2p_en import G2p
import nltk

# Ensure required resources are available
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

g2p = G2p()

def text_to_ipa(text):

    phonemes = g2p(text)

    return " ".join(phonemes)