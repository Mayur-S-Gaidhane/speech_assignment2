# ==========================================
# N-gram Language Model (Bigram + Log Scoring)
# ==========================================

from collections import defaultdict
import math


class NGramLM:
    def __init__(self):
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)

    # ------------------------------------------
    # Train LM
    # ------------------------------------------
    def train(self, text):
        words = text.lower().split()

        for i in range(len(words) - 1):
            self.unigram_counts[words[i]] += 1
            self.bigram_counts[words[i]][words[i + 1]] += 1

        self.unigram_counts[words[-1]] += 1

    # ------------------------------------------
    # Bigram Probability (Laplace Smoothing)
    # ------------------------------------------
    def get_probability(self, w1, w2):

        vocab_size = len(self.unigram_counts)

        count_bigram = self.bigram_counts[w1][w2]
        count_unigram = self.unigram_counts[w1]

        # Laplace smoothing
        prob = (count_bigram + 1) / (count_unigram + vocab_size)

        return prob

    # ------------------------------------------
    # Score Sentence (Log Probability)
    # ------------------------------------------
    def score(self, sentence):

        words = sentence.lower().split()
        score = 0.0

        for i in range(len(words) - 1):
            prob = self.get_probability(words[i], words[i + 1])
            score += math.log(prob)

        return score