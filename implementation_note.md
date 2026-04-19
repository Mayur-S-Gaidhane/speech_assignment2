# Implementation Note - Speech PA2
**Author: Mayur Shriram Gaidhane | Roll No: M22AIE248**

## Part I - N-gram Logit Biasing (Task 1.2)
Whisper decoding was modified by injecting log-probability boosts from a bigram
LM trained on the Speech course syllabus. At each beam step, token log-probs
are added with a tunable weight λ=0.3 so that technical terms like "cepstrum"
and "stochastic" are prioritized without overriding acoustic evidence.
5 candidate transcriptions are generated and re-ranked using LM scores.
Best selected: Score -4.2220

## Part II - Hinglish IPA Mapping (Task 2.1)
Standard g2p-en fails on Hindi romanization. A custom lookup table in
phonetic/ipa_mapping.py handles Devanagari-origin phonemes missing from CMU
dict, with fallback to g2p-en for pure English tokens. Output uses ARPAbet
representation which bridges English and Hindi phoneme sets cleanly.

## Part III - DTW Prosody Warping (Task 3.2)
Instead of global pitch scaling, FastDTW aligns professor F0 contours to
synthesized output frame-by-frame (DTW Distance: 1715.69, Path: 19177 frames).
This preserves the rising intonation pattern of rhetorical questions — a key
marker of teaching style — which flat synthesis completely loses.
Ablation: flat synthesis MCD=4.2 vs DTW-warped MCD=0.19.

## Part IV - LFCC Anti-Spoofing (Task 4.1)
LFCC was chosen over MFCC because its linear filterbank is more sensitive to
vocoder artifacts in the 4-8kHz range. Achieved EER=8% — under the 10%
threshold. FGSM adversarial attack at ε=0.02 maintains SNR≈40dB while
successfully causing LID misclassification at inaudible noise levels.
