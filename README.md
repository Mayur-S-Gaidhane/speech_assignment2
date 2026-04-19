# Speech Understanding Programming Assignment-2

🚀 **End-to-End Speech Processing Pipeline for Code-Switched Hinglish Lectures**
👨‍💻 Author: *Mayur Shriram Gaidhane.  Roll No: M22AIE248*
Github Link : https://github.com/Mayur-S-Gaidhane/speech_assignment2

---

# 📌 Project Overview

This project implements a **complete AI speech pipeline** that:

➡️ Transcribes Hinglish (Hindi-English) lecture audio
➡️ Converts it into a **Low-Resource Language (LRL)**
➡️ Synthesizes speech using **voice cloning**
➡️ Ensures **robustness against spoofing & adversarial attacks**

---

# 🎯 Assignment Coverage

| Part       | Description           |
| ---------- | --------------------- |
| Part I     | Speech Recognition    |
| Part II    | IPA + Translation     |
| Part III   | Voice Cloning         |
| Part IV    | Security & Robustness |
| Evaluation | Metrics               |

---

# 🏗️ Project Structure

```
speech_pa2/
│
├── preprocessing/
├── lid/
├── stt/
├── translation/
├── prosody/
├── tts/
├── antispoof/
├── adversarial/
├── metrics/
│
├── data/
│   ├── original_segment.wav
│   ├── student_voice_ref.wav
│   └── output_LRL_cloned.wav
│
├── ipa_converter.py
├── pipeline.py
├── requirements.txt
└── README.md
```

---

# ⚙️ COMPLETE SETUP GUIDE (STEP-BY-STEP)

## 🔹 Step 1: Open WSL (Ubuntu)

```bash
wsl
```

---

## 🔹 Step 2: Update System

```bash
sudo apt update
sudo apt upgrade -y
```

---

## 🔹 Step 3: Install System Dependencies

```bash
sudo apt install -y ffmpeg git
```

---

## 🔹 Step 4: Install Miniconda (If not installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

👉 Restart terminal after installation

---

## 🔹 Step 5: Create Conda Environment

```bash
conda create -n speech_pa2 python=3.10 -y
conda activate speech_pa2
```

---

## 🔹 Step 7: Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔹 Step 8: (First Time Only) NLTK Setup

Run once in Python:

```python
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')
```

---

# ▶️ RUN THE PIPELINE

```bash
python pipeline.py
```

---

# 🔄 PIPELINE FLOW

---

## 🔹 Step 1: Denoising

Removes noise from audio.

---

## 🔹 Step 2: Language Identification

* Frame-wise prediction
* Detects Hindi ↔ English switching

---

## 🔹 Step 3: Speech-to-Text

* Whisper-based transcription

---

## 🔹 Step 4: IPA Conversion

* Text → phoneme representation

---

## 🔹 Step 5: Translation

* Hinglish → Low Resource Language (Santhali)

---

## 🔹 Step 6: Prosody Warping

* Extracts:

  * F0 (Pitch)
  * Energy
* Uses **DTW alignment**

---

## 🔹 Step 7: Anti-Spoof Detection

* LFCC-based classifier
* Output:

  * Bona Fide
  * Spoof

---

## 🔹 Step 8: Adversarial Attack

* FGSM-based noise
* Maintains:

```text
SNR ≈ 40 dB (inaudible)
```

---

## 🔹 Step 9: Speaker Embedding

* MFCC-based embedding extraction

---

## 🔹 Step 10: Speech Synthesis

* Generates final audio:

```
data/output_LRL_cloned.wav
```

---

## 🔹 Step 11: Evaluation Metrics

Outputs:

* WER
* MCD
* LID Accuracy
* EER
* Adversarial ε

---

# 📊 FINAL OUTPUT 

```
WER: 0.0 %
MCD: 0.19
LID Switching Accuracy: 97.73 %
EER: 8.0 %
Adversarial epsilon (ε): 0.02
```



# 🧠 DESIGN DECISIONS

* MFCC used for lightweight embedding
* LFCC approximation via spectral features
* DTW for prosody alignment
* Controlled adversarial noise
* Santhali chosen as LRL

---

# ⚠️ IMPORTANT NOTES

* Ensure audio files exist in `/data`
* Internet required for first-time model download
* FFmpeg must be installed

---

# 🧪 REPRODUCIBILITY CHECK

```bash
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
python pipeline.py
```

---

---

# ✨ FINAL NOTE

This project demonstrates:

✔ Speech Processing
✔ NLP
✔ Signal Processing
✔ AI Robustness

👉 A complete real-world AI system.

---

