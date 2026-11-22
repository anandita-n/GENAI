# SignBridge AI: A Generative AI-Powered Sign Language Recognition and Communication Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks-FF6F00.svg)](https://developers.google.com/mediapipe)
[![Tests](https://img.shields.io/badge/Tests-41%20passing-success.svg)]()

> Communication between Deaf and hearing individuals remains difficult because of language barriers and limited access to sign language interpreters. **SignBridge AI** is an intelligent, end-to-end communication assistant that integrates **computer vision, machine learning, and Generative Artificial Intelligence** to facilitate accessible, natural, and dignified two-way communication.

---

## 🌟 Key Highlights & Architecture

SignBridge AI combines three unified layers into a real-time, responsive assistive system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SignBridge AI System Pipeline                         │
└─────────────────────────────────────────────────────────────────────────────┘
  [Live Webcam / Video] ──► [MediaPipe Hand Tracking (21 3D Landmarks)]
                                     │
                                     ▼
                [Scale & Translation Invariant Normalization]
                     (63 Normalized Coords + 20 Geo Features)
                                     │
                                     ▼
                  [Machine Learning Classifier (Random Forest)]
                  (Synthetic + real landmark data, Sub-Millisecond Latency)
                                     │
                                     ▼
              [Confidence Gating (≥ 0.65) & Temporal Stabilizer]
                        (Debounce & Majority Voting)
                                     │
                                     ▼
                  [Raw Sign Token Accumulator & Editor]
                   (e.g., "N E D  D O C T E R  W H E R E")
                                     │
                                     ▼
           [Generative AI / LLM Engine (Gemini 2.5 Flash + Fallback)]
             ├── Spelling & Grammar Reconstruction
             ├── Style Transfer (Casual, Formal, Concise, Empathetic)
             └── Context-Aware Two-Way Conversational Reply Generation
                                     │
                                     ▼
              [Interactive Application & Audio Synthesis (TTS)]
             ├── Visual 21-Keypoint Skeleton Overlay (Canvas)
             ├── Web Speech API & Native Host TTS (pyttsx3 / say)
             ├── Two-Way Deaf ↔ Hearing Dialogue Thread
             └── Comprehensive Model Performance Dashboard
```

### 1. Vision & Landmark Extraction Layer
- Captures live video stream from standard commodity webcams (no specialized sensor gloves or depth cameras required).
- Extracts **21 3D hand landmarks** ($x, y, z$) using Google MediaPipe.
- Applies **translation invariance** (centering on Wrist landmark 0) and **scale invariance** (normalized by hand dimension), ensuring stability across varying camera distances and positions.
- Computes **20 additional biomechanical geometric features** (finger extension ratios, curl state indicators, and pairwise fingertip distances) yielding an 83-dimensional feature representation.

### 2. Machine Learning Gesture Classifier & Temporal Stabilizer
- Classifies 29 distinct gestures:
  - **All 26 ASL Alphabet Letters** ('A' through 'Z')
  - **Functional Gestures**: `SPACE`, `DEL` (backspace), and `CLEAR`
- **Performance**: see [Evaluation Benchmark Results](#evaluation-benchmark-results-held-out-test-set-2900-synthetic-samples) below, including an honest note on synthetic vs. real-data accuracy.
- **J and Z are motion letters.** A single frame cannot tell J from I or Z from a pointing finger, so `app/motion_detector.py` tracks the fingertip path (about 1.6 s): **J** = pinky-up handshape drawing a downward stroke plus a sideways hook, **Z** = pointing finger drawing right, diagonal down-left, right. The classifier alone never commits J or Z. These thresholds are heuristics tested on synthetic paths; tune them on your own camera.
- **Confidence Gating**: Filters out low-confidence, ambiguous hand transitions.
- **Temporal Debouncing & Majority Voting**: Requires a gesture to remain stable over a sliding window of frames before committing the character, eliminating jitter and accidental letter repetition.

### 3. Generative AI & LLM Language Processing Engine
Raw fingerspelling streams are often telegraphic, missing functional grammar, or subject to sign-similarity typos (e.g., M/N, D/1, B/4 confusion). The Generative AI component acts as the core linguistic intelligence:
- **Prompt Engineering**: Transforms raw character sequences into grammatically correct and contextually appropriate English sentences.
- **Four Distinct Communication Styles**:
  - **Casual / Everyday**: Natural, friendly phrasing for social conversations.
  - **Formal / Professional**: Polite, structured language tailored for workplaces, healthcare, classrooms, and legal settings.
  - **Concise / Direct**: Brief, unambiguous phrasing ideal for urgent alerts or quick queries.
  - **Empathetic / Warm**: Respectful, appreciative communication.
- **Context-Aware Two-Way Replies**: Generates three dynamic suggested responses for the hearing conversation partner, accelerating conversation pace.
- **Dual-Mode Execution**: Powered by Google Gemini 2.5 Flash (`google-genai`), backed by a built-in offline linguistic grammar and spell-correction engine so the system works even without an internet connection or API key.

### 3b. Vision-Language Model Image Captioning (`app/vision_engine.py`)
- The **Caption This Sign** button sends the current camera frame, plus the classifier's top-3 candidate letters, to a vision-language model (Gemini 2.5 Flash).
- The VLM returns a natural-language **caption** of the handshape (which fingers are extended or curled, thumb position), chooses the letter that actually matches the image, and explains why. This resolves look-alike signs such as M/N, U/V and K/P.
- A MediaPipe **hand check** runs first, so empty frames are never sent to the VLM (it can otherwise invent a hand).
- With no `GEMINI_API_KEY` set, or if the call fails (e.g. free-tier rate limit), an offline caption is built from the classifier's top guess.
- Set the key in a git-ignored `.env` file: `GEMINI_API_KEY=your-key`.

### 4. Interaction Layer & Speech Synthesis
- **Interactive Single-Page Application**: Sleek dark-mode interface with live MediaPipe skeleton overlay, real-time prediction badges, confidence gauges, and inline editable text buffer.
- **Audible Text-to-Speech (TTS)**: Dual-tier audio output utilizing the zero-latency browser Web Speech Synthesis API and host audio engines (`say` / `pyttsx3`).
- **Two-Way Dialogue**: Live conversation window connecting Deaf signers and hearing partners with voice-to-text microphone input.
- **Performance Evaluation Dashboard**: Full confusion matrix heatmap and per-class precision/recall/F1 breakdowns.
- **Translation History**: Persistent history log with audio replay and JSON export.

---

## 📚 Grounded Literature Foundation

SignBridge AI directly synthesizes findings and solves open limitations identified in ten recent studies (2023–2026):

1. **Papatsimouli et al. (Technologies 2023)**: Frames the necessity for accessible, affordable webcam-based sign language translation without expensive wearable sensors.
2. **Georg et al. FSboard (CVPR 2025)**: Demonstrates that landmark-based fingerspelling recognition is practical on mobile hardware, while showing character error rates that motivate a post-recognition linguistic correction layer.
3. **Alsharif et al. (Franklin Open 2024)**: Demonstrates that MediaPipe 21-point landmarks achieve ~98% accuracy and real-time CPU detection for ASL alphabet recognition, directly validating our vision and feature pipeline.
4. **Gong et al. SignLLM (CVPR 2024)**: Proves that frozen off-the-shelf Large Language Models can act as state-of-the-art sign translation engines.
5. **Sincan & Bowden Spotter+GPT (ACM IVA 2025)**: Demonstrates the recognizer-plus-LLM prompting modular pattern, which SignBridge AI builds upon.
6. **Huffman et al. (arXiv 2024 / rev. 2025)**: Human-centered study revealing that Deaf users value text AI tools that support style-aware, user-editable language generation.
7. **Fonya (arXiv 2025)**: Validates webcam-to-speech pipelines with confidence gating, while highlighting the need for complete sentence and word construction.
8. **Buttar et al. (Mathematics 2023)**: Supports real-time landmark recognition paired with visible on-screen confidence scores.
9. **Tanzer (arXiv 2024)**: Demonstrates that character-level modeling substantially improves fingerspelling comprehension, motivating passing raw token streams to LLMs with spelling repair prompts.
10. **DeVries et al. (CHI 2026)**: Shows that LLM-generated context-aware suggestions dramatically streamline assistive communication, directly inspiring our two-way quick reply generator.

---

## 📁 Project Directory Structure

```
signbridge-ai/
├── app/
│   ├── __init__.py
│   ├── config.py             # Global configuration, model paths, thresholds
│   ├── landmark_extractor.py # MediaPipe HandLandmarker wrapper & 83-dim normalizer
│   ├── asl_dataset.py        # Biomechanical 3D kinematic model & dataset synthesizer
│   ├── gesture_classifier.py # ML inference & temporal stabilization engine
│   ├── llm_engine.py         # Generative AI, prompt engineering & grammar repair
│   ├── vision_engine.py      # VLM image captioning & sign disambiguation (Gemini)
│   ├── tts_engine.py         # Text-to-speech audio synthesis engine
│   ├── main.py               # FastAPI application server & REST/WebSocket routes
│   └── static/               # Interactive Single-Page Application (SPA)
│       ├── index.html        # Modern dashboard layout & views
│       ├── css/style.css     # Responsive glassmorphism styling & animations
│       └── js/app.js         # WebRTC camera, MediaPipe Canvas, WebSpeech API & state
├── data/
│   ├── asl_landmarks_test.npz# Held-out 2,900 test samples
│   ├── eval_metrics.json     # Precomputed evaluation report & confusion matrix
│   └── tts_cache/            # Generated audio cache
├── models/
│   ├── hand_landmarker.task  # MediaPipe LiteRT model bundle
│   ├── asl_classifier.joblib # Trained Random Forest classifier
│   ├── label_encoder.joblib  # Class label encoder
│   └── scaler.joblib         # Feature normalizer
├── scripts/
│   ├── train_model.py        # Dataset generation, model training & metrics evaluation
│   ├── evaluate_model.py     # Standalone evaluation & classification report generator
│   └── record_landmarks.py   # Live webcam landmark data recorder
├── tests/
│   ├── test_landmark_extractor.py # Normalization and invariance tests
│   ├── test_gesture_classifier.py # Classifier & stabilizer tests
│   ├── test_llm_engine.py         # Prompt engineering & style transfer tests
│   └── test_api.py                # FastAPI endpoint integration tests
├── requirements.txt          # Python dependencies
├── run.sh                    # One-click startup script
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- **Operating System**: macOS, Linux, or Windows.
- **Python**: Version 3.10 or higher.
- **Webcam**: Standard built-in or USB webcam.

### 2. Installation
Clone or navigate to the repository directory:
```bash
cd /Users/vivanrajbelur/.gemini/antigravity/scratch/signbridge-ai
```

Create a virtual environment and install the required dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Application
Run the startup script:
```bash
./run.sh
```
Or start uvicorn directly:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🧪 Model Training & Benchmark Evaluation

To retrain the model and re-evaluate benchmark metrics:
```bash
python scripts/train_model.py
```

To run the standalone benchmark verification on the test split:
```bash
python scripts/evaluate_model.py
```

### Training on real data (recommended)

The classifier is trained on a synthetic kinematic dataset plus any **real** landmark recordings found in `data/`:

```bash
# Download + convert the public MIT-licensed sid220/asl-now-fingerspelling landmark dataset (~3 MB)
python scripts/import_hf_landmarks.py
# (optional) record your own: python scripts/record_landmarks.py A 60
python scripts/train_model.py
```

Real recordings are weighted 5x and 25% of them are held out to report real-world accuracy separately.

### Evaluation Benchmark Results (Held-Out Test Set: 2,900 synthetic samples)
| Metric | Score |
| :--- | :--- |
| **Synthetic test accuracy** | **99.03%** |
| **Macro F1-Score** | **99.03%** |
| **Real-data held-out accuracy** (484 real samples) | **98.14%** |
| **Inference Latency** | **~0.035 ms / frame** |

> **Read this honestly.** The synthetic test set comes from the same generator as the training data, so it measures little about real people. A model trained on synthetic data **only** scored just **35.6%** on the 1,874 real samples from the Hugging Face dataset. The real-data number above comes from a *random* split of frames from the same recordings, so near-identical neighbouring frames leak between train and test and it is optimistic too. True accuracy on new signers is unknown and likely lower; the dataset has no signer IDs to split by.

---

## 🧩 Running Automated Tests

Run the full automated test suite covering all units:
```bash
python -m unittest discover tests
```
All 41 unit and integration tests pass in a few seconds. The tests run offline (no API key or network needed).

---

## 🌐 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/status` | System health, model readiness, classes count, and active config. |
| `GET` | `/api/metrics` | Accuracy, precision, recall, F1-scores, and confusion matrix data. |
| `GET` | `/api/alphabet` | Directory of all 26 ASL alphabet signs and functional gestures. |
| `POST`| `/api/recognize/landmarks` | Classifies 21 3D landmarks and updates temporal stabilizer. |
| `POST`| `/api/recognize/frame` | Decodes base64 frame, detects hand via MediaPipe, and classifies. |
| `GET` | `/api/buffer` | Returns active character stream, current word, and sentence. |
| `POST`| `/api/buffer/space` | Appends a space to the text accumulator. |
| `POST`| `/api/buffer/backspace` | Deletes the last character or word. |
| `POST`| `/api/buffer/clear` | Clears all text buffers. |
| `POST`| `/api/buffer/set` | Manually sets or edits the full text accumulator. |
| `POST`| `/api/caption` | Vision-language image captioning: describes the hand sign in a camera frame and picks the most likely letter from the classifier's candidates (Gemini VLM, with a hand-detection gate and offline fallback). |
| `POST`| `/api/refine` | Transforms raw sign tokens into styled, grammatical English via Generative AI. |
| `POST`| `/api/chat/reply` | Generates 3 contextual conversational response suggestions. |
| `POST`| `/api/tts/speak` | Triggers audible speech synthesis. |
| `GET` | `/api/history` | Fetches translation history log. |
| `POST`| `/api/settings` | Updates Gemini API key, confidence threshold, or stability frames. |
| `WS`  | `/ws/stream` | Real-time bidirectional WebSocket stream for low-latency recognition. |

---

## 🔒 Privacy & Accessibility
- Landmark coordinates are processed locally on-device in real-time.
- No video feeds or camera streams are permanently stored on servers.
- Dual-mode Generative AI ensures full usability both online (via Gemini) and completely offline (via linguistic rule engine).
