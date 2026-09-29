# Signa (SignFlow) - Real-Time Sign Language Recognition & Translation

**Signa** is an end-to-end real-time Sign Language Recognition and Translation system powered by PyTorch, MediaPipe, and OpenCV. It recognizes static signs (alphabet letters A-Z and numbers 0-9), isolated dynamic words, and continuous gesture sequences with live video streaming, auto-spacing, word completion, and text-to-speech feedback.

---

## Features
- **174-D Normalized Landmark Tracking:** Extracts wrist-relative coordinates, finger extensions, neighboring tip distances, and joint angles using MediaPipe Hands across two hands.
- **Dual-Pathway Routing:** Intelligently routes static gestures to a 1D CNN and dynamic motions to a Bidirectional LSTM/GRU model.
- **Continuous Gesture CTC:** Supports continuous sequence decoding with Connectionist Temporal Classification (CTCLoss).
- **Interactive Web Interface:** Modern Flask UI streaming MJPEG video feed and Server-Sent Events (SSE).
- **Smart Word Completion & Gloss Translation:** Levenshtein distance-based word suggestions and English sentence translation.
- **Text-to-Speech (TTS):** Offline speech synthesis via `pyttsx3`.

---

## Project Structure
```
signa/
├── dataset/                         # Static frame dataset (.npy) and hand_landmarks.csv
├── dataset_sequences/               # Isolated 15-frame dynamic gesture sequences
├── dataset_sequences_continuous/    # Continuous multi-gesture sequences and labels
├── templates/                       # Flask HTML frontend (index.html)
├── static/                          # Static UI assets and images
├── training_reports/                # Accuracy plots, metrics, and confusion matrices
├── custom_classes.json              # Custom class descriptions and definitions
├── label_mapping.json               # Static class label mappings
├── label_mapping_sequence.json      # Dynamic sequence class mappings
├── label_mapping_ctc.json           # CTC continuous sequence mappings
├── model.pth                        # Pretrained Static 1D CNN model weights
├── model_sequence.pth               # Pretrained Dynamic Bi-LSTM model weights
├── model_sequence_ctc.pth           # Pretrained Continuous Bi-GRU CTC model weights
├── data_collection.py               # Interactive OpenCV data collection tool
├── augment_dataset.py               # Spatial and temporal data augmentation tool
├── clean_classes.py                 # Low-accuracy class cleanup tool
├── process_video.py                 # Automated video dataset extraction tool
├── synthesize_continuous.py         # Synthetic continuous sequence generator
├── train.py                         # Static 1D CNN training script
├── train_sequence.py                # Bi-LSTM sequence training script
├── train_ctc.py                     # Continuous CTC model training script
├── evaluate_ctc.py                  # CTC model evaluation and diagnostic tool
├── inference.py                     # Real-time Flask web inference server
└── requirements.txt                 # Project dependencies
```

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Stannoh453/signa.git
cd signa
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `mediapipe==0.10.14` is required for the legacy `mediapipe.solutions` API used in landmark extraction.

---

## Usage

### Run Real-Time Web Inference
```bash
python3 inference.py
```
Open your browser and navigate to **`http://localhost:5000`**.

### Collect Training Data
```bash
python3 data_collection.py
```

### Train Models
- **Static Model (1D CNN):**
  ```bash
  python3 train.py
  ```
- **Dynamic Sequence Model (Bi-LSTM):**
  ```bash
  python3 train_sequence.py
  ```
- **Continuous CTC Model (Bi-GRU):**
  ```bash
  python3 train_sequence.py
  ```
