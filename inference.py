#!/usr/bin/env python3
"""
Sign Language Alphabet & Numbers Real-Time Web Inference Script (Phase 3)
Captures frames from the webcam, detects hand landmarks using MediaPipe,
normalizes features, and runs predictions using the trained PyTorch DNN.
Serves a modern HTML/CSS interface via Flask with real-time video streaming.
"""

import os
import cv2
import json
import math
import time
import threading
from collections import deque, Counter
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from flask import Flask, render_template, Response, request, jsonify

# ==========================================
# Flask Application Setup
# ==========================================
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Try to initialize pyttsx3 for optional Text-to-Speech
try:
    import pyttsx3
    has_tts = True
except Exception as e:
    has_tts = False

# ==========================================
# Configuration & Constants
# ==========================================
MODEL_PATH = "model.pth"
LABEL_MAPPING_PATH = "label_mapping.json"
MODEL_SEQ_PATH = "model_sequence.pth"
LABEL_MAPPING_SEQ_PATH = "label_mapping_sequence.json"

# Temporal smoothing history length
SMOOTHING_WINDOW = 10
# Minimum frames in the smoothing window that must agree to update prediction
CONFIRMATION_THRESHOLD = 6
# Minimum model confidence to consider a prediction valid
MIN_CONFIDENCE_THRESHOLD = 0.70
# Motion threshold for routing between static (MLP) and dynamic (LSTM/CTC) models
MOTION_THRESHOLD = 0.007
# Minimum time (seconds) between adding symbols
SYMBOL_COOLDOWN = 0.5
# Pause/idle detection for auto-spacing
WORD_COMMIT_TIMEOUT = 1.2

# Load custom classes descriptions
CUSTOM_CLASSES_PATH = "custom_classes.json"
CLASS_DESCRIPTIONS = {
    "hello": "Salute: Raise flat hand to forehead and move it outward.",
    "thank_you": "Flat hand touches lips/chin, then moves down and forward.",
    "please": "Flat hand rubs in a circle on the chest.",
    "sorry": "Fist rubs in a circle on the chest.",
    "yes": "Fist nods up and down at the wrist.",
    "no": "Index and middle fingers tap down onto the thumb.",
    "iloveyou": "Extend thumb, index, and pinky fingers (horn sign)."
}
if os.path.exists(CUSTOM_CLASSES_PATH):
    try:
        with open(CUSTOM_CLASSES_PATH, 'r') as f:
            data = json.load(f)
            for k, v in data.items():
                CLASS_DESCRIPTIONS[k] = v.get("description", "")
    except Exception as e:
        print(f"[WARNING] Error loading {CUSTOM_CLASSES_PATH}: {e}")

# Gloss translation dictionary
GLOSS_TRANSLATIONS = {
    "hello thank_you": "Hello! Thank you.",
    "hello how_are_you": "Hello, how are you?",
    "sorry please": "I'm sorry, please.",
    "yes thank_you": "Yes, thank you.",
    "no sorry": "No, I'm sorry.",
    "iloveyou thank_you": "I love you, thank you.",
    "please thank_you": "Please, thank you.",
    "hello my name stanley": "Hello, my name is Stanley.",
    "hello scouts": "Hello, Scouts!"
}

# ==========================================
# Word Prediction Engine
# ==========================================
# High-frequency English words organized by priority tiers
WORD_DICTIONARY = [
    # Tier 1: Sign language core vocabulary (highest priority)
    "hello", "goodbye", "please", "thank", "thanks", "sorry", "yes", "no",
    "help", "stop", "wait", "come", "go", "sit", "stand", "eat", "drink",
    "water", "food", "want", "need", "like", "love", "happy", "sad",
    "angry", "tired", "good", "bad", "more", "done", "again", "family",
    "father", "mother", "brother", "sister", "friend", "baby", "child",
    "man", "woman", "people", "name", "what", "where", "when", "who",
    "why", "how", "peace", "school", "home", "work", "play",
    # Tier 2: Common everyday words
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has",
    "him", "his", "how", "its", "may", "new", "now", "old", "see",
    "way", "who", "did", "got", "let", "say", "she", "too", "use",
    "about", "after", "also", "back", "been", "call", "came", "each",
    "find", "from", "give", "have", "here", "high", "into", "just",
    "know", "last", "long", "look", "made", "make", "many", "most",
    "much", "must", "name", "next", "only", "open", "over", "part",
    "said", "same", "some", "such", "take", "tell", "than", "that",
    "them", "then", "they", "this", "time", "turn", "upon", "very",
    "well", "went", "were", "will", "with", "word", "year",
    # Tier 3: Extended vocabulary
    "able", "area", "away", "best", "body", "book", "both", "city",
    "close", "cold", "color", "could", "dear", "different", "down",
    "early", "earth", "enough", "every", "example", "face", "feel",
    "first", "follow", "great", "group", "hand", "hard", "head",
    "heart", "house", "important", "keep", "kind", "large", "late",
    "learn", "left", "life", "light", "line", "list", "live", "little",
    "might", "mind", "miss", "money", "morning", "move", "never",
    "night", "number", "often", "order", "other", "paper", "place",
    "point", "power", "read", "real", "right", "room", "round",
    "run", "seem", "show", "side", "small", "start", "still",
    "story", "study", "sure", "talk", "teach", "think", "today",
    "together", "tomorrow", "under", "understand", "until", "walk",
    "watch", "white", "whole", "world", "write", "young",
    # Tier 4: Additional useful words
    "always", "animal", "answer", "beautiful", "because", "before",
    "begin", "believe", "between", "big", "black", "blue", "boy",
    "bring", "build", "buy", "car", "care", "carry", "change",
    "clean", "clear", "country", "cover", "cross", "cut", "dark",
    "daughter", "door", "draw", "dream", "drive", "during", "end",
    "enjoy", "eye", "fall", "far", "fast", "few", "fire", "five",
    "floor", "fly", "four", "free", "full", "game", "garden",
    "girl", "glass", "gold", "green", "grow", "hair", "half",
    "happen", "hear", "hour", "human", "hundred", "idea", "interest",
    "job", "join", "jump", "king", "land", "language", "laugh",
    "lead", "leave", "letter", "listen", "lose", "lunch", "map",
    "market", "matter", "meet", "minute", "month", "music",
    "natural", "news", "note", "nothing", "offer", "office",
    "once", "own", "pair", "parent", "pass", "past", "pay",
    "person", "picture", "piece", "plant", "poor", "possible",
    "practice", "problem", "program", "pull", "push", "put",
    "question", "quiet", "rain", "raise", "reach", "reason",
    "remember", "rest", "result", "return", "rich", "river",
    "road", "rock", "safe", "science", "sea", "second",
    "sell", "send", "serve", "set", "seven", "several",
    "short", "should", "sign", "simple", "since", "sing",
    "six", "sleep", "smile", "social", "son", "soon",
    "sound", "south", "speak", "special", "spend", "spring",
    "strong", "student", "summer", "sun", "table", "ten",
    "test", "third", "three", "top", "town", "tree",
    "true", "try", "two", "type", "voice", "warm",
    "week", "west", "wide", "wife", "win", "window",
    "winter", "wish", "without", "wonder", "wood", "zero",
]
# Remove duplicates while preserving priority order
_seen = set()
WORD_DICTIONARY = [w for w in WORD_DICTIONARY if not (w in _seen or _seen.add(w))]

def _levenshtein_distance(s1, s2):
    """Calculate the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def predict_words(prefix, max_results=5):
    """
    Predict complete words from a partial spelling prefix.
    Uses prefix matching with Levenshtein distance scoring for fuzzy matches.
    Returns up to max_results word suggestions.
    """
    if not prefix or len(prefix) < 1:
        return []
    
    prefix_lower = prefix.lower()
    prefix_len = len(prefix_lower)
    
    candidates = []
    
    for idx, word in enumerate(WORD_DICTIONARY):
        # Exact prefix match (highest priority)
        if word.startswith(prefix_lower):
            # Score: priority index (lower = better) + length bonus for shorter words
            score = idx * 0.01 + abs(len(word) - prefix_len) * 0.1
            candidates.append((word, score, "prefix"))
            continue
        
        # Fuzzy match: allow 1 edit for prefixes >= 3 chars
        if prefix_len >= 3:
            word_prefix = word[:prefix_len]
            edit_dist = _levenshtein_distance(prefix_lower, word_prefix)
            if edit_dist <= 1:
                score = idx * 0.01 + edit_dist * 5.0 + abs(len(word) - prefix_len) * 0.1
                candidates.append((word, score, "fuzzy"))
    
    # Sort by score (lower is better)
    candidates.sort(key=lambda x: x[1])
    
    # Return unique words
    seen = set()
    results = []
    for word, score, match_type in candidates:
        if word not in seen:
            seen.add(word)
            results.append(word)
            if len(results) >= max_results:
                break
    
    return results


# ==========================================
# Global Application State (Thread-Safe)
# ==========================================
state_lock = threading.Lock()
latest_processed_frame = None

# Inference State Variables
smoothed_prediction = "?"
smoothed_confidence = 0.0
current_spelled_word = ""
completed_words = []
active_pathway = "STATIC"
motion_energy = 0.0
min_conf_threshold = MIN_CONFIDENCE_THRESHOLD

# Loaded Models and Mappings
model = None
model_seq = None
idx_to_label = {}
idx_to_label_seq = {}
has_sequence_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Video Stream Threading Utility
# ==========================================
class VideoStream:
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.grabbed and self.frame is not None else (False, None)

    def release(self):
        self.stopped = True
        self.stream.release()

    def isOpened(self):
        return self.stream.isOpened()

cap = VideoStream(src=0, width=1280, height=720)

# ==========================================
# Model Architectures
# ==========================================
class SignLanguageCNN(nn.Module):
    def __init__(self, input_size=174, num_classes=32):
        super(SignLanguageCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5376, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=174, hidden_size=128, num_layers=2, num_classes=5):
        super(SignLanguageLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ==========================================
# Feature Processing Utilities
# ==========================================
def decode_sentence(words_list, spelling_buffer=""):
    combined_words = list(words_list)
    if spelling_buffer:
        combined_words.append(spelling_buffer)
    if not combined_words:
        return ""
    raw_sequence = " ".join(combined_words).lower()
    if raw_sequence in GLOSS_TRANSLATIONS:
        return GLOSS_TRANSLATIONS[raw_sequence]
    formatted_words = [w.replace("_", " ") for w in combined_words]
    sentence = " ".join(formatted_words)
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence[-1] in [".", "?", "!"]:
            sentence += "."
    return sentence

def extract_single_hand_features(hand_landmarks):
    coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    wrist = coords[0]
    relative_coords = [[pt[0] - wrist[0], pt[1] - wrist[1], pt[2] - wrist[2]] for pt in coords]
    palm_scale = math.sqrt(
        (coords[9][0] - coords[0][0])**2 +
        (coords[9][1] - coords[0][1])**2 +
        (coords[9][2] - coords[0][2])**2
    )
    if palm_scale > 0:
        normalized_coords = []
        for pt in relative_coords:
            normalized_coords.extend([pt[0]/palm_scale, pt[1]/palm_scale, pt[2]/palm_scale])
    else:
        normalized_coords = [coord for pt in relative_coords for coord in pt]
        palm_scale = 1.0
        
    tips = [4, 8, 12, 16, 20]
    extensions = [math.sqrt((coords[tip][0] - coords[0][0])**2 + (coords[tip][1] - coords[0][1])**2 + (coords[tip][2] - coords[0][2])**2) / palm_scale for tip in tips]
    
    tip_distances = []
    for t1, t2 in [(4, 8), (8, 12), (12, 16), (16, 20)]:
        dist = math.sqrt((coords[t1][0] - coords[t2][0])**2 + (coords[t1][1] - coords[t2][1])**2 + (coords[t1][2] - coords[t2][2])**2)
        tip_distances.append(dist / palm_scale)
        
    angles = []
    finger_joints = [
        [(0, 1, 2), (1, 2, 3), (2, 3, 4)],
        [(0, 5, 6), (5, 6, 7), (6, 7, 8)],
        [(0, 9, 10), (9, 10, 11), (10, 11, 12)],
        [(0, 13, 14), (13, 14, 15), (14, 15, 16)],
        [(0, 17, 18), (17, 18, 19), (18, 19, 20)]
    ]
    for finger in finger_joints:
        for p1, p2, p3 in finger:
            a, b, c = coords[p1], coords[p2], coords[p3]
            ba_x, ba_y, ba_z = a[0]-b[0], a[1]-b[1], a[2]-b[2]
            bc_x, bc_y, bc_z = c[0]-b[0], c[1]-b[1], c[2]-b[2]
            dot = ba_x*bc_x + ba_y*bc_y + ba_z*bc_z
            mag_ba = math.sqrt(ba_x**2 + ba_y**2 + ba_z**2)
            mag_bc = math.sqrt(bc_x**2 + bc_y**2 + bc_z**2)
            if mag_ba * mag_bc == 0:
                angles.append(0.0)
            else:
                cosine = max(min(dot / (mag_ba * mag_bc), 1.0), -1.0)
                angles.append(math.degrees(math.acos(cosine)) / 180.0)
                
    return normalized_coords + extensions + tip_distances + angles

def process_landmarks(results):
    left_hand_features = [0.0] * 87
    right_hand_features = [0.0] * 87
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = hand_handedness.classification[0].label
            features = extract_single_hand_features(hand_landmarks)
            if hand_type == "Left":
                left_hand_features = features
            elif hand_type == "Right":
                right_hand_features = features
    return left_hand_features + right_hand_features

# ==========================================
# Background Processing Loop (MediaPipe & PyTorch)
# ==========================================
def inference_loop():
    global latest_processed_frame, smoothed_prediction, smoothed_confidence
    global current_spelled_word, completed_words, active_pathway, motion_energy
    
    # Initialize MediaPipe Hands in this thread context
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,  # Lite model for CPU speed & lag-free 30 FPS
        min_detection_confidence=0.55,
        min_tracking_confidence=0.65
    )
    
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    motion_history = deque(maxlen=6)
    sequence_buffer = deque(maxlen=30)
    
    last_symbol_added_time = 0.0
    ready_for_next_symbol = True
    last_hand_seen_time = time.time()
    word_committed_on_timeout = True  # Start true to prevent immediate commit
    last_sequence_frame_time = 0.0
    last_added_symbol = None
    last_smoothed_wrist = None
    
    cap.start()
    print("[INFO] Background inference loop started.")
    
    while not cap.stopped:
        grabbed, frame = cap.read()
        if not grabbed or frame is None:
            time.sleep(0.01)
            continue
            
        # Critical horizontal mirroring (must match data collection)
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
        raw_pred_label = None
        raw_confidence = 0.0
        active_pathway_local = "STATIC"
        
        has_hands = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0
        if has_hands:
            last_hand_seen_time = time.time()
            word_committed_on_timeout = False
            
            features = process_landmarks(results)
            features_tensor = torch.tensor([features], dtype=torch.float32).to(device)
            
            current_time = time.time()
            if current_time - last_sequence_frame_time >= 0.033:
                sequence_buffer.append(features)
                last_sequence_frame_time = current_time
                
            first_hand_landmarks = results.multi_hand_landmarks[0]
            wrist = first_hand_landmarks.landmark[0]
            
            # Low-pass filter (EMA) to eliminate tracking jitter from motion metric
            if last_smoothed_wrist is None:
                smoothed_wrist = (wrist.x, wrist.y)
            else:
                alpha = 0.35  # Slightly faster EMA response
                smoothed_wrist = (
                    alpha * wrist.x + (1 - alpha) * last_smoothed_wrist[0],
                    alpha * wrist.y + (1 - alpha) * last_smoothed_wrist[1]
                )
            last_smoothed_wrist = smoothed_wrist
            motion_history.append(smoothed_wrist)
        else:
            prediction_history.append(("?", 0.0))
            motion_history.append((0.0, 0.0))
            sequence_buffer.clear()
            last_smoothed_wrist = None
            
            if not word_committed_on_timeout and (time.time() - last_hand_seen_time >= WORD_COMMIT_TIMEOUT):
                with state_lock:
                    if current_spelled_word:
                        completed_words.append(current_spelled_word)
                        current_spelled_word = ""
                        print(f"[ENGINE] Committing word due to pause. Sentence: {' '.join(completed_words)}")
                word_committed_on_timeout = True
                ready_for_next_symbol = True
                last_added_symbol = None
                
        motion_energy_local = 0.0
        if len(motion_history) >= 2:
            total_diff = 0.0
            valid_steps = 0
            for i in range(1, len(motion_history)):
                p1 = motion_history[i-1]
                p2 = motion_history[i]
                if p1 != (0.0, 0.0) and p2 != (0.0, 0.0):
                    total_diff += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    valid_steps += 1
            if valid_steps > 0:
                motion_energy_local = total_diff / valid_steps
                
        if has_hands:
            if has_sequence_model and motion_energy_local >= MOTION_THRESHOLD and len(sequence_buffer) >= 6:
                active_pathway_local = "DYNAMIC"
                # Use latest 15 frames for LSTM model, or full sequence buffer for CTC
                seq_frames = list(sequence_buffer)[-15:] if len(sequence_buffer) >= 15 else list(sequence_buffer)
                # Pad to 15 if sequence is between 6 and 14 frames
                while len(seq_frames) < 15:
                    seq_frames.insert(0, seq_frames[0])
                    
                seq_tensor = torch.tensor([seq_frames], dtype=torch.float32).to(device)
                with torch.no_grad():
                    outputs = model_seq(seq_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, dim=1)
                    raw_confidence = confidence.item()
                    raw_pred_label = idx_to_label_seq[str(predicted_idx.item())]
                
                # FALLBACK: If dynamic sequence detects neutral 'background', route back to static model immediately
                if raw_pred_label == "background":
                    active_pathway_local = "STATIC (FALLBACK)"
                    with torch.no_grad():
                        outputs = model(features_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, dim=1)
                        raw_confidence = confidence.item()
                        raw_pred_label = idx_to_label[str(predicted_idx.item())]
            else:
                active_pathway_local = "STATIC"
                with torch.no_grad():
                    outputs = model(features_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, dim=1)
                    raw_confidence = confidence.item()
                    raw_pred_label = idx_to_label[str(predicted_idx.item())]
                    
            prediction_history.append((raw_pred_label, raw_confidence))
            
        if has_hands and raw_pred_label is not None and raw_pred_label != "?":
            box_color = (0, 255, 0) if raw_confidence >= min_conf_threshold else (0, 165, 255)
            for hand_landmarks in results.multi_hand_landmarks:
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(xs) * w), int(max(xs) * w)
                y_min, y_max = int(min(ys) * h), int(max(ys) * h)
                margin = 25
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                cv2.putText(frame, f"{raw_pred_label} ({raw_confidence*100:.0f}%)", 
                            (x_min, max(y_min - 10, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                
        smoothed_prediction_local = "?"
        smoothed_confidence_local = 0.0
        
        if len(prediction_history) > 0:
            labels_in_history = [item[0] for item in prediction_history]
            counter = Counter(labels_in_history)
            most_common_label, count = counter.most_common(1)[0]
            
            if most_common_label not in {"?", "background", "other", "none"} and count >= CONFIRMATION_THRESHOLD:
                confidences = [item[1] for item in prediction_history if item[0] == most_common_label]
                avg_conf = sum(confidences) / len(confidences)
                
                if avg_conf >= min_conf_threshold:
                    smoothed_prediction_local = most_common_label
                    smoothed_confidence_local = avg_conf
                    
                    time_since_last = time.time() - last_symbol_added_time
                    is_fast_transition = (len(most_common_label) == 1) and (motion_energy_local > MOTION_THRESHOLD * 1.2)
                    
                    if not is_fast_transition and (ready_for_next_symbol or (most_common_label != last_added_symbol)) and time_since_last >= SYMBOL_COOLDOWN:
                        with state_lock:
                            if len(most_common_label) == 1:
                                current_spelled_word += most_common_label
                                print(f"[ENGINE] Spelled: '{most_common_label}' -> Buffer: '{current_spelled_word}'")
                            else:
                                if current_spelled_word:
                                    completed_words.append(current_spelled_word)
                                    current_spelled_word = ""
                                word_text = most_common_label.replace("_", " ")
                                completed_words.append(word_text)
                                print(f"[ENGINE] Word Sign: '{word_text}' -> Sentence: {' '.join(completed_words)}")
                                sequence_buffer.clear()
                                
                        last_added_symbol = most_common_label
                        last_symbol_added_time = time.time()
                        ready_for_next_symbol = False
            else:
                # If we get a background/none/other prediction with high frequency, we reset ready_for_next_symbol so the next letter can be added
                if most_common_label in {"?", "background", "other", "none"}:
                    smoothed_prediction_local = "?"
                    smoothed_confidence_local = 0.0
                    ready_for_next_symbol = True
                    last_added_symbol = None
                
        with state_lock:
            smoothed_prediction = smoothed_prediction_local
            smoothed_confidence = smoothed_confidence_local
            active_pathway = active_pathway_local
            motion_energy = motion_energy_local
            
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        with state_lock:
            latest_processed_frame = jpeg.tobytes()
            
        time.sleep(0.005)
        
    hands.close()

# ==========================================
# Flask Web Routes
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        with state_lock:
            frame_bytes = latest_processed_frame
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_state')
def stream_state():
    def event_stream():
        while True:
            with state_lock:
                suggestions = predict_words(current_spelled_word) if current_spelled_word else []
                state = {
                    "live_gesture": smoothed_prediction,
                    "confidence": smoothed_confidence,
                    "spelled": current_spelled_word,
                    "sentence": " ".join(completed_words),
                    "decoded": decode_sentence(completed_words, current_spelled_word),
                    "active_pathway": active_pathway,
                    "motion_energy": motion_energy,
                    "motion_threshold": MOTION_THRESHOLD,
                    "min_conf": min_conf_threshold,
                    "word_suggestions": suggestions
                }
            yield f"data: {json.dumps(state)}\n\n"
            time.sleep(0.1)
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/api/speak', methods=['POST'])
def api_speak():
    global completed_words, current_spelled_word
    with state_lock:
        full_speech = decode_sentence(completed_words, current_spelled_word)
    if full_speech:
        print(f"[TTS] Speaking: '{full_speech}'")
        def speak_text_bg(text_to_speak):
            import subprocess
            subprocess.Popen(['python3', '-c', f"import pyttsx3; e=pyttsx3.init(); e.setProperty('rate', 150); e.say('{text_to_speak}'); e.runAndWait()"])
        threading.Thread(target=speak_text_bg, args=(full_speech,), daemon=True).start()
        return jsonify({"status": "success", "spoken": full_speech})
    return jsonify({"status": "error", "message": "Nothing to speak"})

@app.route('/api/clear', methods=['POST'])
def api_clear():
    global completed_words, current_spelled_word
    with state_lock:
        completed_words = []
        current_spelled_word = ""
    print("[ENGINE] Cleared sentence.")
    return jsonify({"status": "success"})

@app.route('/api/backspace', methods=['POST'])
def api_backspace():
    global completed_words, current_spelled_word
    with state_lock:
        if current_spelled_word:
            current_spelled_word = current_spelled_word[:-1]
        elif completed_words:
            completed_words.pop()
    return jsonify({"status": "success"})

@app.route('/api/commit', methods=['POST'])
def api_commit():
    global completed_words, current_spelled_word
    with state_lock:
        if current_spelled_word:
            completed_words.append(current_spelled_word)
            current_spelled_word = ""
    return jsonify({"status": "success"})

@app.route('/api/config_threshold', methods=['POST'])
def api_config_threshold():
    global min_conf_threshold
    data = request.json or {}
    threshold = data.get("threshold")
    if threshold is not None:
        with state_lock:
            min_conf_threshold = max(0.30, min(0.95, float(threshold)))
        return jsonify({"status": "success", "threshold": min_conf_threshold})
    return jsonify({"status": "error", "message": "Invalid threshold"})

@app.route('/api/accept_suggestion', methods=['POST'])
def api_accept_suggestion():
    global completed_words, current_spelled_word
    data = request.json or {}
    word = data.get("word", "").strip()
    if word:
        with state_lock:
            current_spelled_word = ""
            completed_words.append(word)
        print(f"[ENGINE] Accepted word suggestion: '{word}' -> Sentence: {' '.join(completed_words)}")
        return jsonify({"status": "success", "word": word})
    return jsonify({"status": "error", "message": "No word provided"})

# ==========================================
# Main Startup
# ==========================================
def main():
    global model, model_seq, idx_to_label, idx_to_label_seq, has_sequence_model
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAPPING_PATH):
        print("\n" + "="*60)
        print("[ERROR] Missing required model files!")
        print(f"Please run 'train.py' first to generate '{MODEL_PATH}' and '{LABEL_MAPPING_PATH}'.")
        print("="*60 + "\n")
        return

    # Load mappings
    with open(LABEL_MAPPING_PATH, 'r') as f:
        idx_to_label = json.load(f)
    
    # Load PyTorch models
    model = SignLanguageCNN(input_size=174, num_classes=len(idx_to_label))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded static model onto: {device}")

    if os.path.exists(MODEL_SEQ_PATH) and os.path.exists(LABEL_MAPPING_SEQ_PATH):
        try:
            with open(LABEL_MAPPING_SEQ_PATH, 'r') as f:
                idx_to_label_seq = json.load(f)
            num_classes_seq = len(idx_to_label_seq)
            if num_classes_seq >= 2:
                model_seq = SignLanguageLSTM(input_size=174, hidden_size=128, num_layers=2, num_classes=num_classes_seq)
                model_seq.load_state_dict(torch.load(MODEL_SEQ_PATH, map_location=device))
                model_seq.to(device)
                model_seq.eval()
                has_sequence_model = True
                print(f"[INFO] Loaded sequence model onto: {device}")
        except Exception as e:
            print(f"[WARNING] Failed to load sequence model: {e}")

    # Start the background computer vision loop
    threading.Thread(target=inference_loop, daemon=True).start()
    
    # Run the web server
    print("[INFO] Starting SignFlow web server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()
