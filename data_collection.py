#!/usr/bin/env python3
"""
Sign Language Alphabet & Numbers Data Collection Script (Phase 1)
Extracts 3D landmarks for the dominant hand, performs wrist-relative 
and scale normalization, and saves data to both .npy files and a CSV file.
"""

import os
import cv2
import csv
import math
import time
import json
import argparse
import threading
import subprocess
import numpy as np
import mediapipe as mp

# ==========================================
# Configuration & Constants
# ==========================================
DATASET_DIR = "dataset"
DATASET_SEQ_DIR = "dataset_sequences"
CSV_FILENAME = "hand_landmarks.csv"
CSV_PATH = os.path.join(DATASET_DIR, CSV_FILENAME)

# Target number of samples per class
MAX_SAMPLES = 500

# Minimum time interval between sample captures (in seconds)
# A delay of 0.033s (30 FPS) allows the user to rotate and move their hand
# to gather a more diverse, robust dataset quickly.
SAMPLE_INTERVAL = 0.033 

# Generate A-Z and 0-9 classes, excluding '7', plus new word-level gestures
ALPHABET = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
NUMBERS = [str(i) for i in range(10) if str(i) != '7']

CUSTOM_CLASSES_PATH = "custom_classes.json"
CLASS_DESCRIPTIONS = {}

def load_custom_classes():
    global CLASS_DESCRIPTIONS
    # Default words
    default_words = ["hello", "thank_you", "please", "sorry", "yes", "no", "iloveyou"]
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
            
    # Make sure default words exist in descriptions
    for word in default_words:
        if word not in CLASS_DESCRIPTIONS:
            CLASS_DESCRIPTIONS[word] = ""
            
    return list(CLASS_DESCRIPTIONS.keys())

WORDS = load_custom_classes()
CLASSES = ALPHABET + NUMBERS + WORDS

def save_custom_class(class_name, description):
    global CLASS_DESCRIPTIONS, CLASSES
    CLASS_DESCRIPTIONS[class_name] = description
    if class_name not in CLASSES:
        CLASSES.append(class_name)
    
    # Save back to custom_classes.json
    data = {}
    if os.path.exists(CUSTOM_CLASSES_PATH):
        try:
            with open(CUSTOM_CLASSES_PATH, 'r') as f:
                data = json.load(f)
        except Exception:
            pass
            
    data[class_name] = {"description": description}
    try:
        with open(CUSTOM_CLASSES_PATH, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[INFO] Saved custom class '{class_name}' to '{CUSTOM_CLASSES_PATH}'")
    except Exception as e:
        print(f"[ERROR] Failed to save custom class: {e}")

# ==========================================
# Background Training State
# ==========================================
training_status = "idle"  # 'idle', 'training', 'completed', 'failed', 'idle_success', 'idle_failed'
training_accuracy = "N/A"
training_error_msg = ""
training_thread = None

def run_training_subprocess():
    global training_status, training_accuracy, training_error_msg
    training_status = "training"
    try:
        python_exe = os.path.join("venv", "bin", "python3")
        if not os.path.exists(python_exe):
            python_exe = "python3"
            
        print(f"[TRAINING] Starting training via {python_exe} train.py...")
        result = subprocess.run([python_exe, "train.py"], capture_output=True, text=True)
        if result.returncode == 0:
            training_status = "completed"
            print("[TRAINING] Training finished successfully.")
            for line in result.stdout.splitlines():
                if "Best Validation Accuracy:" in line:
                    parts = line.split("Best Validation Accuracy:")
                    if len(parts) > 1:
                        training_accuracy = parts[1].strip()
                        print(f"[TRAINING] Parsed Accuracy: {training_accuracy}")
        else:
            training_status = "failed"
            training_error_msg = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "Subprocess returned error code"
            print(f"[TRAINING ERROR] Subprocess failed with code {result.returncode}")
            print(result.stderr)
    except Exception as e:
        training_status = "failed"
        training_error_msg = str(e)
        print(f"[TRAINING ERROR] Exception during training: {e}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,             # Track up to 2 hands
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# ==========================================
# Utility Functions
# ==========================================
def setup_directories():
    """Create the dataset base directory and subdirectories for each class."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(DATASET_SEQ_DIR, exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(DATASET_SEQ_DIR, cls), exist_ok=True)
    
    # Initialize the CSV file with headers if it doesn't exist
    if not os.path.exists(CSV_PATH):
        headers = ["class"]
        # Left Hand features
        for i in range(21):
            headers.extend([f"left_lm{i}_x", f"left_lm{i}_y", f"left_lm{i}_z"])
        headers.extend([f"left_ext_{i}" for i in range(5)])
        headers.extend([f"left_tip_dist_{i}" for i in range(4)])
        headers.extend([f"left_angle_{i}" for i in range(15)])
        
        # Right Hand features
        for i in range(21):
            headers.extend([f"right_lm{i}_x", f"right_lm{i}_y", f"right_lm{i}_z"])
        headers.extend([f"right_ext_{i}" for i in range(5)])
        headers.extend([f"right_tip_dist_{i}" for i in range(4)])
        headers.extend([f"right_angle_{i}" for i in range(15)])
        
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def get_existing_sample_count(class_label):
    """Returns the number of already saved .npy files for a given class."""
    class_dir = os.path.join(DATASET_DIR, class_label)
    if not os.path.exists(class_dir):
        return 0
    return len([f for f in os.listdir(class_dir) if f.endswith('.npy')])

def get_existing_sequence_count(class_label):
    """Returns the number of already saved sequence .npy files for a given class."""
    class_dir = os.path.join(DATASET_SEQ_DIR, class_label)
    if not os.path.exists(class_dir):
        return 0
    return len([f for f in os.listdir(class_dir) if f.endswith('.npy')])

def reset_class_data(class_label, mode="frame"):
    """Deletes all .npy files for the class and removes corresponding rows from CSV if in frame mode."""
    if mode == "frame":
        # Delete .npy files
        class_dir = os.path.join(DATASET_DIR, class_label)
        if os.path.exists(class_dir):
            for f in os.listdir(class_dir):
                if f.endswith('.npy'):
                    try:
                        os.remove(os.path.join(class_dir, f))
                    except OSError as e:
                        print(f"Error removing file {f}: {e}")

        # Remove class rows from central CSV
        if os.path.exists(CSV_PATH):
            temp_rows = []
            try:
                with open(CSV_PATH, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        temp_rows.append(header)
                    for row in reader:
                        if row and row[0] != class_label:
                            temp_rows.append(row)
                
                with open(CSV_PATH, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(temp_rows)
            except Exception as e:
                print(f"Error updating CSV: {e}")
    else:
        # Delete sequence files
        class_dir = os.path.join(DATASET_SEQ_DIR, class_label)
        if os.path.exists(class_dir):
            for f in os.listdir(class_dir):
                if f.endswith('.npy'):
                    try:
                        os.remove(os.path.join(class_dir, f))
                    except OSError as e:
                        print(f"Error removing sequence file {f}: {e}")

def save_sample(class_label, feature_vector, sample_idx):
    """Saves the feature vector as an .npy file and appends it to the central CSV."""
    # 1. Save as NumPy file
    npy_path = os.path.join(DATASET_DIR, class_label, f"{sample_idx}.npy")
    np.save(npy_path, np.array(feature_vector, dtype=np.float32))

    # 2. Append to CSV
    row = [class_label] + feature_vector
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def extract_single_hand_features(hand_landmarks):
    # Extract coordinates
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
    
    # 1. Translation normalization (relative to wrist)
    wrist = coords[0]
    relative_coords = []
    for pt in coords:
        relative_coords.append([pt[0] - wrist[0], pt[1] - wrist[1], pt[2] - wrist[2]])
    
    # 2. Scale normalization using a rigid palm bone (wrist to middle finger MCP)
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
        palm_scale = 1.0  # Set fallback to avoid zero division in geometric features
        
    # 3. Geometric Features: Extensions (5 features)
    tips = [4, 8, 12, 16, 20]
    extensions = []
    for tip in tips:
        dist = math.sqrt(
            (coords[tip][0] - coords[0][0])**2 +
            (coords[tip][1] - coords[0][1])**2 +
            (coords[tip][2] - coords[0][2])**2
        )
        extensions.append(dist / palm_scale)
        
    # 4. Geometric Features: Neighboring tip-to-tip distances (4 features)
    tip_distances = []
    for t1, t2 in [(4, 8), (8, 12), (12, 16), (16, 20)]:
        dist = math.sqrt(
            (coords[t1][0] - coords[t2][0])**2 +
            (coords[t1][1] - coords[t2][1])**2 +
            (coords[t1][2] - coords[t2][2])**2
        )
        tip_distances.append(dist / palm_scale)
        
    # 5. Geometric Features: Joint Angles (15 features)
    angles = []
    finger_joints = [
        [(0, 1, 2), (1, 2, 3), (2, 3, 4)],       # Thumb
        [(0, 5, 6), (5, 6, 7), (6, 7, 8)],       # Index
        [(0, 9, 10), (9, 10, 11), (10, 11, 12)], # Middle
        [(0, 13, 14), (13, 14, 15), (14, 15, 16)], # Ring
        [(0, 17, 18), (17, 18, 19), (18, 19, 20)]  # Pinky
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
                # Normalize angle to [0, 1] for neural network
                angles.append(math.degrees(math.acos(cosine)) / 180.0)
                
    return normalized_coords + extensions + tip_distances + angles

def process_landmarks(results):
    """
    Processes hand landmarks from MediaPipe Hands results and returns a 174-dimensional
    feature vector containing:
    - Left hand block (87 features): 63 coordinates, 5 extensions, 4 tip-to-tip distances, 15 joint angles.
    - Right hand block (87 features): 63 coordinates, 5 extensions, 4 tip-to-tip distances, 15 joint angles.
    If a hand is missing, its block is filled with zeros.
    """
    left_hand_features = [0.0] * 87
    right_hand_features = [0.0] * 87
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = hand_handedness.classification[0].label # 'Left' or 'Right'
            features = extract_single_hand_features(hand_landmarks)
            if hand_type == "Left":
                left_hand_features = features
            elif hand_type == "Right":
                right_hand_features = features
                
    return left_hand_features + right_hand_features

# ==========================================
# Threaded Video Stream (supports USB cam & IP phone cameras)
# ==========================================
class VideoStream:
    """
    Captures frames in a background thread to prevent blocking the main loop.
    Supports both local webcam (src=0) and network streams (e.g. IP Webcam app URL).
    """
    def __init__(self, src):
        # Accept int for webcam index, or string for network URL
        self.stream = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        if str(src).isdigit():
            # Force MJPG codec to achieve high FPS on compatible cameras
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.grabbed and self.frame is not None else (False, None)

    def isOpened(self):
        return self.stream.isOpened()

    def release(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# Main Execution
# ==========================================
def main():
    global CLASSES, CLASS_DESCRIPTIONS, training_status, training_accuracy, training_thread

    # ------------------------------------------------------------------
    # Argument Parsing: allows switching between webcam and phone camera
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Sign Language Data Collector")
    parser.add_argument(
        "--source", default="0",
        help=(
            "Video source. Use '0' for default webcam, or a network stream URL "
            "for your phone camera.\n"
            "  Android (IP Webcam app): http://192.168.x.x:8080/video\n"
            "  DroidCam:                http://192.168.x.x:4747/video\n"
            "  iPhone (EpocCam/Camo):  Use the app's stream URL"
        )
    )
    args = parser.parse_args()

    setup_directories()

    # Initialize threaded video stream
    print(f"[INFO] Opening video source: '{args.source}'...")
    cap = VideoStream(src=args.source).start()
    if not cap.isOpened():
        print("[ERROR] Could not open video source. Check your --source argument.")
        print("  For webcam : python data_collection.py --source 0")
        print("  For phone  : python data_collection.py --source http://192.168.x.x:8080/video")
        return
    # Give the stream a moment to warm up
    time.sleep(1.0)

    # State variables
    class_idx = 0
    recording = False
    last_sample_time = 0
    
    # Mode variables
    collection_mode = "frame"
    sequence_recording = False
    sequence_buffer = []
    last_sequence_save_time = 0
    last_frame_captured_time = 0.0
    
    # FPS variables
    prev_time = 0
    fps = 0

    print("\n" + "="*50)
    print(" SIGN LANGUAGE LANDMARK DATA COLLECTION INITIALIZED")
    print("="*50)
    print("Controls:")
    print("  [S] : Start / Pause Recording")
    print("  [R] : Reset current class samples")
    print("  [N] : Next Class")
    print("  [P] : Previous Class")
    print("  [J] : Jump to specific Class")
    print("  [A] : Add New Custom Class/Word")
    print("  [T] : Train Model on current dataset")
    print("  [Q] : Quit Script")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        # Flip horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_class = CLASSES[class_idx]
        if collection_mode == "frame":
            sample_count = get_existing_sample_count(current_class)
            target_samples = MAX_SAMPLES
        else:
            sample_count = get_existing_sequence_count(current_class)
            target_samples = 30  # Only 30 required for words (augment_dataset.py will boost this)
            
        class_completed = sample_count >= target_samples

        # Auto-pause recording if maximum samples reached
        if class_completed and (recording or sequence_recording):
            recording = False
            sequence_recording = False

        has_hands = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0
        detected_hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        is_two_handed = current_class in {"6", "8", "9"}

        # Draw hand landmarks for all visible hands
        if has_hands:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Handle active recording
        if has_hands and not class_completed:
            if collection_mode == "frame" and recording:
                current_time = time.time()
                if current_time - last_sample_time >= SAMPLE_INTERVAL:
                    # Extract and normalize 144-dimensional features
                    features = process_landmarks(results)
                    # Save the sample
                    save_sample(current_class, features, sample_count)
                    last_sample_time = current_time
            elif collection_mode == "sequence" and sequence_recording:
                current_time = time.time()
                # Enforce a 1-second cooldown after a sequence is saved so user can reset hand
                if current_time - last_sequence_save_time > 1.0:
                    if current_time - last_frame_captured_time >= 0.08:
                        # Extract and normalize 144-dimensional features
                        features = process_landmarks(results)
                        sequence_buffer.append(features)
                        last_frame_captured_time = current_time
                        
                        if len(sequence_buffer) >= 15:
                            # Save the 15-frame sequence as .npy file
                            seq_dir = os.path.join(DATASET_SEQ_DIR, current_class)
                            npy_path = os.path.join(seq_dir, f"{sample_count}.npy")
                            np.save(npy_path, np.array(sequence_buffer, dtype=np.float32))
                            print(f"[INFO] Saved sequence sample {sample_count} for class '{current_class}'")
                            sequence_buffer = []
                            last_sequence_save_time = current_time

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Check and handle training status transitions
        if training_status == "completed":
            training_success_show_time = time.time()
            training_status = "idle_success"
        elif training_status == "failed":
            training_success_show_time = time.time()
            training_status = "idle_failed"

        # Force pause recording if training is running
        if training_status == "training":
            recording = False
            sequence_recording = False
            sequence_buffer = []

        # ----------------------------------------------------
        # Sleek HUD / UI Drawing
        # ----------------------------------------------------
        # Determine top bar height based on whether instructions are present
        instruction = CLASS_DESCRIPTIONS.get(current_class, "")
        top_bar_height = 105 if instruction else 80
        
        # Top bar overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, top_bar_height), (15, 15, 15), -1)
        # Bottom bar overlay
        cv2.rectangle(overlay, (0, h - 50), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # UI Text details
        cv2.putText(frame, f"CLASS: {current_class}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display progress
        prog_color = (0, 255, 0) if class_completed else (255, 255, 255)
        cv2.putText(frame, f"SAMPLES: {sample_count}/{target_samples}", (340, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, prog_color, 2, cv2.LINE_AA)

        # Display mode
        cv2.putText(frame, f"MODE: {collection_mode.upper()}", (580, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Display recording state
        if training_status == "training":
            status_text = "STATUS: TRAINING MODEL..."
            status_color = (0, 165, 255)  # Orange
        elif collection_mode == "sequence" and sequence_recording:
            if time.time() - last_sequence_save_time <= 1.0:
                status_text = "STATUS: COOLDOWN..."
                status_color = (0, 255, 255)  # Yellow
            else:
                status_text = f"REC SEQ: {len(sequence_buffer)}/15"
                status_color = (0, 0, 255)  # Red
        elif recording:
            status_text = "STATUS: RECORDING"
            status_color = (0, 0, 255)  # Red
        else:
            status_text = "STATUS: PAUSED" if not class_completed else "STATUS: COMPLETE"
            status_color = (0, 255, 0) if class_completed else (0, 255, 255)  # Green or Yellow
        cv2.putText(frame, status_text, (780, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw physical movement instruction if available
        if instruction:
            cv2.putText(frame, f"HOW-TO: {instruction}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Display instructions in bottom bar
        instructions = "[S] Record  |  [R] Reset  |  [P] Prev  |  [N] Next  |  [J] Jump  |  [M] Mode  |  [A] Add Word  |  [T] Train  |  [Q] Quit"
        cv2.putText(frame, instructions, (30, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Overlay training popup if currently training
        if training_status == "training":
            cv2.rectangle(frame, (w // 2 - 250, h // 2 - 40), (w // 2 + 250, h // 2 + 40), (0, 0, 0), -1)
            cv2.rectangle(frame, (w // 2 - 250, h // 2 - 40), (w // 2 + 250, h // 2 + 40), (0, 165, 255), 2)
            cv2.putText(frame, "TRAINING NEURAL NETWORK...", (w // 2 - 220, h // 2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Please wait, do not close the window.", (w // 2 - 210, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Overlay training completion popup for 4 seconds
        if 'training_success_show_time' in locals() and time.time() - training_success_show_time < 4.0:
            if training_status == "idle_success":
                cv2.rectangle(frame, (w // 2 - 280, h // 2 - 40), (w // 2 + 280, h // 2 + 40), (0, 60, 0), -1)
                cv2.rectangle(frame, (w // 2 - 280, h // 2 - 40), (w // 2 + 280, h // 2 + 40), (0, 255, 0), 2)
                cv2.putText(frame, "RETRAINING COMPLETE!", (w // 2 - 220, h // 2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Validation Accuracy: {training_accuracy}", (w // 2 - 160, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
            elif training_status == "idle_failed":
                cv2.rectangle(frame, (w // 2 - 280, h // 2 - 40), (w // 2 + 280, h // 2 + 40), (0, 0, 60), -1)
                cv2.rectangle(frame, (w // 2 - 280, h // 2 - 40), (w // 2 + 280, h // 2 + 40), (0, 0, 255), 2)
                cv2.putText(frame, "RETRAINING FAILED!", (w // 2 - 220, h // 2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Err: {training_error_msg[:45]}", (w // 2 - 250, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1, cv2.LINE_AA)

        # Big alert box if collection is complete
        if class_completed:
            # Draw overlay banner
            cv2.rectangle(frame, (w // 2 - 300, h // 2 - 60), (w // 2 + 300, h // 2 + 60), (0, 100, 0), -1)
            cv2.rectangle(frame, (w // 2 - 300, h // 2 - 60), (w // 2 + 300, h // 2 + 60), (0, 255, 0), 2)
            cv2.putText(frame, f"CLASS '{current_class}' COMPLETED!", (w // 2 - 250, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "Press [N] for next, or [R] to re-record", (w // 2 - 275, h // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2, cv2.LINE_AA)

        # Warnings and cues based on expected vs detected hand counts
        if recording or sequence_recording:
            if not has_hands:
                cv2.rectangle(frame, (w // 2 - 250, h // 2 - 30), (w // 2 + 250, h // 2 + 30), (0, 0, 100), -1)
                cv2.rectangle(frame, (w // 2 - 250, h // 2 - 30), (w // 2 + 250, h // 2 + 30), (0, 0, 255), 2)
                cv2.putText(frame, "WARNING: NO HAND DETECTED", (w // 2 - 220, h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            elif is_two_handed and detected_hand_count < 2:
                cv2.rectangle(frame, (w // 2 - 250, h // 2 - 30), (w // 2 + 250, h // 2 + 30), (0, 0, 100), -1)
                cv2.rectangle(frame, (w // 2 - 250, h // 2 - 30), (w // 2 + 250, h // 2 + 30), (0, 0, 255), 2)
                cv2.putText(frame, "WARNING: EXPECTED TWO HANDS!", (w // 2 - 230, h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            elif not is_two_handed and detected_hand_count > 1:
                cv2.rectangle(frame, (w // 2 - 250, h // 2 - 30), (w // 2 + 250, h // 2 + 30), (0, 0, 100), -1)
                cv2.rectangle(frame, (w // 2 - 250, h // 2 - 30), (w // 2 + 250, h // 2 + 30), (0, 0, 255), 2)
                cv2.putText(frame, "WARNING: EXPECTED ONE HAND!", (w // 2 - 230, h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Display window
        cv2.imshow("Sign Language Data Collector (Phase 1)", frame)

        # Keyboard event handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            if training_status == "training":
                print("[WARNING] Cannot record while training.")
            elif class_completed:
                print(f"[WARNING] Class '{current_class}' is already completed. Move to the next class.")
            elif collection_mode == "frame":
                recording = not recording
                print(f"[INFO] Recording state: {'STARTED' if recording else 'PAUSED'}")
            elif collection_mode == "sequence":
                sequence_recording = not sequence_recording
                if sequence_recording:
                    sequence_buffer = []
                    last_sequence_save_time = 0
                    print(f"[INFO] Continuous sequence recording loop STARTED. Perform the gesture repeatedly!")
                else:
                    sequence_buffer = []
                    print(f"[INFO] Sequence recording loop PAUSED.")
        elif key == ord('n') or key == ord('N'):
            class_idx = (class_idx + 1) % len(CLASSES)
            recording = False
            sequence_recording = False
            sequence_buffer = []
            print(f"[INFO] Switched to Class: {CLASSES[class_idx]}")
        elif key == ord('p') or key == ord('P'):
            class_idx = (class_idx - 1) % len(CLASSES)
            recording = False
            sequence_recording = False
            sequence_buffer = []
            print(f"[INFO] Switched to Class: {CLASSES[class_idx]}")
        elif key == ord('r') or key == ord('R'):
            recording = False
            sequence_recording = False
            sequence_buffer = []
            print(f"[WARNING] Resetting data for Class '{current_class}' ({collection_mode.upper()} mode)...")
            reset_class_data(current_class, mode=collection_mode)
            print(f"[INFO] Reset completed. Ready to re-record Class '{current_class}'.")
        elif key == ord('j') or key == ord('J'):
            recording = False
            sequence_recording = False
            sequence_buffer = []
            print("\n" + "="*50)
            print("[JUMP CLASS MODE]")
            print(f"Current class: {current_class}")
            target = input("Enter class label (A-Z or 0-9 or custom word): ").strip()
            # Check upper case for alphabet/numbers, keep original case for custom words
            target_upper = target.upper()
            if target_upper in CLASSES:
                class_idx = CLASSES.index(target_upper)
                print(f"[INFO] Switched to Class: {target_upper}")
            elif target in CLASSES:
                class_idx = CLASSES.index(target)
                print(f"[INFO] Switched to Class: {target}")
            else:
                print(f"[ERROR] Class '{target}' not found.")
            print("="*50 + "\n")
        elif key == ord('a') or key == ord('A'):
            recording = False
            sequence_recording = False
            sequence_buffer = []
            print("\n" + "="*50)
            print("[ADD NEW CUSTOM GESTURE/WORD]")
            # Prompt user in console
            name = input("Enter new gesture/word name (e.g., thank_you): ").strip().lower().replace(" ", "_")
            if not name:
                print("[ERROR] Word name cannot be empty.")
            elif name in CLASSES:
                print(f"[ERROR] Class '{name}' already exists.")
            else:
                description = input("Enter description of how to perform it: ").strip()
                save_custom_class(name, description)
                os.makedirs(os.path.join(DATASET_DIR, name), exist_ok=True)
                os.makedirs(os.path.join(DATASET_SEQ_DIR, name), exist_ok=True)
                # Select the new class
                class_idx = CLASSES.index(name)
                print(f"[SUCCESS] Switched to new Class: {name}")
            print("="*50 + "\n")
        elif key == ord('m') or key == ord('M'):
            if recording or sequence_recording:
                print("[WARNING] Cannot change mode while recording is active.")
            else:
                collection_mode = "sequence" if collection_mode == "frame" else "frame"
                print(f"[INFO] Toggled collection mode to: {collection_mode.upper()}")
        elif key == ord('t') or key == ord('T'):
            if training_status == "training":
                print("[WARNING] Training is already in progress.")
            else:
                recording = False
                sequence_recording = False
                sequence_buffer = []
                training_thread = threading.Thread(target=run_training_subprocess)
                training_thread.start()

    # Clean cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n" + "="*50)
    print(" DATA COLLECTION SESSION ENDED")
    print("="*50)

if __name__ == "__main__":
    main()
