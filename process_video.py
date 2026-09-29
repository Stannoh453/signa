#!/usr/bin/env python3
"""
Sign Language Video Sequence Extractor
--------------------------------------
Automates dataset collection by processing directories of pre-recorded video clips.
For each video:
  1. Reads frame-by-frame.
  2. Extracts MediaPipe hands landmarks.
  3. Formulates 174-dimensional feature vectors.
  4. Resamples the video's sequence length to exactly 15 frames.
  5. Saves the sequence to 'dataset_sequences/<label>/' for LSTM training.

Usage:
  1. Create a directory named 'raw_videos/'
  2. Place folders named after classes (e.g. 'raw_videos/hello/', 'raw_videos/peace/')
  3. Put video files (.mp4, .avi, .mov) inside those folders.
  4. Run: python3 process_video.py
"""

import os
import cv2
import math
import numpy as np
import mediapipe as mp

RAW_VIDEOS_DIR = "raw_videos"
DATASET_SEQ_DIR = "dataset_sequences"
SEQUENCE_LENGTH = 15

# ==========================================
# Feature Processing Utilities
# ==========================================
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

def resample_sequence(frames, target_len=SEQUENCE_LENGTH):
    """Interpolates or slices a sequence of frame features to exactly target_len frames."""
    num_frames = len(frames)
    if num_frames == 0:
        return None
        
    arr = np.array(frames, dtype=np.float32)
    feature_dim = arr.shape[1]
    
    indices = np.linspace(0, num_frames - 1, target_len)
    resampled = np.zeros((target_len, feature_dim), dtype=np.float32)
    for i in range(feature_dim):
        resampled[:, i] = np.interp(indices, np.arange(num_frames), arr[:, i])
        
    return resampled

# ==========================================
# Main Processing Pipeline
# ==========================================
def main():
    if not os.path.exists(RAW_VIDEOS_DIR):
        os.makedirs(RAW_VIDEOS_DIR)
        print(f"\n[INFO] Created '{RAW_VIDEOS_DIR}/' folder. Please place folders of video clips inside it.")
        print(f"Example: '{RAW_VIDEOS_DIR}/hello/video1.mp4'\n")
        return

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.50
    )

    classes = [d for d in os.listdir(RAW_VIDEOS_DIR) if os.path.isdir(os.path.join(RAW_VIDEOS_DIR, d))]
    if not classes:
        print(f"\n[WARNING] No class subfolders found inside '{RAW_VIDEOS_DIR}/'.")
        print("Please create folders (e.g. 'raw_videos/hello') containing video files and run again.\n")
        return

    print("="*60)
    print(" AUTOMATED VIDEO SEQUENCE EXTRACTOR ACTIVE")
    print("="*60)
    print(f"Target sequence length: {SEQUENCE_LENGTH} frames")
    print(f"Processing subfolders: {', '.join(classes)}\n")

    for label in classes:
        class_video_dir = os.path.join(RAW_VIDEOS_DIR, label)
        output_class_dir = os.path.join(DATASET_SEQ_DIR, label)
        os.makedirs(output_class_dir, exist_ok=True)

        video_files = [f for f in os.listdir(class_video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
        if not video_files:
            print(f"  - [{label}]: No video files found in '{class_video_dir}/'")
            continue

        print(f"  - [{label}]: Found {len(video_files)} video files.")
        
        # Start count index based on existing files in sequence folder
        existing_samples = len([f for f in os.listdir(output_class_dir) if f.endswith('.npy')])
        sample_idx = existing_samples

        for video_file in video_files:
            video_path = os.path.join(class_video_dir, video_file)
            print(f"    * Processing video: '{video_file}'...")
            
            cap = cv2.VideoCapture(video_path)
            raw_frames_features = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply mirroring to stay consistent with data collection and training format
                frame = cv2.flip(frame, 1)
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                features = process_landmarks(results)
                raw_frames_features.append(features)
                
            cap.release()

            # Filter out frames where no hand was detected to avoid blanking
            valid_frames = [f for f in raw_frames_features if any(val != 0.0 for val in f)]

            if len(valid_frames) < 5:
                print(f"      [WARNING] Video '{video_file}' has too few frames with hand landmarks detected ({len(valid_frames)}). Skipping.")
                continue

            # Resample sequence to target length
            sequence_data = resample_sequence(valid_frames, SEQUENCE_LENGTH)
            if sequence_data is not None:
                out_path = os.path.join(output_class_dir, f"{sample_idx}.npy")
                np.save(out_path, sequence_data)
                print(f"      [SUCCESS] Saved sequence: '{out_path}' ({len(valid_frames)} raw frames -> 15 frames)")
                sample_idx += 1
            else:
                print(f"      [ERROR] Could not resample video '{video_file}'")

    hands.close()
    print("\n[COMPLETE] Video processing done. You can now run 'python3 train_sequence.py' to train the model.")

if __name__ == "__main__":
    main()
