#!/usr/bin/env python3
"""
Dataset Augmentation and Sequence Automation Tool
- Automatically builds 60-frame static sequences from the static CSV database.
- Synthetically augments dynamic gestures from a few seed recordings (using spatial/temporal warping).
"""

import os
import csv
import random
import numpy as np

CSV_PATH = "dataset/hand_landmarks.csv"
DATASET_SEQ_DIR = "dataset_sequences"
TARGET_SAMPLES = 50
SEQUENCE_LENGTH = 15
FEATURE_SIZE = 174

def temporal_warp(seq, warp_factor=1.0):
    """Speeds up or slows down a sequence using linear interpolation."""
    num_frames = seq.shape[0]
    new_len = max(5, int(num_frames * warp_factor))
    indices = np.linspace(0, num_frames - 1, new_len)
    
    warped_seq = np.zeros((new_len, seq.shape[1]), dtype=np.float32)
    for i in range(seq.shape[1]):
        warped_seq[:, i] = np.interp(indices, np.arange(num_frames), seq[:, i])
        
    # Resample back to exactly target sequence length
    final_indices = np.linspace(0, new_len - 1, SEQUENCE_LENGTH)
    final_seq = np.zeros((SEQUENCE_LENGTH, seq.shape[1]), dtype=np.float32)
    for i in range(seq.shape[1]):
        final_seq[:, i] = np.interp(final_indices, np.arange(new_len), warped_seq[:, i])
        
    return final_seq

def main():
    print("="*60)
    print(" SIGN LANGUAGE DATASET AUTOMATION & AUGMENTATION")
    print("="*60)

    # Ensure output directory exists
    os.makedirs(DATASET_SEQ_DIR, exist_ok=True)

    # 1. Load static features from CSV
    static_data = {}
    if os.path.exists(CSV_PATH):
        print(f"[INFO] Reading static classes from '{CSV_PATH}'...")
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header
            for row in reader:
                if not row:
                    continue
                label = row[0]
                features = np.array([float(val) for val in row[1:]], dtype=np.float32)
                if label not in static_data:
                    static_data[label] = []
                static_data[label].append(features)
        print(f"[INFO] Found {len(static_data)} static classes in CSV database.")
    else:
        print(f"[WARNING] Static CSV database '{CSV_PATH}' not found. Skipping static sequence synthesis.")

    # Clean up old synthesized static class folders in dataset_sequences to avoid polluting the LSTM
    print("\n[CLEANUP] Cleaning up old individual static sequence folders...")
    for label in static_data.keys():
        cls_dir = os.path.join(DATASET_SEQ_DIR, label)
        if os.path.exists(cls_dir):
            import shutil
            shutil.rmtree(cls_dir)
            print(f"  - Removed old static sequence directory: {label}")

    # 2. Synthesize Background Class (representing static holds and transitions)
    print("\n[STEP 1] Synthesizing 'background' sequences from static database...")
    bg_dir = os.path.join(DATASET_SEQ_DIR, "background")
    os.makedirs(bg_dir, exist_ok=True)
    
    # We want to generate 150 background sequences
    NUM_BG_SAMPLES = 150
    existing_bg_files = [f for f in os.listdir(bg_dir) if f.endswith('.npy')]
    current_bg_count = len(existing_bg_files)
    
    if current_bg_count < NUM_BG_SAMPLES:
        needed_bg = NUM_BG_SAMPLES - current_bg_count
        print(f"  - Generating {needed_bg} background sequences...")
        
        all_static_labels = list(static_data.keys())
        for idx in range(needed_bg):
            seq = np.zeros((SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32)
            
            # 50% chance: Static hold of a single letter
            # 50% chance: Transition from letter A to letter B
            if random.random() < 0.5:
                label = random.choice(all_static_labels)
                base_frame = random.choice(static_data[label])
                seq = np.tile(base_frame, (SEQUENCE_LENGTH, 1))
            else:
                label1 = random.choice(all_static_labels)
                label2 = random.choice(all_static_labels)
                frame1 = random.choice(static_data[label1])
                frame2 = random.choice(static_data[label2])
                
                # Linear interpolation to simulate transition
                for f_idx in range(SEQUENCE_LENGTH):
                    alpha = f_idx / (SEQUENCE_LENGTH - 1)
                    seq[f_idx] = (1 - alpha) * frame1 + alpha * frame2
            
            # Add Gaussian noise
            noise = np.random.normal(0, 0.003, seq.shape).astype(np.float32)
            for hand_offset in [0, 87]:
                hand_block = seq[:, hand_offset : hand_offset + 87]
                inactive_mask = (np.sum(np.abs(hand_block), axis=1) < 0.001)
                noise[inactive_mask, hand_offset : hand_offset + 87] = 0.0
            seq += noise
            
            # Save npy
            np.save(os.path.join(bg_dir, f"{current_bg_count + idx}.npy"), seq)

    # 3. Process Dynamic Classes (augmenting user seed recordings)
    print("\n[STEP 2] Augmenting dynamic gesture sequences...")
    all_dirs = sorted([d for d in os.listdir(DATASET_SEQ_DIR) if os.path.isdir(os.path.join(DATASET_SEQ_DIR, d))])
    
    for label in all_dirs:
        # Static classes were already handled or skipped
        if label in static_data:
            continue
            
        cls_dir = os.path.join(DATASET_SEQ_DIR, label)
        existing_files = sorted([f for f in os.listdir(cls_dir) if f.endswith('.npy')])
        current_count = len(existing_files)
        
        if current_count == 0:
            print(f"  - Class '{label}': 0 samples found. [ACTION REQUIRED] Please record at least 1-2 real sequence samples first.")
            continue
            
        if current_count >= TARGET_SAMPLES:
            print(f"  - Class '{label}': Already has {current_count} sequence samples. Skipping.")
            continue
            
        needed = TARGET_SAMPLES - current_count
        print(f"  - Class '{label}': Found {current_count} real samples. Synthesizing {needed} augmented samples...")
        
        # Load all existing real sequences
        real_seqs = []
        for f in existing_files:
            try:
                seq = np.load(os.path.join(cls_dir, f))
                # Resample seed sequence to target SEQUENCE_LENGTH if shape is different
                if seq.shape[0] != SEQUENCE_LENGTH:
                    indices = np.linspace(0, seq.shape[0] - 1, SEQUENCE_LENGTH)
                    resampled_seq = np.zeros((SEQUENCE_LENGTH, seq.shape[1]), dtype=np.float32)
                    for i in range(seq.shape[1]):
                        resampled_seq[:, i] = np.interp(indices, np.arange(seq.shape[0]), seq[:, i])
                    seq = resampled_seq
                    np.save(os.path.join(cls_dir, f), seq) # overwrite with 15 frames
                real_seqs.append(seq)
            except Exception as e:
                print(f"    [ERROR] Failed to load {f}: {e}")
                
        if not real_seqs:
            continue
            
        for idx in range(needed):
            # Select a random seed sequence
            seed_seq = random.choice(real_seqs)
            
            # Apply 1: Gaussian Jitter
            augmented = seed_seq.copy()
            noise = np.random.normal(0, 0.004, augmented.shape).astype(np.float32)
            
            # Apply 2: Spatial Scaling (stretch or contract hand landmarks slightly)
            scale = random.uniform(0.95, 1.05)
            augmented = augmented * scale
            
            # Apply 3: Temporal warping (speed up or slow down)
            warp = random.uniform(0.85, 1.15)
            augmented = temporal_warp(augmented, warp_factor=warp)
            
            # Apply active hand checks to avoid noise on zero-padded inactive hands
            for hand_offset in [0, 87]:
                hand_block = augmented[:, hand_offset : hand_offset + 87]
                inactive_mask = (np.sum(np.abs(hand_block), axis=1) < 0.001)
                noise[inactive_mask, hand_offset : hand_offset + 87] = 0.0
                
            augmented += noise
            
            # Save
            file_name = f"{current_count + idx}.npy"
            np.save(os.path.join(cls_dir, file_name), augmented)

    print("\n" + "="*60)
    print(" DATASET AUTOMATION & AUGMENTATION COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
