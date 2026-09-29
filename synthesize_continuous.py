#!/usr/bin/env python3
"""
Continuous Sequence Synthesis Tool (Phase 2 CTC Edition)
Synthesizes continuous multi-gesture sentences with isolated dynamic gestures.
Pads sequences with background segments (mapped to CTC blank token 0).
Dynamically handles variable feature dimensions (174 or 195 dims).
"""

import os
import json
import random
import numpy as np

DATASET_SEQ_DIR = "dataset_sequences"
OUTPUT_DIR = "dataset_sequences_continuous"
NUM_SYNTHETIC_SAMPLES = 500
FEATURE_SIZE = 174  # Dynamic fallback

def load_isolated_sequences():
    """Loads all npy sequences into a dictionary grouped by class."""
    global FEATURE_SIZE
    classes = [d for d in os.listdir(DATASET_SEQ_DIR) if os.path.isdir(os.path.join(DATASET_SEQ_DIR, d))]
    dataset = {}
    for cls in classes:
        cls_dir = os.path.join(DATASET_SEQ_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
        if len(files) > 0:
            dataset[cls] = []
            for f in files:
                seq = np.load(os.path.join(cls_dir, f))
                dataset[cls].append(seq)
                if len(seq.shape) > 1:
                    FEATURE_SIZE = seq.shape[1]
            print(f"[INFO] Loaded {len(dataset[cls])} isolated samples for class '{cls}' (Feature Dimension: {FEATURE_SIZE})")
    return dataset

def interpolate_frames(frame1, frame2, steps):
    """Linearly interpolates between two feature frames."""
    transition = np.zeros((steps, FEATURE_SIZE), dtype=np.float32)
    for i in range(steps):
        alpha = i / (steps - 1) if steps > 1 else 0.5
        transition[i] = (1 - alpha) * frame1 + alpha * frame2
    noise = np.random.normal(0, 0.003, transition.shape).astype(np.float32)
    transition += noise
    return transition

def main():
    print("="*60)
    print(" CONTINUOUS SEQUENCE SYNTHESIS (CTC DYNAMIC FEATURE SIZE)")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        isolated = load_isolated_sequences()
    except Exception as e:
        print(f"[ERROR] Failed to load isolated sequences: {e}")
        return
        
    meaningful_classes = [c for c in isolated.keys() if c != "background"]
    if not meaningful_classes:
        print("[ERROR] No non-background classes found in dataset_sequences.")
        return
        
    print(f"[INFO] Target classes for synthesis: {meaningful_classes}")
    label_to_idx = {cls: idx + 1 for idx, cls in enumerate(meaningful_classes)}
    print(f"[INFO] Label to Index mapping (1-indexed for CTC, 0 is blank): {label_to_idx}")
    
    with open(os.path.join(OUTPUT_DIR, "class_mapping.json"), "w") as f:
        json.dump(label_to_idx, f, indent=4)
        
    labels_dict = {}
    
    for idx in range(NUM_SYNTHETIC_SAMPLES):
        num_gestures = random.choices([1, 2, 3], weights=[0.25, 0.55, 0.20])[0]
        chosen_gestures = [random.choice(meaningful_classes) for _ in range(num_gestures)]
        
        sequence_parts = []
        
        # 1. Initial background padding
        if "background" in isolated and len(isolated["background"]) > 0:
            bg_seq = random.choice(isolated["background"])
            bg_len = random.randint(5, 15)
            sequence_parts.append(bg_seq[:bg_len])
            
        # 2. Add gestures and transitions
        for g_idx, gesture_name in enumerate(chosen_gestures):
            gest_seq = random.choice(isolated[gesture_name])
            
            # Temporal warping
            warp_factor = random.uniform(0.8, 1.2)
            orig_len = gest_seq.shape[0]
            new_len = max(5, int(orig_len * warp_factor))
            indices = np.linspace(0, orig_len - 1, new_len)
            warped_gest = np.zeros((new_len, FEATURE_SIZE), dtype=np.float32)
            for d in range(FEATURE_SIZE):
                warped_gest[:, d] = np.interp(indices, np.arange(orig_len), gest_seq[:, d])
            
            sequence_parts.append(warped_gest)
            
            if g_idx < num_gestures - 1:
                next_gesture_name = chosen_gestures[g_idx + 1]
                next_gest_seq = random.choice(isolated[next_gesture_name])
                trans_len = random.randint(5, 15)
                transition = interpolate_frames(warped_gest[-1], next_gest_seq[0], trans_len)
                sequence_parts.append(transition)
                
        # 3. Final background padding
        if "background" in isolated and len(isolated["background"]) > 0:
            bg_seq = random.choice(isolated["background"])
            bg_len = random.randint(5, 15)
            sequence_parts.append(bg_seq[-bg_len:])
            
        # Concatenate and add minor noise
        full_sequence = np.concatenate(sequence_parts, axis=0)
        global_noise = np.random.normal(0, 0.002, full_sequence.shape).astype(np.float32)
        full_sequence += global_noise
        
        filename = f"seq_{idx}.npy"
        np.save(os.path.join(OUTPUT_DIR, filename), full_sequence)
        
        target_idxs = [label_to_idx[g] for g in chosen_gestures]
        labels_dict[filename] = target_idxs
        
    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w") as f:
        json.dump(labels_dict, f, indent=4)
        
    print(f"[SUCCESS] Synthesized {NUM_SYNTHETIC_SAMPLES} continuous sequences in '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()
