#!/usr/bin/env python3
"""
Sign Language CTC Evaluation & Diagnostic Tool
Loads 'model_sequence_ctc.pth' and 'label_mapping_ctc.json' and evaluates predictions
on synthetic continuous test sequences and isolated gestures.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = "model_sequence_ctc.pth"
LABEL_MAPPING_PATH = "label_mapping_ctc.json"
DATASET_CONT_DIR = "dataset_sequences_continuous"
DATASET_SEQ_DIR = "dataset_sequences"

class SignLanguageGRU_CTC(nn.Module):
    def __init__(self, input_size=174, hidden_size=128, num_layers=2, num_classes=5):
        super(SignLanguageGRU_CTC, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, lengths):
        lengths_cpu = lengths.cpu().to(torch.int64)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        logits = self.fc(out)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
        return log_probs

def ctc_decode(seq_logits, idx_to_label):
    pred_indices = torch.argmax(seq_logits, dim=-1).cpu().numpy()
    collapsed = []
    prev = -1
    for idx in pred_indices:
        if idx != prev:
            if idx != 0:
                collapsed.append(idx)
            prev = idx
    labels = [idx_to_label[str(p)] for p in collapsed if str(p) in idx_to_label]
    return labels

def main():
    print("="*60)
    print(" CTC SEQUENCE MODEL EVALUATION")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAPPING_PATH):
        print("[ERROR] Model files not found! Please run 'train_ctc.py' first.")
        return

    with open(LABEL_MAPPING_PATH, 'r') as f:
        idx_to_label = json.load(f)
    print(f"[INFO] Loaded label mapping: {idx_to_label}")

    labels_file = os.path.join(DATASET_CONT_DIR, "labels.json")
    if not os.path.exists(labels_file):
        print("[ERROR] Continuous labels not found! Please run 'synthesize_continuous.py'.")
        return
        
    with open(labels_file, 'r') as f:
        continuous_labels = json.load(f)

    # Detect input feature size dynamically from first sample
    sample_file = list(continuous_labels.keys())[0]
    sample_seq = np.load(os.path.join(DATASET_CONT_DIR, sample_file))
    input_size = sample_seq.shape[1]
    print(f"[INFO] Detected input feature size from continuous data: {input_size}")

    num_classes = len(idx_to_label)
    model = SignLanguageGRU_CTC(input_size=input_size, hidden_size=128, num_layers=2, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    samples = list(continuous_labels.items())[:15]
    print("\n--- Evaluating Synthetic Continuous Sequences ---")
    correct = 0
    total = 0
    
    for filename, target_idxs in samples:
        seq_path = os.path.join(DATASET_CONT_DIR, filename)
        if not os.path.exists(seq_path):
            continue
        seq = np.load(seq_path)
        seq_len = len(seq)
        seq_tensor = torch.tensor([seq], dtype=torch.float32)
        lengths = torch.tensor([seq_len], dtype=torch.long)
        
        with torch.no_grad():
            log_probs = model(seq_tensor, lengths)  # (seq_len, 1, num_classes)
            log_probs_t = log_probs.transpose(0, 1)  # (1, seq_len, num_classes)
            pred_labels = ctc_decode(log_probs_t[0], idx_to_label)
            
        target_labels = [idx_to_label[str(i)] for i in target_idxs]
        is_match = (pred_labels == target_labels)
        if is_match:
            correct += 1
        total += 1
        print(f"File: {filename:10s} | Target: {str(target_labels):25s} | Pred: {str(pred_labels):25s} | Match: {is_match}")
        
    acc = (correct / total) * 100.0 if total > 0 else 0
    print(f"\nExact Sequence Match Accuracy (Sample): {acc:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()
