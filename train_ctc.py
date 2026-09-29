#!/usr/bin/env python3
"""
Sign Language Continuous CTC PyTorch Sequence Training Script (Phase 2)
Trains a Bidirectional GRU model using Connectionist Temporal Classification (CTCLoss).
Dynamically handles variable input feature dimensions (e.g. 174 or 195 dims).
Saves model weights to 'model_sequence_ctc.pth' and mapping to 'label_mapping_ctc.json'.
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# Configuration & Constants
# ==========================================
DATASET_SEQ_DIR = "dataset_sequences"
DATASET_CONT_DIR = "dataset_sequences_continuous"
MODEL_SAVE_PATH = "model_sequence_ctc.pth"
LABEL_MAPPING_PATH = "label_mapping_ctc.json"
REPORTS_DIR = "training_reports"

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8

# ==========================================
# Dataset Loader Utility
# ==========================================
def load_combined_dataset():
    """
    Loads synthetic continuous sequences and isolated sequences.
    Label 0 is reserved for CTC <blank>.
    """
    labels_file = os.path.join(DATASET_CONT_DIR, "labels.json")
    mapping_file = os.path.join(DATASET_CONT_DIR, "class_mapping.json")
    
    if not os.path.exists(labels_file) or not os.path.exists(mapping_file):
        raise FileNotFoundError(f"[ERROR] Continuous dataset missing! Please run 'synthesize_continuous.py' first.")
        
    with open(mapping_file, "r") as f:
        label_to_idx = json.load(f)  # e.g., {"J": 1, "Z": 2, "hello": 3, "peace": 4}
        
    with open(labels_file, "r") as f:
        continuous_labels = json.load(f)
        
    # Map index back to label for inference (including blank 0)
    idx_to_label = {"0": "blank"}
    for label, idx in label_to_idx.items():
        idx_to_label[str(idx)] = label
        
    X_list = []
    y_list = []
    
    # 1. Load synthetic continuous sequences
    print(f"[INFO] Loading synthetic continuous sequences from '{DATASET_CONT_DIR}'...")
    for filename, targets in continuous_labels.items():
        filepath = os.path.join(DATASET_CONT_DIR, filename)
        if os.path.exists(filepath):
            seq = np.load(filepath)
            X_list.append(seq)
            y_list.append(targets)
            
    # 2. Load isolated sequences as single-gesture target sequences
    print(f"[INFO] Loading isolated sequences from '{DATASET_SEQ_DIR}'...")
    for cls_name, cls_idx in label_to_idx.items():
        cls_dir = os.path.join(DATASET_SEQ_DIR, cls_name)
        if os.path.exists(cls_dir):
            files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
            for f in files:
                seq = np.load(os.path.join(cls_dir, f))
                X_list.append(seq)
                y_list.append([cls_idx])
                
    num_classes = len(idx_to_label)  # total including blank
    print(f"[INFO] Loaded total {len(X_list)} sequence samples. Vocabulary classes (including CTC blank): {num_classes}")
    
    return X_list, y_list, num_classes, idx_to_label

# ==========================================
# PyTorch CTC Dataset & Collate Function
# ==========================================
class CTCDataset(Dataset):
    def __init__(self, X_list, y_list, augment=False):
        self.X_list = X_list
        self.y_list = y_list
        self.augment = augment

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        x = self.X_list[idx].copy()
        y = self.y_list[idx]

        if self.augment:
            # Subtle feature noise
            noise = np.random.normal(0, 0.003, x.shape).astype(np.float32)
            x += noise
            # Subtle random scaling
            scale = np.random.uniform(0.95, 1.05)
            x *= scale

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def ctc_collate_fn(batch):
    """Pads feature sequences in batch and creates length vectors for CTCLoss."""
    sequences, targets = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    target_lengths = torch.tensor([tgt.size(0) for tgt in targets], dtype=torch.long)
    
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    flat_targets = torch.cat(targets)
    
    return padded_sequences, flat_targets, lengths, target_lengths

# ==========================================
# PyTorch CTC Model Definition
# ==========================================
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
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)  # (seq_len, batch_size, num_classes)
        return log_probs

# ==========================================
# Decoding & Evaluation Helpers
# ==========================================
def ctc_greedy_decode(log_probs, lengths):
    """Greedy CTC decoder: collapses repeating indices and removes blank token (0)."""
    # log_probs shape: (seq_len, batch_size, num_classes)
    log_probs_t = log_probs.transpose(0, 1)  # (batch_size, seq_len, num_classes)
    predictions = []
    
    for i in range(log_probs_t.size(0)):
        seq_len = lengths[i].item()
        seq_logits = log_probs_t[i, :seq_len]
        pred_indices = torch.argmax(seq_logits, dim=-1).cpu().numpy()
        
        collapsed = []
        prev = -1
        for idx in pred_indices:
            if idx != prev:
                if idx != 0:  # 0 is CTC blank
                    collapsed.append(idx)
                prev = idx
        predictions.append(collapsed)
        
    return predictions

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    exact_matches = 0
    total_sequences = 0
    
    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in data_loader:
            inputs = inputs.to(device)
            input_lengths = input_lengths.to(device)
            
            log_probs = model(inputs, input_lengths)
            pred_sequences = ctc_greedy_decode(log_probs, input_lengths)
            
            # Reconstruct target sequences from flat_targets using target_lengths
            start_idx = 0
            for i, tgt_len in enumerate(target_lengths):
                tgt_seq = targets[start_idx : start_idx + tgt_len.item()].tolist()
                start_idx += tgt_len.item()
                
                if pred_sequences[i] == tgt_seq:
                    exact_matches += 1
                total_sequences += 1
                
    return (exact_matches / total_sequences) * 100.0 if total_sequences > 0 else 0.0

# ==========================================
# Main Training Loop
# ==========================================
def main():
    print("="*60)
    print(" SIGNFLOW CTC SEQUENCE MODEL TRAINING")
    print("="*60)
    
    try:
        X, y, num_classes, idx_to_label = load_combined_dataset()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Train/Validation Split
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split_idx = int(num_samples * TRAIN_SPLIT)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    
    train_dataset = CTCDataset(X_train, y_train, augment=True)
    val_dataset = CTCDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ctc_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ctc_collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    input_size = X[0].shape[1]
    print(f"[INFO] Detected input feature size: {input_size}")
    
    model = SignLanguageGRU_CTC(input_size=input_size, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    start_time = time.time()
    
    print("\n--- Training Progress ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for inputs, targets, input_lengths, target_lengths in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            log_probs = model(inputs, input_lengths)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        scheduler.step()
        train_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets, input_lengths, target_lengths in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)
                
                log_probs = model(inputs, input_lengths)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_running_loss += loss.item() * inputs.size(0)
                
        val_loss = val_running_loss / len(val_dataset)
        val_acc = evaluate_accuracy(model, val_loader, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            with open(LABEL_MAPPING_PATH, 'w') as f:
                json.dump(idx_to_label, f, indent=4)
                
        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"Epoch [{epoch:3d}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Seq Acc: {val_acc:.2f}%")
            
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"[SUCCESS] Training Complete in {elapsed:.2f} seconds!")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Best Validation Acc: {best_val_acc:.2f}%")
    print(f"Saved model to '{MODEL_SAVE_PATH}' and mapping to '{LABEL_MAPPING_PATH}'.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
