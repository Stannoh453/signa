#!/usr/bin/env python3
"""
Sign Language Gesture Sequence PyTorch Training Script (Phase 2 - Sequence)
Loads 12-frame gesture sequences from dataset_sequences,
trains a Bidirectional LSTM model, saves the best weights,
and generates training curves.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import datetime

# ==========================================
# Configuration & Constants
# ==========================================
DATASET_SEQ_DIR = "dataset_sequences"
MODEL_SAVE_PATH = "model_sequence.pth"
LABEL_MAPPING_PATH = "label_mapping_sequence.json"

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
REPORT_DIR = os.path.join("training_reports", f"sequence_model_{timestamp}")
os.makedirs(REPORT_DIR, exist_ok=True)

METRICS_PLOT_PATH = os.path.join(REPORT_DIR, "training_metrics_sequence.png")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ==========================================
# Data Loading & Preprocessing
# ==========================================
def load_sequence_dataset():
    """Scans the dataset_sequences folder, loads all .npy sequences, and builds label mapping."""
    if not os.path.exists(DATASET_SEQ_DIR):
        os.makedirs(DATASET_SEQ_DIR, exist_ok=True)
        raise FileNotFoundError(f"[ERROR] '{DATASET_SEQ_DIR}' directory is empty. Please collect sequence data first.")

    classes = sorted([d for d in os.listdir(DATASET_SEQ_DIR) if os.path.isdir(os.path.join(DATASET_SEQ_DIR, d))])
    
    # Filter classes that actually have samples
    valid_classes = []
    features = []
    str_labels = []
    
    for cls in classes:
        cls_dir = os.path.join(DATASET_SEQ_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
        if len(files) > 0:
            valid_classes.append(cls)
            print(f"[INFO] Class '{cls}': found {len(files)} sequence samples.")
            for f in files:
                filepath = os.path.join(cls_dir, f)
                try:
                    seq = np.load(filepath)
                    # Expected shape: (15, 174)
                    if seq.shape == (15, 174):
                        features.append(seq)
                        str_labels.append(cls)
                    else:
                        print(f"[WARNING] Skipping {filepath} due to incorrect shape: {seq.shape}")
                except Exception as e:
                    print(f"[WARNING] Failed to load {filepath}: {e}")
                    
    num_classes = len(valid_classes)
    if num_classes == 0:
        raise ValueError(f"[ERROR] No valid sequence samples found in '{DATASET_SEQ_DIR}'.")
        
    features = np.array(features, dtype=np.float32)
    print(f"[INFO] Loaded {len(features)} total sequences across {num_classes} classes.")
    print(f"[INFO] Sequence Classes: {', '.join(valid_classes)}")

    # Create encoding mappings
    label_to_idx = {label: idx for idx, label in enumerate(valid_classes)}
    idx_to_label = {idx: label for idx, label in enumerate(valid_classes)}
    
    # Save the label mapping for inference
    with open(LABEL_MAPPING_PATH, 'w') as f:
        json.dump(idx_to_label, f, indent=4)
    print(f"[INFO] Saved label mapping to '{LABEL_MAPPING_PATH}'.")

    # Encode label strings to integer targets
    encoded_labels = np.array([label_to_idx[lbl] for lbl in str_labels], dtype=np.int64)

    return features, encoded_labels, num_classes, valid_classes

class SequenceDataset(Dataset):
    """Custom PyTorch Dataset for landmark sequences with optional noise augmentation."""
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Add subtle Gaussian noise to simulate tracker jitter
            noise = torch.randn_like(x) * 0.005
            # Zero out noise where hand coordinates are inactive (i.e. zero-padded)
            # x shape: (15, 174)
            for frame_idx in range(15):
                for hand_offset in [0, 87]:
                    hand_block = x[frame_idx, hand_offset : hand_offset + 87]
                    if torch.sum(torch.abs(hand_block)) < 0.001:
                        noise[frame_idx, hand_offset : hand_offset + 87] = 0.0
                    else:
                        # Apply coordinate dropout to active hands (simulating occlusion)
                        if np.random.rand() < 0.15:
                            drop_idx = np.random.randint(0, 63)
                            drop_len = np.random.randint(3, 15)
                            x[frame_idx, hand_offset + drop_idx : hand_offset + min(drop_idx + drop_len, 63)] = 0.0
            x += noise
        return x, self.y[idx]

# ==========================================
# Model Architecture
# ==========================================
class SignLanguageLSTM(nn.Module):
    """Bidirectional LSTM Classifier for Hand Landmark Sequences."""
    def __init__(self, input_size=144, hidden_size=128, num_layers=2, num_classes=5):
        super(SignLanguageLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM to capture spatio-temporal features forward and backward
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Pull output from the final time-step of the LSTM (combining forward and backward directions)
        out = out[:, -1, :] # shape: (batch_size, hidden_size * 2)
        
        # Classification
        out = self.fc(out)
        return out

# ==========================================
# Training and Evaluation
# ==========================================
def train_model():
    # Load dataset
    try:
        X, y, num_classes, classes = load_sequence_dataset()
    except Exception as e:
        print(e)
        return

    # Train/Validation Split
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split_idx = int(num_samples * TRAIN_SPLIT)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    train_dataset = SequenceDataset(X_train, y_train, augment=True)
    val_dataset = SequenceDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Model instantiation
    model = SignLanguageLSTM(input_size=174, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Cosine annealing scheduler: reduces LR smoothly over training epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print("\n" + "="*50)
    print(" STARTING LSTM SEQUENCE TRAINING")
    print("="*50)
    
    for epoch in range(1, EPOCHS + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()  # Step LR after each epoch
        
        # 2. Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                running_val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total if total > 0 else 0
        
        # Save metrics history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Check and save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")
            
    print("="*50)
    print(f" Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print("="*50 + "\n")
    
    # Save training curve plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot([acc * 100 for acc in history['val_acc']], label='Val Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(METRICS_PLOT_PATH)
    plt.close()
    print(f"[INFO] Saved training metrics plot to '{METRICS_PLOT_PATH}'.")

if __name__ == "__main__":
    train_model()
