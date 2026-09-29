#!/usr/bin/env python3
"""
Sign Language Alphabet & Numbers PyTorch Training Script (Phase 2)
Loads the wrist-relative and scaled landmark dataset from CSV,
trains a custom Deep Neural Network, saves the best weights,
and generates training curves and a confusion matrix.
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import datetime

# ==========================================
# Configuration & Constants
# ==========================================
CSV_PATH = "dataset/hand_landmarks.csv"
MODEL_SAVE_PATH = "model.pth"
LABEL_MAPPING_PATH = "label_mapping.json"

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
REPORT_DIR = os.path.join("training_reports", f"static_model_{timestamp}")
os.makedirs(REPORT_DIR, exist_ok=True)

METRICS_PLOT_PATH = os.path.join(REPORT_DIR, "training_metrics.png")
CONF_MATRIX_PLOT_PATH = os.path.join(REPORT_DIR, "confusion_matrix.png")
ACCURACIES_PATH = os.path.join(REPORT_DIR, "class_accuracies.json")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 80
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ==========================================
# Data Loading & Preprocessing
# ==========================================
def load_dataset():
    """Reads the CSV file, encodes string labels, and returns features/labels along with mapping."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"[ERROR] '{CSV_PATH}' not found. Please collect data first using 'data_collection.py'.")

    features = []
    str_labels = []

    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header row
        
        for row in reader:
            if not row:
                continue
            str_labels.append(row[0])
            features.append([float(val) for val in row[1:]])

    features = np.array(features, dtype=np.float32)
    
    # Identify unique classes present in the dataset
    unique_classes = sorted(list(set(str_labels)))
    num_classes = len(unique_classes)
    
    print(f"[INFO] Loaded {len(features)} samples across {num_classes} unique classes.")
    print(f"[INFO] Detected Classes: {', '.join(unique_classes)}")

    # Create encoding mappings
    label_to_idx = {label: idx for idx, label in enumerate(unique_classes)}
    idx_to_label = {idx: label for idx, label in enumerate(unique_classes)}
    
    # Save the label mapping for Phase 3 inference
    with open(LABEL_MAPPING_PATH, 'w') as f:
        json.dump(idx_to_label, f, indent=4)
    print(f"[INFO] Saved label mapping dictionary to '{LABEL_MAPPING_PATH}'.")

    # Encode label strings to integer targets
    encoded_labels = np.array([label_to_idx[lbl] for lbl in str_labels], dtype=np.int64)

    return features, encoded_labels, num_classes, unique_classes

class LandmarkDataset(Dataset):
    """Custom PyTorch Dataset for hand landmark coordinate vectors with augmentation."""
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # We apply augmentation to each hand block independently if that hand is present (not all zeros)
            for start_idx in [0, 87]:
                hand_block = x[start_idx : start_idx + 87]
                if torch.sum(torch.abs(hand_block)) > 0.001:
                    # 1. 3D Rotation (simulating hand tilt/yaw/roll)
                    pitch = (np.random.rand() - 0.5) * 0.2
                    yaw = (np.random.rand() - 0.5) * 0.2
                    roll = (np.random.rand() - 0.5) * 0.2
                    
                    Rx = torch.tensor([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]], dtype=torch.float32)
                    Ry = torch.tensor([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]], dtype=torch.float32)
                    Rz = torch.tensor([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]], dtype=torch.float32)
                    R = torch.matmul(Rz, torch.matmul(Ry, Rx))
                    
                    coords = hand_block[:63].view(-1, 3)
                    coords = torch.matmul(coords, R)
                    hand_block[:63] = coords.view(-1)
                    
                    # 2. Additive noise (simulating tracker jitter)
                    noise = torch.randn(87) * 0.01
                    
                    # 3. Scaling variation (simulating slight distance/scale changes)
                    scale = 0.95 + np.random.rand() * 0.1  # 0.95 to 1.05
                    
                    x[start_idx : start_idx + 87] = (hand_block + noise) * scale
                    
                    # 4. Occlusion (simulating missing fingers/tracker failure)
                    if np.random.rand() < 0.15:
                        drop_idx = np.random.randint(0, 63)
                        drop_len = np.random.randint(3, 15)
                        x[start_idx + drop_idx : start_idx + min(drop_idx + drop_len, 63)] = 0.0
                    
        return x, self.y[idx]

# ==========================================
# Model Architecture
# ==========================================
class SignLanguageCNN(nn.Module):
    """
    1D Convolutional Neural Network for spatial pattern recognition
    in hand landmarks.
    """
    def __init__(self, input_size=174, num_classes=32):
        super(SignLanguageCNN, self).__init__()
        
        # input shape: (batch_size, 1, 174)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate flattened size:
        # 174 / 2 = 87
        # 87 / 2 = 43
        # 43 / 2 = 21
        # Flattened size = 256 * 21 = 5376
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5376, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x is originally (batch, 174). Reshape to (batch, 1, 174) for Conv1d
        x = x.unsqueeze(1)
        
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ==========================================
# Confusion Matrix Utility
# ==========================================
def generate_and_save_confusion_matrix(model, val_loader, unique_classes, device):
    """Generates predictions on the validation set, computes the confusion matrix, and saves it."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    num_classes = len(unique_classes)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    # Plot Confusion Matrix
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Validation Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, unique_classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, unique_classes, fontsize=10)

    # Label grid cells with counts if size is reasonable
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=8)

    plt.ylabel('True Class Label', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Class Label', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PLOT_PATH, dpi=300)
    print(f"[INFO] Confusion matrix plot saved to '{CONF_MATRIX_PLOT_PATH}'.")

# ==========================================
# Main Training Loop
# ==========================================
def main():
    try:
        X, y, num_classes, unique_classes = load_dataset()
    except Exception as e:
        print(e)
        return

    # Stratified split into Train (80%) and Val (20%)
    # Ensures every class is represented proportionally in both sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=(1.0 - TRAIN_SPLIT),
        random_state=RANDOM_SEED,
        stratify=y
    )

    print(f"[INFO] Training set size: {len(X_train)} samples")
    print(f"[INFO] Validation set size: {len(X_val)} samples")

    # Create PyTorch DataLoaders with training augmentation
    train_dataset = LandmarkDataset(X_train, y_train, augment=True)
    val_dataset = LandmarkDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss, and optimizer
    model = SignLanguageCNN(input_size=174, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    # Add L2 regularization (weight_decay) to help generalization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Cosine annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Device configuration (use CPU as landmark networks are extremely lightweight)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[INFO] Training on device: {device}")

    # Track metrics for plotting
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    best_val_acc = 0.0

    print("\n" + "="*50)
    print(" TRAINING PROGRESS")
    print("="*50)
    print(f"{'Epoch':<8}{'Train Loss':<12}{'Train Acc (%)':<15}{'Val Loss':<12}{'Val Acc (%)'}")
    print("-"*50)

    for epoch in range(1, EPOCHS + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100

        # 2. Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100

        # Store statistics
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        # Print stats every 5 epochs, or first and last
        is_printed_epoch = (epoch == 1 or epoch == EPOCHS or epoch % 5 == 0)
        
        # Save model if validation accuracy improves
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            # Print explicit save notice to avoid confusion about save checkpoints
            if not is_printed_epoch:
                print(f"[SAVED]  Epoch {epoch:<4}: Val Acc improved to {epoch_val_acc:.2f}% (Loss: {epoch_val_loss:.4f})")
            else:
                # Append a marker to the standard output line
                save_marker = " *Saved*"
            state_str = " *Saved*"
        else:
            state_str = ""

        # Step learning rate scheduler
        scheduler.step()

        if is_printed_epoch:
            print(f"{epoch:<8}{epoch_train_loss:<12.4f}{epoch_train_acc:<15.2f}{epoch_val_loss:<12.4f}{epoch_val_acc:<11.2f}{state_str}")

    print("="*50)
    print(f"[SUCCESS] Training completed! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[INFO] Best model weights saved to '{MODEL_SAVE_PATH}'.")

    # ==========================================
    # Generate Confusion Matrix & Reports
    # ==========================================
    # Load the best saved model state
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    generate_and_save_confusion_matrix(model, val_loader, unique_classes, device)

    # Evaluate class-wise accuracy
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            for t, p in zip(targets, preds):
                class_total[t.item()] += 1
                if t.item() == p.item():
                    class_correct[t.item()] += 1
                    
    print("\n" + "="*50)
    print(" CLASS-WISE VALIDATION ACCURACY REPORT")
    print("="*50)
    class_accuracies = {}
    for idx, label in enumerate(unique_classes):
        total = class_total[idx]
        acc = (class_correct[idx] / total * 100) if total > 0 else 0
        print(f"  Class {label:<3} : {class_correct[idx]:>3}/{total:<3} ({acc:.2f}%)")
        class_accuracies[label] = acc / 100.0
    print("="*50 + "\n")

    # Save class accuracies for dataset cleaning tool
    with open(ACCURACIES_PATH, "w") as f:
        json.dump(class_accuracies, f, indent=4)
    print(f"[INFO] Saved class-wise validation accuracies to '{ACCURACIES_PATH}'.")

    # ==========================================
    # Plot Metrics & Save
    # ==========================================
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="blue")
    plt.plot(history["val_loss"], label="Val Loss", color="red")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", color="blue")
    plt.plot(history["val_acc"], label="Val Acc", color="red")
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(METRICS_PLOT_PATH)
    print(f"[INFO] Training curves plot saved to '{METRICS_PLOT_PATH}'.")

if __name__ == "__main__":
    main()
