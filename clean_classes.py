#!/usr/bin/env python3
"""
Dataset Cleaning Utility for Selective Sign Language Class Retraining
Deletes local samples and CSV records for low-accuracy classes and class '7'.
"""

import os
import csv
import json
import shutil

DATASET_DIR = "dataset"
CSV_PATH = os.path.join(DATASET_DIR, "hand_landmarks.csv")
ACCURACIES_PATH = "class_accuracies.json"

# Default low-accuracy classes from user's last run (used if class_accuracies.json is missing)
DEFAULT_LOW_ACC_CLASSES = ["2", "4", "5", "A", "D", "N", "O", "R", "U", "V"]

def load_accuracies():
    if os.path.exists(ACCURACIES_PATH):
        try:
            with open(ACCURACIES_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Error reading '{ACCURACIES_PATH}': {e}. Using fallback defaults.")
    return None

def clear_class_data(class_label):
    """Deletes all sample .npy files and the class directory."""
    class_dir = os.path.join(DATASET_DIR, class_label)
    deleted_files = 0
    if os.path.exists(class_dir):
        for f in os.listdir(class_dir):
            if f.endswith('.npy'):
                try:
                    os.remove(os.path.join(class_dir, f))
                    deleted_files += 1
                except OSError as e:
                    print(f"[ERROR] Failed to remove file {f}: {e}")
        try:
            os.rmdir(class_dir)
            print(f"  [DELETED] Class '{class_label}' directory and {deleted_files} sample files.")
        except OSError:
            # Folder might have other files, just leave it empty
            print(f"  [CLEARED] Class '{class_label}' directory files ({deleted_files} samples).")
    else:
        print(f"  [INFO] Class '{class_label}' directory does not exist. Skipping file deletion.")
    return deleted_files

def sweep_csv_records(cleared_classes):
    """Removes all rows for the cleared classes from the central CSV file."""
    if not os.path.exists(CSV_PATH):
        print(f"  [INFO] Central CSV file '{CSV_PATH}' not found. Skipping CSV sweep.")
        return 0

    temp_rows = []
    removed_count = 0
    try:
        with open(CSV_PATH, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                temp_rows.append(header)
            
            for row in reader:
                if row:
                    if row[0] in cleared_classes:
                        removed_count += 1
                    else:
                        temp_rows.append(row)
        
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(temp_rows)
            
        print(f"  [SWEPT] Removed {removed_count} rows matching cleared classes from '{CSV_PATH}'.")
    except Exception as e:
        print(f"[ERROR] Failed to update CSV file: {e}")
    return removed_count

def main():
    print("="*60)
    print(" SIGN LANGUAGE DATASET CLEANING & OPTIMIZATION UTILITY")
    print("="*60)

    # 1. Identify classes to clear
    accuracies = load_accuracies()
    cleared_classes = {}
    
    # Class '7' is always cleared as requested
    cleared_classes["7"] = "Explicit request (remove class entirely)"

    if accuracies:
        print("[INFO] Loaded validation accuracies from 'class_accuracies.json'")
        for label, acc in accuracies.items():
            if acc < 0.95 and label != "7":
                cleared_classes[label] = f"Validation Accuracy: {acc*100:.2f}% (< 95.00%)"
    else:
        print("[INFO] 'class_accuracies.json' not found. Using reported low-accuracy fallback defaults.")
        for label in DEFAULT_LOW_ACC_CLASSES:
            cleared_classes[label] = "Fallback default (< 95.00% validation accuracy)"

    # 2. Present details to user
    print("\nThe following classes will be CLEARED and reset for retraining:")
    for label, reason in sorted(cleared_classes.items()):
        print(f"  - Class '{label:<2}': {reason}")
        
    print("\n[WARNING] This action is irreversible. All local samples and CSV records")
    print("for these classes will be permanently deleted.")
    
    # Prompt for confirmation
    confirm = input("\nDo you want to proceed with clearing these classes? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("\n[INFO] Dataset cleaning cancelled. No files were modified.")
        print("="*60)
        return

    print("\nStarting dataset clean-up...")
    print("-"*60)
    
    total_files_deleted = 0
    for label in cleared_classes.keys():
        total_files_deleted += clear_class_data(label)
        
    total_rows_removed = sweep_csv_records(cleared_classes.keys())
    
    # Remove class_accuracies.json to keep state fresh
    if os.path.exists(ACCURACIES_PATH):
        try:
            os.remove(ACCURACIES_PATH)
        except OSError:
            pass

    print("-"*60)
    print("[SUCCESS] Dataset cleaning complete!")
    print(f"[INFO] Deleted {total_files_deleted} .npy sample files.")
    print(f"[INFO] Removed {total_rows_removed} rows from central CSV.")
    print("\nYou can now run 'python3 data_collection.py' to record high-quality")
    print("samples specifically for the cleared classes.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
