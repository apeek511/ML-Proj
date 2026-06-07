"""
Load and evaluate the SVM baseline and GRU main model
"""

from pathlib import Path

import joblib
import torch

from src.utils import (
    evaluate_predictions,
    file_to_sequences,
    logits_to_labels,
    make_binary_labels,
    make_label_array,
)

ROOT = Path(__file__).resolve().parent
GRU_FILE_NAME = "gru_final.pth"
SVM_FILE_NAME = "svm_final.pkl"
SCALER_FILE_NAME = "scaler.pkl"
test_file_path = ROOT / "data" / "ae.test"
save_dir = ROOT / "saved_models"

AUTHENTICATED_SPEAKERS = [0, 1, 2]
LABEL_NAMES = ["stranger", "authenticated"]

if __name__ == "__main__":
    # === LOAD TEST DATA ===
    test_seqs_raw = file_to_sequences(test_file_path, pad_sequences=True, max_seq_len=29)
 
    # === STANDARDIZE ===
    # Load the scaler fit during GRU training - same scaler used for both models
    scaler = joblib.load(save_dir / SCALER_FILE_NAME)
 
    n_test, seq_len, n_features = test_seqs_raw.shape
    test_seqs_scaled = scaler.transform(
        test_seqs_raw.reshape(-1, n_features)
    ).reshape(n_test, seq_len, n_features)
 
    # === LABELS ===
    label_test_counts = [31, 35, 88, 44, 29, 24, 40, 50, 29]
    original_test_labels = make_label_array(label_test_counts)
    test_labels_binary = make_binary_labels(original_test_labels, AUTHENTICATED_SPEAKERS)
 
    # === SVM BASELINE ===
    print("Evaluating SVM baseline...")
    svm = joblib.load(save_dir / SVM_FILE_NAME)

    # SVM needs flattened 2D input
    X_test = test_seqs_scaled.reshape(n_test, -1)  
    svm_preds = svm.predict(X_test)
 
    evaluate_predictions(test_labels_binary, svm_preds, title="SVM (Baseline model)", label_names=LABEL_NAMES)

    # === GRU MAIN MODEL ===
    print("Evaluating GRU main model...")
    stored_gru_model = torch.load(save_dir / GRU_FILE_NAME, weights_only=False)
    device = stored_gru_model.device
 
    # GRU needs 3D tensor input
    test_seqs_tensor = torch.from_numpy(test_seqs_scaled).to(device)
    test_labels_tensor = torch.from_numpy(test_labels_binary).to(device)
 
    stored_gru_model.eval()
    with torch.no_grad():
        logits = stored_gru_model(test_seqs_tensor)
 
    gru_preds = logits_to_labels(logits)
 
    evaluate_predictions(test_labels_tensor, gru_preds, title="GRU (Main model)", label_names=LABEL_NAMES)

