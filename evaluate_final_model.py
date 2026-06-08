"""
Load and evaluate the SVM baseline and GRU main model
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve


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

def evaluate_auroc(labels_true: np.ndarray, proba_scores: np.ndarray, title: str) -> float:
    """
    Compute and plot AUROC for a binary classifier
 
    Args:
        labels_true (np.ndarray): True binary labels of shape (n_samples,)
        proba_scores (np.ndarray): Predicted probability of authenticated class of shape (n_samples,)
        title (str): Model name for plot title and print output
 
    Returns:
        float: The AUROC score
    """
    auroc = roc_auc_score(labels_true, proba_scores)
    print(f"\n--- AUROC for {title}: {auroc:.4f}")
 
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(labels_true, proba_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{title} (AUROC = {auroc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title}")
    plt.legend()
    plt.tight_layout()
    plt.show()
 
    return auroc


def evaluate_calibration(labels_true: np.ndarray, proba_scores: np.ndarray, title: str) -> None:
    """
    Plot a calibration curve to assess whether confidence scores are trustworthy
 
    A well calibrated model's confidence scores reflect its true accuracy
 
    Args:
        labels_true (np.ndarray): True binary labels of shape (n_samples,)
        proba_scores (np.ndarray): Predicted probability of authenticated class of shape (n_samples,)
        title (str): Model name for plot title
    """
    # calibration_curve bins predictions and computes mean predicted vs actual fraction
    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels_true, proba_scores, n_bins=10
    )
 
    plt.figure(figsize=(6, 5))
    plt.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label=f"{title}",
    )
    # Perfect calibration line - if the model is perfectly calibrated
    # it should follow this diagonal exactly
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Fraction actually authenticated")
    plt.title(f"Calibration Curve - {title}")
    plt.legend()
    plt.tight_layout()
    plt.show()


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

    # returns probabilities for each class, take the authenticated column
    svm_proba = svm.predict_proba(X_test)[:, 1]

    evaluate_predictions(test_labels_binary, svm_preds, title="SVM (Baseline model)", label_names=LABEL_NAMES)
    evaluate_auroc(test_labels_binary, svm_proba, title="SVM (Baseline model)")
    evaluate_calibration(test_labels_binary, svm_proba, title="SVM (Baseline model)")

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

    # convert logits to probabilities using softmax, take authenticated column
    gru_proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
 
    evaluate_predictions(test_labels_tensor, gru_preds, title="GRU (Main model)", label_names=LABEL_NAMES)
    evaluate_auroc(test_labels_binary, gru_proba, title="GRU (Main model)")
    evaluate_calibration(test_labels_binary, gru_proba, title="GRU (Main model)")


