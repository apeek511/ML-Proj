"""
Train and store an SVM baseline model
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from src.utils import (
    file_to_sequences,
    make_binary_labels,
    make_label_array,
    evaluate_predictions,
)

ROOT = Path(__file__).resolve().parent
SVM_FILE_NAME = "svm_final.pkl"
SVM_SCALER_FILE_NAME = "scaler.pkl"
train_file_path = ROOT / "data" / "ae.train"
test_file_path = ROOT / "data" / "ae.test"
save_dir = ROOT / "saved_models"


AUTHENTICATED_SPEAKERS = [0, 1, 2]

LABEL_NAMES = ["stranger", "authenticated"]

if __name__ == "__main__":

    # === LOAD DATA ===
    train_seqs_raw = file_to_sequences(train_file_path, pad_sequences=True, max_seq_len=29)
    print("train shape:", train_seqs_raw.shape)
    test_seqs_raw = file_to_sequences(test_file_path, pad_sequences=True, max_seq_len=29)
    print("test shape:", test_seqs_raw.shape)

    # === STANDARDIZE ===
    # Load the scaler saved during GRU training for identical preprocessing
    scaler_path = save_dir / SVM_SCALER_FILE_NAME
    scaler = joblib.load(scaler_path)

    # Apply the same scaler to both train and test data
    n_train, seq_len, n_features = train_seqs_raw.shape
    n_test = test_seqs_raw.shape[0]

    train_seqs_scaled = scaler.transform(
        train_seqs_raw.reshape(-1, n_features)
    ).reshape(n_train, seq_len, n_features)

    test_seqs_scaled = scaler.transform(
        test_seqs_raw.reshape(-1, n_features)
    ).reshape(n_test, seq_len, n_features)

    # === FLATTEN ===
    # SVM needs 2D input: (n_samples, n_features)
    # Preserves all timestep values as independent features
    # Temporal ordering is implicit in feature position rather than explicitly modeled
    X_train = train_seqs_scaled.reshape(n_train, -1)  
    X_test = test_seqs_scaled.reshape(n_test, -1)     

    # === LABELS ===
    # Create original 9-class labels first then remap to binary
    label_train_counts = [30, 30, 30, 30, 30, 30, 30, 30, 30]
    original_train_labels = make_label_array(label_train_counts)
    train_labels = make_binary_labels(original_train_labels, AUTHENTICATED_SPEAKERS)

    label_test_counts = [31, 35, 88, 44, 29, 24, 40, 50, 29]
    original_test_labels = make_label_array(label_test_counts)
    test_labels = make_binary_labels(original_test_labels, AUTHENTICATED_SPEAKERS)

    print(f"Authenticated speakers: {[s + 1 for s in AUTHENTICATED_SPEAKERS]}")
    print(f"Train - authenticated: {train_labels.sum()}, stranger: {(train_labels == 0).sum()}")
    print(f"Test  - authenticated: {test_labels.sum()}, stranger: {(test_labels == 0).sum()}")

    # === HYPERPARAMETER SEARCH ===
    # Grid search over C and gamma using stratified 5-fold cross validation
    # Logarithmically spaced values 
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1, "scale"],
    }

    cv = StratifiedKFold(n_splits=5)
    svm = GridSearchCV(
        SVC(kernel="rbf", probability=True),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        verbose=1,
    )
    svm.fit(X_train, train_labels)

    print(f"\nBest parameters: {svm.best_params_}")
    print(f"Best CV F1: {svm.best_score_:.4f}")

    # === EVALUATE ON TEST SET ===
    test_preds = svm.predict(X_test)
    evaluate_predictions(test_labels, test_preds, title="SVM Baseline", label_names=LABEL_NAMES)

    # === SAVE MODEL ===
    save_dir.mkdir(exist_ok=True)
    svm_path = save_dir / SVM_FILE_NAME
    joblib.dump(svm, svm_path)
    print(f"SVM model saved to {svm_path}.")