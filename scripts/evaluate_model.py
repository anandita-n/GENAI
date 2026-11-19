"""
SignBridge AI - Standalone Model Evaluation & Benchmark Script
Loads saved test dataset, evaluates accuracy, precision, recall, F1-scores,
and prints a formatted classification table.
"""

import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from app.config import (
    CLASSIFIER_MODEL_PATH,
    LABEL_ENCODER_PATH,
    SCALER_PATH,
    METRICS_PATH,
    DATA_DIR
)


def run_evaluation():
    test_file = DATA_DIR / "asl_landmarks_test.npz"
    if not test_file.exists():
        print(f"Error: Test dataset not found at {test_file}. Please run scripts/train_model.py first.")
        return

    print("Loading test dataset and model artifacts...")
    data = np.load(test_file, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    classes = [str(c) for c in data["classes"]]

    clf = joblib.load(CLASSIFIER_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    print("\n" + "=" * 60)
    print("      SignBridge AI - Model Performance Evaluation Report")
    print("=" * 60)
    print(f"Total Test Samples:       {len(y_test)}")
    print(f"Total Classes Evaluated:  {len(classes)}")
    print(f"Overall Test Accuracy:    {acc * 100:.2f}%")
    print(f"Macro Precision:          {p_macro * 100:.2f}%")
    print(f"Macro Recall:             {r_macro * 100:.2f}%")
    print(f"Macro F1-Score:           {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score:        {f1_weighted * 100:.2f}%")
    print("=" * 60)

    print("\nDetailed Per-Class Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))


if __name__ == "__main__":
    run_evaluation()
