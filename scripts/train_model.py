"""
SignBridge AI - Model Training and Evaluation Script
Trains the ASL gesture classifier on normalized MediaPipe 3D landmark features,
computes evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix),
and serializes artifacts for production deployment.
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from app.config import (
    CLASSIFIER_MODEL_PATH,
    LABEL_ENCODER_PATH,
    SCALER_PATH,
    METRICS_PATH,
    ALL_CLASSES,
    DATA_DIR,
    MODELS_DIR
)
from app.asl_dataset import ASLKinematics


REAL_SAMPLE_WEIGHT = 5.0  # real webcam recordings count more than synthetic samples


def load_real_recordings():
    """
    Loads real landmark recordings made with scripts/record_landmarks.py
    (data/recorded_<SIGN>_<timestamp>.json). Returns (X, y) or (None, None).
    """
    X_list, y_list = [], []
    for path in sorted(DATA_DIR.glob("recorded_*.json")):
        try:
            with open(path, "r") as f:
                rec = json.load(f)
            sign = str(rec["sign"]).upper()
            if sign not in ALL_CLASSES:
                continue
            for feats in rec["features"]:
                if len(feats) == 83:
                    X_list.append(feats)
                    y_list.append(sign)
        except Exception as e:
            print(f"Skipping unreadable recording {path.name}: {e}")
    if not X_list:
        return None, None
    return np.array(X_list, dtype=np.float32), np.array(y_list)


def train_and_evaluate(samples_per_class: int = 500):
    print(f"==================================================")
    print(f" SignBridge AI - Classifier Training & Evaluation ")
    print(f"==================================================")
    print(f"Generating kinematic 3D landmark dataset for {len(ALL_CLASSES)} classes...")
    start_time = time.time()

    X, y = ASLKinematics.generate_dataset(samples_per_class=samples_per_class)
    print(f"Dataset generated: {X.shape[0]} samples with {X.shape[1]} features in {time.time() - start_time:.2f}s")

    # Encode labels
    le = LabelEncoder()
    le.fit(ALL_CLASSES)
    y_encoded = le.transform(y)

    # Train / Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # Real webcam recordings (optional): part goes into training with a higher
    # weight, the rest is held out to measure real-world accuracy honestly.
    sample_weight = np.ones(len(X_train), dtype=np.float32)
    X_real_test, y_real_test = None, None
    X_real, y_real_raw = load_real_recordings()
    if X_real is not None:
        y_real = le.transform(y_real_raw)
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real, test_size=0.25, random_state=42
        )
        print(f"Real recordings: {len(X_real)} samples "
              f"({len(X_real_train)} train / {len(X_real_test)} held-out test)")
        X_train = np.vstack([X_train, X_real_train])
        y_train = np.concatenate([y_train, y_real_train])
        sample_weight = np.concatenate(
            [sample_weight, np.full(len(X_real_train), REAL_SAMPLE_WEIGHT, dtype=np.float32)]
        )
    else:
        print("No real recordings found in data/. Training on synthetic data only "
              "(record real signs with scripts/record_landmarks.py to improve real-world accuracy).")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Model
    print("Training Random Forest Classifier (n_estimators=150, max_features='sqrt')...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train, sample_weight=sample_weight)

    real_world_metrics = None
    if X_real_test is not None and len(X_real_test) > 0:
        real_acc = float(accuracy_score(y_real_test, clf.predict(scaler.transform(X_real_test))))
        real_world_metrics = {"accuracy": round(real_acc, 4), "test_samples": int(len(X_real_test))}
        print(f"Real-world held-out accuracy: {real_acc * 100:.2f}% ({len(X_real_test)} samples)")

    # Evaluation on Test Set
    print("\n--- Evaluating Model on Test Set ---")
    eval_start = time.time()
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)
    inference_latency_ms = ((time.time() - eval_start) / len(X_test)) * 1000

    acc = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    print(f"Test Accuracy:         {acc * 100:.2f}%")
    print(f"Macro Precision:       {p_macro * 100:.2f}%")
    print(f"Macro Recall:          {r_macro * 100:.2f}%")
    print(f"Macro F1-Score:        {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score:     {f1_weight * 100:.2f}%")
    print(f"Avg Inference Latency: {inference_latency_ms:.3f} ms / frame")

    # Per-class metrics
    class_names = [str(c) for c in le.classes_]
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = confusion_matrix(y_test, y_pred, normalize="true")

    # Metrics payload for UI and reporting
    metrics_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": int(X.shape[0]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "num_classes": len(class_names),
        "classes": class_names,
        "overall": {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weight), 4),
            "inference_latency_ms": round(inference_latency_ms, 3)
        },
        "per_class": {
            cls_name: {
                "precision": round(report_dict[cls_name]["precision"], 4),
                "recall": round(report_dict[cls_name]["recall"], 4),
                "f1_score": round(report_dict[cls_name]["f1-score"], 4),
                "support": int(report_dict[cls_name]["support"])
            }
            for cls_name in class_names
        },
        "real_world": real_world_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_normalized, 4).tolist()
    }

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, CLASSIFIER_MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Save test set npz for offline benchmarking
    np.savez_compressed(
        DATA_DIR / "asl_landmarks_test.npz",
        X_test=X_test,
        y_test=y_test,
        classes=class_names
    )

    print(f"\nArtifacts successfully saved:")
    print(f" - Classifier:   {CLASSIFIER_MODEL_PATH}")
    print(f" - Label Encoder:{LABEL_ENCODER_PATH}")
    print(f" - Scaler:       {SCALER_PATH}")
    print(f" - Metrics JSON: {METRICS_PATH}")
    print(f" - Test Set:     {DATA_DIR / 'asl_landmarks_test.npz'}")
    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate(samples_per_class=500)
