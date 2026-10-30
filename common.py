"""Shared dataset schema and evaluation used during training and in the app."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
TARGET = "fault_type"
FEATURES = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity",
    "Maximum_of_Luminosity", "Length_of_Conveyer", "TypeOfSteel_A300",
    "TypeOfSteel_A400", "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
    "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
    "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
    "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas",
]
# Alphabetical order is also the order sklearn uses for string class labels.
CLASSES = sorted([
    "Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"
])
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
SEED = 5801
SOURCE_URL = "https://archive.ics.uci.edu/dataset/198/steel+plates+faults"


def validate_test_data(frame):
    """Require a labelled, numeric CSV; never silently drop or impute bad rows."""
    if frame.empty:
        raise ValueError("The CSV has no data rows.")
    if frame.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(required))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if unexpected:
        raise ValueError("Unexpected columns: " + ", ".join(unexpected))
    clean = frame.loc[:, required].copy()
    if clean.isna().any().any():
        raise ValueError("Missing values are not allowed in test features or labels.")
    try:
        clean[FEATURES] = clean[FEATURES].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("All 27 input features must be numeric.") from exc
    if not np.isfinite(clean[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Input features must be finite; NaN and infinity are invalid.")
    clean[TARGET] = clean[TARGET].astype(str)
    unknown = sorted(set(clean[TARGET]) - set(CLASSES))
    if unknown:
        raise ValueError("Unknown fault_type labels: " + ", ".join(unknown))
    return clean


def evaluate_model(pipeline, frame):
    """Evaluate an already fitted pipeline. No fitting occurs in this function."""
    y_true = frame[TARGET]
    x_test = frame[FEATURES]
    y_pred = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)
    # Align probability columns explicitly, rather than relying on file ordering.
    probability_order = [list(pipeline.classes_).index(label) for label in CLASSES]
    probabilities = probabilities[:, probability_order]
    auc = None
    if set(y_true) == set(CLASSES):
        auc = float(roc_auc_score(
            y_true, probabilities, labels=CLASSES, multi_class="ovr", average="macro"
        ))
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": auc,
        "Precision": float(precision_score(
            y_true, y_pred, labels=CLASSES, average="macro", zero_division=0
        )),
        "Recall": float(recall_score(
            y_true, y_pred, labels=CLASSES, average="macro", zero_division=0
        )),
        "F1": float(f1_score(
            y_true, y_pred, labels=CLASSES, average="macro", zero_division=0
        )),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    details = {
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
        ),
    }
    return metrics, details, y_pred, probabilities
