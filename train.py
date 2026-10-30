"""Fit the five required classifiers on one split and save reproducible outputs.

Run this script in BITS Virtual Lab after reviewing and adapting the project.
The Streamlit app never imports or calls this training script.
"""

import hashlib
import json
import platform
from contextlib import redirect_stdout
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import (
    CLASSES, FEATURES, METRICS, MODEL_FILES, ROOT, SEED, SOURCE_URL, TARGET,
    evaluate_model, validate_test_data,
)


def build_models():
    """Fixed baseline settings, chosen before inspecting held-out test scores."""
    def scaled(classifier):
        return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])

    return {
        "Logistic Regression": scaled(LogisticRegression(C=1.0, max_iter=3000)),
        "Decision Tree": Pipeline([("classifier", DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=4, random_state=SEED
        ))]),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=9, weights="distance")),
        "Naive Bayes": scaled(GaussianNB(var_smoothing=1e-9)),
        "Support Vector Machine": scaled(SVC(
            C=5.0, kernel="rbf", gamma="scale", probability=True, random_state=SEED
        )),
    }


def main():
    for folder in [ROOT / "model", ROOT / "results"]:
        folder.mkdir(exist_ok=True)
    dataset_path = ROOT / "data" / "steel_plates_faults.csv"
    dataset = validate_test_data(pd.read_csv(dataset_path))
    if dataset.shape != (1941, 28) or dataset.duplicated().any():
        raise ValueError("The bundled source dataset has changed; review it before training.")
    # Partition before fitting any scaler. Every classifier receives identical rows.
    train_ids, test_ids = train_test_split(
        np.arange(len(dataset)), test_size=0.2, stratify=dataset[TARGET], random_state=SEED
    )
    training = dataset.iloc[train_ids].reset_index(drop=True)
    testing = dataset.iloc[test_ids].reset_index(drop=True)
    training.to_csv(ROOT / "data" / "train_data.csv", index=False, float_format="%.12g")
    testing.to_csv(ROOT / "test_data.csv", index=False, float_format="%.12g")
    # Evaluate the exact CSV values that an evaluator will upload to the app.
    testing = validate_test_data(pd.read_csv(ROOT / "test_data.csv"))
    comparison, details_by_model = [], {}
    predictions = pd.DataFrame({"actual_fault": testing[TARGET]})
    print("Steel Plates Faults - ML Assignment 2")
    print(f"Python {platform.python_version()} | scikit-learn {sklearn.__version__}")
    print(f"Dataset: {len(dataset)} rows, {len(FEATURES)} features, {len(CLASSES)} classes")
    print(f"Stratified split: train={len(training)}, test={len(testing)}, seed={SEED}")
    print("Precision / Recall / F1: macro across seven classes; AUC: macro OvR")
    for name, pipeline in build_models().items():
        pipeline.fit(training[FEATURES], training[TARGET])
        model_path = ROOT / "model" / MODEL_FILES[name]
        joblib.dump(pipeline, model_path, compress=3)
        # Report metrics from reloaded artifacts, the same route used by the app.
        restored = joblib.load(model_path)
        scores, details, predicted, _ = evaluate_model(restored, testing)
        comparison.append({"ML Model Name": name, **scores})
        details_by_model[name] = details
        predictions[name] = predicted
    table = pd.DataFrame(comparison)
    table.to_csv(ROOT / "results" / "metrics.csv", index=False, float_format="%.12g")
    predictions.to_csv(ROOT / "results" / "predictions.csv", index=False)
    (ROOT / "results" / "evaluation_details.json").write_text(
        json.dumps({"labels": CLASSES, "models": details_by_model}, indent=2), encoding="utf-8"
    )
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    winner = table.sort_values(["F1", "MCC", "Accuracy"], ascending=False).iloc[0]
    print(f"\nHighest held-out macro F1: {winner['ML Model Name']} ({winner['F1']:.4f})")
    print("This execution does not by itself prove BITS Virtual Lab usage.")
    model_hashes = {
        filename: hashlib.sha256((ROOT / "model" / filename).read_bytes()).hexdigest()
        for filename in MODEL_FILES.values()
    }
    metadata = {
        "dataset": "UCI Steel Plates Faults", "source_url": SOURCE_URL,
        "source_doi": "10.24432/C5J88N", "license": "CC BY 4.0",
        "features": FEATURES, "target": TARGET, "classes": CLASSES,
        "seed": SEED, "test_fraction": 0.2, "training_rows": len(training),
        "test_rows": len(testing), "train_row_indices": train_ids.tolist(),
        "test_row_indices": test_ids.tolist(), "model_files": MODEL_FILES,
        "model_sha256": model_hashes,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "test_csv_sha256": hashlib.sha256((ROOT / "test_data.csv").read_bytes()).hexdigest(),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metric_average": "macro", "auc": "macro one-vs-rest probability AUC",
        "hyperparameters_selected_using_test_data": False,
    }
    (ROOT / "model" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), constrained_layout=True)
    short_names = ["Logistic\nRegression", "Decision\nTree", "KNN", "Naive\nBayes", "SVM"]
    for axis, metric in zip(axes.flat, METRICS):
        values = table[metric].to_numpy()
        axis.bar(short_names, values, color=["#167d8d"] * 4 + ["#dc9149"])
        axis.set_title(metric if metric not in ["F1", "Recall", "Precision"] else f"{metric} (macro)")
        axis.set_ylim(-0.05 if metric == "MCC" else 0, 1.08)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        for index, value in enumerate(values):
            axis.text(index, value + 0.016, f"{value:.3f}", ha="center", fontsize=8)
        axis.tick_params(axis="x", labelsize=8)
    figure.suptitle(f"Steel Plates Faults | {len(testing)} held-out rows", fontsize=14)
    figure.savefig(ROOT / "results" / "model_comparison.png", dpi=160)
    plt.close(figure)
    print("\nSaved five pipelines, test CSV, comparison chart and evaluation outputs.")


if __name__ == "__main__":
    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, value):
            for stream in self.streams:
                stream.write(value)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    import sys
    (ROOT / "results").mkdir(exist_ok=True)
    with (ROOT / "results" / "training_output.txt").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)):
            main()
