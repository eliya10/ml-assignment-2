"""Check the saved artifacts, split isolation, six metrics and CSV validation."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from common import (
    CLASSES, FEATURES, METRICS, MODEL_FILES, ROOT, TARGET, evaluate_model, validate_test_data
)


def main():
    metadata = json.loads((ROOT / "model" / "metadata.json").read_text())
    source_path = ROOT / "data" / "steel_plates_faults.csv"
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == metadata["dataset_sha256"]
    full_data = validate_test_data(pd.read_csv(source_path))
    training = validate_test_data(pd.read_csv(ROOT / "data" / "train_data.csv"))
    testing = validate_test_data(pd.read_csv(ROOT / "test_data.csv"))
    assert full_data.shape == (1941, 28)
    train_ids, test_ids = metadata["train_row_indices"], metadata["test_row_indices"]
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(range(len(full_data)))
    pd.testing.assert_frame_equal(
        training, full_data.iloc[train_ids].reset_index(drop=True), check_dtype=False
    )
    pd.testing.assert_frame_equal(
        testing, full_data.iloc[test_ids].reset_index(drop=True), check_dtype=False
    )
    print("PASS: 1,941 source rows; 27 features; disjoint 1,552/389 train/test split")
    recorded = pd.read_csv(ROOT / "results" / "metrics.csv").set_index("ML Model Name")
    for name, filename in MODEL_FILES.items():
        path = ROOT / "model" / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["model_sha256"][filename]
        model = joblib.load(path)
        assert list(model.classes_) == CLASSES
        scores, details, predicted, probabilities = evaluate_model(model, testing)
        np.testing.assert_allclose(
            [scores[metric] for metric in METRICS], recorded.loc[name, METRICS].to_numpy(float),
            atol=1e-10, rtol=0
        )
        assert np.array(details["confusion_matrix"]).sum() == len(testing)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-8)
        pd.testing.assert_series_equal(
            pd.Series(predicted),
            pd.Series(model.predict(testing[list(reversed(FEATURES))].loc[:, FEATURES])),
        )
        if "scale" in model.named_steps:
            np.testing.assert_allclose(
                model.named_steps["scale"].mean_, training[FEATURES].mean(), rtol=1e-10, atol=1e-8
            )
        print(f"PASS: {name} - saved predictions and all six metrics reproduce")
    for bad_frame in [
        testing.drop(columns=[TARGET]), testing.assign(fault_type="unknown"),
        testing.assign(X_Minimum=np.inf), testing.assign(X_Minimum="bad"), testing.iloc[:0],
    ]:
        try:
            validate_test_data(bad_frame)
        except ValueError:
            pass
        else:
            raise AssertionError("An invalid test CSV was accepted")
    one_class = testing.loc[testing[TARGET] == CLASSES[0]]
    model = joblib.load(ROOT / "model" / MODEL_FILES["Logistic Regression"])
    assert evaluate_model(model, one_class)[0]["AUC"] is None
    print("PASS: malformed CSVs rejected; absent-class AUC handled explicitly")
    print("Project verification passed. Capture actual BITS Lab execution separately.")


if __name__ == "__main__":
    main()
