"""PlateScope: evaluate saved steel-fault classifiers on labelled test CSVs."""

import csv
import io
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from common import (
    CLASSES, FEATURES, METRICS, MODEL_FILES, ROOT, SOURCE_URL, TARGET,
    evaluate_model, validate_test_data,
)

st.set_page_config(page_title="PlateScope | Steel Faults", page_icon="🔎", layout="wide")
st.markdown("""
<style>
.block-container {max-width:1200px; padding-top:2.1rem;}
[data-testid="stMetric"] {border:1px solid #c5dadc; border-radius:12px; padding:16px;}
.eyebrow {color:#137885; letter-spacing:0.15em; font-size:0.8rem; font-weight:700;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">INDUSTRIAL QUALITY / MACHINE LEARNING LAB</div>', unsafe_allow_html=True)
st.title("PlateScope")
st.write("Compare five classifiers for seven types of steel plate surface faults.")
st.caption("Evaluate labelled test data using saved models and preprocessing pipelines.")


@st.cache_resource
def load_models():
    # Load only the project's own model files; uploaded files are CSV data only.
    with (ROOT / "model" / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata["features"] != FEATURES or metadata["classes"] != CLASSES:
        raise ValueError("Saved model metadata does not match this application's schema.")
    return {
        name: joblib.load(ROOT / "model" / filename)
        for name, filename in MODEL_FILES.items()
    }


@st.cache_data(show_spinner=False)
def score_csv(csv_bytes):
    """Validation and prediction only. No model is fitted on uploaded data."""
    text = csv_bytes.decode("utf-8-sig")
    header = next(csv.reader(io.StringIO(text)), [])
    if len(header) != len(set(header)):
        raise ValueError("The CSV contains duplicate column names.")
    frame = pd.read_csv(io.StringIO(text))
    if len(frame) > 5000:
        raise ValueError("Use test data only, with at most 5,000 rows per upload.")
    frame = validate_test_data(frame)
    results = {}
    for name, model in load_models().items():
        scores, details, predicted, probabilities = evaluate_model(model, frame)
        results[name] = {
            "metrics": scores, "details": details,
            "predicted": predicted, "probabilities": probabilities,
        }
    return frame, results


with st.sidebar:
    st.header("Test bench")
    selected_model = st.selectbox("Classifier", list(MODEL_FILES))
    st.write("Upload a labelled test CSV")
    uploaded_csv = st.file_uploader(
        "CSV test data", type=["csv"],
        help="All 27 input features and the fault_type target column are required.",
    )
    use_bundled = st.checkbox("Use the bundled held-out test data", value=True)
    st.download_button(
        "Download test_data.csv", (ROOT / "test_data.csv").read_bytes(),
        file_name="test_data.csv", mime="text/csv",
    )
    st.caption("An uploaded CSV takes priority over the bundled data. Maximum: 5,000 rows / 5 MB.")
    with st.expander("CSV schema and class labels"):
        st.write("Required input columns:")
        st.code("\n".join(FEATURES), language=None)
        st.write(f"Target column: {TARGET}")
        st.write(", ".join(CLASSES))
        st.caption("K_Scatch is the spelling used in the original UCI dataset.")
    st.link_button("UCI dataset source", SOURCE_URL)

if uploaded_csv is not None:
    if uploaded_csv.size > 5 * 1024 * 1024:
        st.error("The CSV exceeds 5 MB. Upload a smaller labelled test set.")
        st.stop()
    csv_bytes = uploaded_csv.getvalue()
    source_label = "Uploaded test CSV"
elif use_bundled:
    csv_bytes = (ROOT / "test_data.csv").read_bytes()
    source_label = "Bundled held-out test CSV"
else:
    st.info("Upload a labelled test CSV or enable the bundled held-out test data.")
    st.stop()

try:
    with st.spinner("Evaluating the five saved models..."):
        data, evaluations = score_csv(csv_bytes)
except (ValueError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
    st.error(f"CSV validation failed: {exc}")
    st.stop()
except (FileNotFoundError, ImportError, AttributeError, KeyError) as exc:
    st.error(f"Saved models are unavailable or incompatible: {exc}")
    st.info("Install requirements.txt and run python train.py in the project folder.")
    st.stop()

st.caption(f"{source_label}  |  {len(data):,} rows  |  {len(FEATURES)} features  |  {data[TARGET].nunique()} observed classes")
st.subheader(selected_model)
chosen = evaluations[selected_model]
for column, label in zip(st.columns(6), METRICS):
    value = chosen["metrics"][label]
    column.metric(label, "N/A" if value is None else f"{value:.4f}")
st.caption("Precision, recall and F1 use macro averaging across all seven fault classes. AUC uses macro one-vs-rest probabilities.")
if chosen["metrics"]["AUC"] is None:
    missing_classes = sorted(set(CLASSES) - set(data[TARGET]))
    st.warning(
        "AUC is undefined because these classes are absent: " + ", ".join(missing_classes)
        + ". Macro precision/recall/F1 still use all seven classes; undefined per-class values contribute zero."
    )

compare_tab, errors_tab, data_tab = st.tabs(["Compare models", "Inspect errors", "Test data & predictions"])
with compare_tab:
    comparison = pd.DataFrame([
        {"ML Model Name": name, **result["metrics"]} for name, result in evaluations.items()
    ]).set_index("ML Model Name")
    st.dataframe(comparison.style.format("{:.4f}", na_rep="N/A"), width="stretch")
    st.bar_chart(comparison[["Accuracy", "F1"]], color=["#167d8d", "#dc9149"], height=280)
    leader = comparison.sort_values(["F1", "MCC", "Accuracy"], ascending=False).index[0]
    st.write(f"Highest macro F1 on this test CSV: **{leader}**.")
    st.caption("This ranking describes the supplied test rows; it is not a claim of universal model superiority.")
with errors_tab:
    left, right = st.columns([1.1, 1])
    with left:
        matrix = np.array(chosen["details"]["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
        axis.imshow(matrix, cmap="Blues")
        for row in range(len(CLASSES)):
            for col in range(len(CLASSES)):
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center",
                          color="white" if matrix[row, col] > matrix.max() / 2 else "#17343b")
        axis.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right", fontsize=9)
        axis.set_yticks(range(len(CLASSES)), CLASSES, fontsize=9)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_title("Confusion matrix")
        st.pyplot(figure)
        plt.close(figure)
    with right:
        st.write("**Classification report**")
        report = chosen["details"]["classification_report"]
        per_class = pd.DataFrame({label: report[label] for label in CLASSES}).T
        st.dataframe(per_class.style.format({
            "precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"
        }), width="stretch")
with data_tab:
    predictions = data.copy()
    predictions["predicted_fault"] = chosen["predicted"]
    predictions["correct"] = data[TARGET].to_numpy() == chosen["predicted"]
    st.dataframe(predictions, width="stretch", height=350)
    st.download_button(
        "Download selected-model predictions", predictions.to_csv(index=False).encode("utf-8"),
        file_name="fault_predictions.csv", mime="text/csv",
    )
