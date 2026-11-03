
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("Dry Bean Classification")
st.write(
    "Classification of Dry Bean varieties using "
    "Machine Learning algorithms."
)

# --------------------------------------------------
# 2. DATASET FEATURES
# --------------------------------------------------

feature_columns = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]

# --------------------------------------------------
# 3. MODEL FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "SVM": "svm.joblib"
}

scaled_models = [
    "Logistic Regression",
    "KNN",
    "SVM"
]

# --------------------------------------------------
# 4. LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(model_name):

    model_path = MODEL_DIR / model_files[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(model_path)

    scaler = None

    if model_name in scaled_models:

        scaler_path = MODEL_DIR / "scaler.joblib"

        if not scaler_path.exists():
            raise FileNotFoundError(
                "Scaler file not found."
            )

        scaler = joblib.load(scaler_path)

    return model, scaler


# --------------------------------------------------
# 5. UPLOAD TEST DATA
# --------------------------------------------------

st.sidebar.header("Upload Test Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

    except Exception as e:
        st.error(f"Unable to read the CSV file: {e}")
        st.stop()

else:

    default_file = BASE_DIR / "test_data.csv"

    if default_file.exists():
        data = pd.read_csv(default_file)
        st.sidebar.info("Using the default test dataset.")

    else:
        st.warning("Please upload your test_data.csv file.")
        st.stop()

# --------------------------------------------------
# 6. DISPLAY DATASET
# --------------------------------------------------

st.subheader("Dataset Overview")

col1, col2 = st.columns(2)

col1.metric("Total Records", len(data))
col2.metric("Total Features", len(feature_columns))

st.dataframe(data.head())

# --------------------------------------------------
# 7. DATA VALIDATION
# --------------------------------------------------

required_columns = feature_columns + ["Class"]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:

    st.error(
        f"Missing columns in dataset: {missing_columns}"
    )
    st.stop()

X = data[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

y = data["Class"]

if X.isnull().values.any():

    st.error("Missing or invalid feature values found.")
    st.stop()

if y.isnull().any():

    st.error("Missing values found in the Class column.")
    st.stop()

if not np.isfinite(X.to_numpy()).all():

    st.error("Infinite values found in the dataset.")
    st.stop()

# --------------------------------------------------
# 8. MODEL SELECTION
# --------------------------------------------------

st.sidebar.header("Select Model")

selected_model = st.sidebar.selectbox(
    "Choose a classification algorithm",
    list(model_files.keys())
)

st.subheader(selected_model)

# --------------------------------------------------
# 9. MODEL EVALUATION
# --------------------------------------------------

if st.button("Run Evaluation"):

    try:

        with st.spinner("Loading model and evaluating..."):

            model, scaler = load_model(selected_model)

            # Apply scaling where required
            if scaler is not None:
                X_processed = scaler.transform(X)
            else:
                X_processed = X

            # Predictions
            predictions = model.predict(X_processed)

            # Evaluation metrics
            accuracy = accuracy_score(y, predictions)

            precision = precision_score(
                y,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y,
                predictions,
                average="weighted",
                zero_division=0
            )

            mcc = matthews_corrcoef(y, predictions)

            # Multiclass AUC
            try:

                probabilities = model.predict_proba(
                    X_processed
                )

                auc = roc_auc_score(
                    y,
                    probabilities,
                    labels=model.classes_,
                    multi_class="ovr",
                    average="weighted"
                )

            except (ValueError, AttributeError):
                auc = None

        # --------------------------------------------------
        # 10. DISPLAY METRICS
        # --------------------------------------------------

        st.subheader("Performance Metrics")

        c1, c2, c3 = st.columns(3)

        c1.metric("Accuracy", f"{accuracy:.4f}")

        c2.metric(
            "AUC",
            "N/A" if auc is None else f"{auc:.4f}"
        )

        c3.metric("Precision", f"{precision:.4f}")

        c4, c5, c6 = st.columns(3)

        c4.metric("Recall", f"{recall:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC", f"{mcc:.4f}")

        # --------------------------------------------------
        # 11. PREDICTION RESULTS
        # --------------------------------------------------

        st.subheader("Prediction Results")

        results = pd.DataFrame({
            "Actual Class": y,
            "Predicted Class": predictions
        })

        st.dataframe(results)

        # --------------------------------------------------
        # 12. CONFUSION MATRIX
        # --------------------------------------------------

        st.subheader("Confusion Matrix")

        class_labels = model.classes_

        cm = confusion_matrix(
            y,
            predictions,
            labels=class_labels
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # 13. CLASSIFICATION REPORT
        # --------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y,
            predictions,
            zero_division=0
        )

        st.text(report)

    except Exception as e:

        st.error(
            f"Error while evaluating {selected_model}: {e}"
        )

else:

    st.info("Click 'Run Evaluation' to evaluate the selected model.")

# --------------------------------------------------
# 14. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Dry Bean Classification | Machine Learning Assignment"
)
