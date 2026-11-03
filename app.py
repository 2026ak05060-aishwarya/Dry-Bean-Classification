
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

TARGET = "Class"

FEATURES = [
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

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

SCALED_MODELS = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Support Vector Machine"
]


# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 3. CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,.25);
}

.hero {
    padding: 1.35rem 1.5rem;
    border-radius: 18px;
    background: linear-gradient(115deg,#174734,#47865c);
    color: white;
    margin-bottom: 1rem;
}

.hero h1 {
    font-size: 2rem;
    margin: 0 0 .35rem 0;
    color: white;
}

.hero p {
    font-size: 1rem;
    margin: 0;
    color: #e6f4e9;
}

.metric-card {
    border: 1px solid rgba(128,128,128,.28);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    min-height: 110px;
}

.metric-label {
    font-size: .85rem;
    opacity: .8;
    margin-bottom: .35rem;
}

.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    line-height: 1.2;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. METRIC CARD
# --------------------------------------------------

def metric_card(label, value):

    shown = (
        "N/A"
        if value is None or pd.isna(value)
        else f"{value:.4f}"
    )

    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{shown}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# 5. AVAILABLE MODELS
# --------------------------------------------------

def load_available_models():

    return {
        name: MODEL_DIR / filename
        for name, filename in MODEL_FILES.items()
        if (MODEL_DIR / filename).is_file()
    }


# --------------------------------------------------
# 6. LOAD SELECTED MODEL AND SCALER
# --------------------------------------------------

@st.cache_resource
def load_selected_model(
    model_name,
    model_path,
    model_mtime,
    scaler_path,
    scaler_mtime
):

    model = joblib.load(model_path)

    scaler = None

    if model_name in SCALED_MODELS:

        scaler = joblib.load(scaler_path)

    return model, scaler


# --------------------------------------------------
# 7. EVALUATE MODEL
# --------------------------------------------------

def evaluate_one(name, model_path, X, y_true):

    scaler_path = MODEL_DIR / "scaler.joblib"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path.name}"
        )

    if name in SCALED_MODELS and not scaler_path.is_file():
        raise FileNotFoundError(
            "scaler.joblib not found in model folder."
        )

    model_mtime = model_path.stat().st_mtime_ns

    scaler_mtime = (
        scaler_path.stat().st_mtime_ns
        if scaler_path.exists()
        else 0
    )

    model, scaler = load_selected_model(
        name,
        str(model_path),
        model_mtime,
        str(scaler_path),
        scaler_mtime
    )

    # Preserve the original feature order
    X_model = X[FEATURES].copy()

    # Apply scaling only to the models trained with scaled data
    if scaler is not None:

        X_processed = scaler.transform(X_model)

    else:

        X_processed = X_model

    # Predictions
    y_pred = model.predict(X_processed)

    # Evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_true, y_pred)

    # Multiclass AUC - One vs Rest
    try:

        probabilities = model.predict_proba(X_processed)

        auc = roc_auc_score(
            y_true,
            probabilities,
            labels=model.classes_,
            multi_class="ovr",
            average="weighted"
        )

    except (ValueError, AttributeError):
        auc = np.nan

    metrics = {
        "Model": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    }

    return model, y_pred, metrics, X_model


# --------------------------------------------------
# 8. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🌱 Dry Bean")
    st.caption("Dry Bean Classification · ML Assignment")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Evaluate models",
            "Compare models",
            "Dataset & methodology"
        ]
    )

    st.divider()

    st.markdown("**Test dataset**")

    uploaded = st.file_uploader(
        "Upload a labelled test CSV",
        type=["csv"],
        help=(
            "CSV must contain all 16 input features "
            "and the target column Class."
        )
    )

    st.caption("Expected target column: Class")

    st.divider()

    st.caption(
        "Saved models are loaded from the model/ folder. "
        "Uploaded data is used only for evaluation."
    )


# --------------------------------------------------
# 9. HEADER
# --------------------------------------------------

st.markdown(
    '<div class="hero">'
    '<h1>Dry Bean Classification</h1>'
    '<p>Evaluate and compare five machine-learning '
    'classifiers for Dry Bean variety prediction.</p>'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# 10. CHECK MODEL FILES
# --------------------------------------------------

available_models = load_available_models()

if not available_models:

    st.error(
        "No trained model files were found in the model/ folder."
    )

    st.stop()

missing_models = [
    name
    for name in MODEL_FILES
    if name not in available_models
]

if missing_models:

    st.warning(
        "Some model files are missing: "
        + ", ".join(missing_models)
    )


# --------------------------------------------------
# 11. LOAD TEST DATASET
# --------------------------------------------------

if uploaded is not None:

    try:

        test_df = pd.read_csv(
            uploaded,
            keep_default_na=False
        )

        data_source = f"Uploaded file: {uploaded.name}"

    except Exception as exc:

        st.error(f"Could not read uploaded CSV: {exc}")
        st.stop()

else:

    bundled_test = ROOT / "test_data.csv"

    if bundled_test.exists():

        try:

            test_df = pd.read_csv(
                bundled_test,
                keep_default_na=False
            )

            data_source = "Bundled test_data.csv"

        except Exception as exc:

            st.error(f"Could not read test_data.csv: {exc}")
            st.stop()

    else:

        test_df = None
        data_source = "No test data loaded"


# --------------------------------------------------
# 12. OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    c1, c2, c3 = st.columns(3)

    c1.metric("Algorithms", "5", "Classification models")
    c2.metric("Input features", "16", "Numerical features")
    c3.metric("Target classes", "7", "Bean varieties")

    st.subheader("Project at a glance")

    left, right = st.columns([1.3, 1])

    with left:

        st.markdown("""
        **Objective**

        Predict the variety of Dry Beans using numerical
        characteristics and compare five machine-learning
        classification algorithms.

        **Models included**

        - Logistic Regression
        - Decision Tree Classifier
        - K-Nearest Neighbors
        - Gaussian Naive Bayes
        - Support Vector Machine
        """)

    with right:

        st.markdown("### Evaluation workflow")

        st.markdown("""
        1. Train and save models.
        2. Upload labelled test data.
        3. Select a model or compare all five.
        4. Review performance metrics.
        5. Examine the confusion matrix and classification report.
        """)

        st.info(
            "The application uses previously trained models. "
            "It does not retrain models using uploaded data."
        )

    st.subheader("Current test data")

    st.write(data_source)

    if test_df is None:

        st.warning(
            "No test CSV is available. Add test_data.csv "
            "to the project root or upload a CSV."
        )

    else:

        st.write(
            f"**{len(test_df):,} rows · "
            f"{len(test_df.columns)} columns**"
        )

        st.dataframe(
            test_df.head(8),
            use_container_width=True
        )


# --------------------------------------------------
# 13. DATASET AND METHODOLOGY PAGE
# --------------------------------------------------

elif page == "Dataset & methodology":

    st.subheader("Dataset and experimental setup")

    st.markdown("""
    **Dataset:** Dry Bean Dataset.

    **Target:** `Class`, representing one of seven Dry Bean varieties.

    **Input features:** 16 numerical characteristics of the beans.

    **Data preprocessing:**
    - Duplicate records removed.
    - Features and target separated.
    - Stratified 80:20 train-test split.
    - Random state set to 42.
    - StandardScaler applied to Logistic Regression,
      KNN and SVM.
    - Decision Tree and Gaussian Naive Bayes use
      unscaled features.

    **Model parameters:**
    - Logistic Regression: max_iter=2000.
    - Decision Tree: random_state=42.
    - KNN: n_neighbors=5.
    - Gaussian Naive Bayes: default parameters.
    - SVM: RBF kernel, probability=True.

    **Evaluation metrics:**
    - Accuracy
    - Multiclass ROC AUC (One-vs-Rest)
    - Weighted Precision
    - Weighted Recall
    - Weighted F1-score
    - Matthews Correlation Coefficient (MCC)
    """)

    st.info(
        "The scaler and trained models are loaded from "
        "the saved files in the model/ folder."
    )

    st.warning(
        "Uploaded data is used only for evaluating "
        "the saved models. No retraining takes place."
    )


# --------------------------------------------------
# 14. EVALUATION AND COMPARISON PAGES
# --------------------------------------------------

elif page in ("Evaluate models", "Compare models"):

    if test_df is None:

        st.warning(
            "Upload a labelled test CSV using the sidebar "
            "or add test_data.csv to the project root."
        )

        st.stop()

    # Validate target
    if TARGET not in test_df.columns:

        st.error(
            "The CSV must contain the target column 'Class'."
        )

        st.stop()

    # Validate features
    missing_features = [
        col for col in FEATURES
        if col not in test_df.columns
    ]

    if missing_features:

        st.error(
            "Missing required feature columns: "
            + ", ".join(missing_features)
        )

        st.stop()

    # Extract target
    y_true = test_df[TARGET].astype(str).str.strip()

    if y_true.eq("").any():

        st.error("The Class column contains empty values.")
        st.stop()

    # Extract features
    X = test_df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if X.isnull().values.any():

        st.error(
            "Feature columns contain missing or non-numeric values."
        )

        st.stop()

    if not np.isfinite(X.to_numpy()).all():

        st.error(
            "Feature columns contain infinite values."
        )

        st.stop()

    st.caption(
        f"Data source: {data_source} · "
        f"{len(test_df):,} labelled rows"
    )


    # --------------------------------------------------
    # 15. EVALUATE ONE MODEL
    # --------------------------------------------------

    if page == "Evaluate models":

        model_name = st.selectbox(
            "Choose a classifier",
            list(available_models.keys())
        )

        try:

            model, y_pred, metrics, X_used = evaluate_one(
                model_name,
                available_models[model_name],
                X,
                y_true
            )

        except Exception as exc:

            st.error(
                f"Could not evaluate {model_name}: {exc}"
            )

            st.stop()

        st.subheader(
            f"{model_name} — Evaluation Summary"
        )

        # Metric cards
        metric_cols = st.columns(6)

        metric_names = [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC"
        ]

        for col, label in zip(metric_cols, metric_names):

            with col:

                metric_card(
                    label,
                    metrics[label]
                )

        st.divider()

        # Tabs
        tab1, tab2, tab3 = st.tabs([
            "Confusion Matrix",
            "Classification Report",
            "Test Data Preview"
        ])

        # Confusion Matrix
        with tab1:

            class_labels = model.classes_

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=class_labels
            )

            fig, ax = plt.subplots(figsize=(9, 6))

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=class_labels
            )

            disp.plot(
                ax=ax,
                colorbar=False,
                cmap="Blues",
                values_format="d",
                xticks_rotation=45
            )

            ax.set_title(model_name)
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        # Classification Report
        with tab2:

            report = classification_report(
                y_true,
                y_pred,
                labels=model.classes_,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).T

            st.dataframe(
                report_df.style.format("{:.4f}"),
                use_container_width=True
            )

        # Test Data Preview
        with tab3:

            preview = X_used.copy()

            preview["Actual Class"] = y_true.values
            preview["Predicted Class"] = y_pred

            st.dataframe(
                preview.head(100),
                use_container_width=True
            )

            st.download_button(
                "Download predictions CSV",
                preview.to_csv(index=False),
                "dry_bean_predictions.csv",
                "text/csv"
            )


    # --------------------------------------------------
    # 16. COMPARE ALL MODELS
    # --------------------------------------------------

    else:

        st.subheader("Compare All Classification Models")

        st.write(
            "All available models are evaluated on the same "
            "test dataset. No model is retrained in this app."
        )

        metric_rows = []
        failures = []

        for name, model_path in available_models.items():

            try:

                _, _, metrics, _ = evaluate_one(
                    name,
                    model_path,
                    X,
                    y_true
                )

                metric_rows.append(metrics)

            except Exception as exc:

                failures.append(
                    f"{name}: {exc}"
                )

        if failures:

            for failure in failures:

                st.warning(failure)

        if not metric_rows:

            st.error(
                "No model could be evaluated. "
                "Check your model files and test dataset."
            )

            st.stop()

        results = (
            pd.DataFrame(metric_rows)
            .set_index("Model")
        )

        st.dataframe(
            results.style.format("{:.4f}").highlight_max(
                axis=0,
                props="font-weight: bold;"
            ),
            use_container_width=True
        )

        st.subheader("Metric Comparison")

        selected_metrics = st.multiselect(
            "Metrics to plot",
            [
                "Accuracy",
                "AUC",
                "Precision",
                "Recall",
                "F1",
                "MCC"
            ],
            default=[
                "Accuracy",
                "AUC",
                "F1",
                "MCC"
            ]
        )

        if selected_metrics:

            st.bar_chart(
                results[selected_metrics],
                use_container_width=True
            )

        st.download_button(
            "Download comparison CSV",
            results.reset_index().to_csv(index=False),
            "dry_bean_model_comparison.csv",
            "text/csv"
        )

        st.info(
            "Select a model under 'Evaluate models' to inspect "
            "its confusion matrix and classification report."
        )


# --------------------------------------------------
# 17. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Dry Bean Classification | Machine Learning Assignment"
)
