
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# Page configuration
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("Dry Bean Classification")
st.write("Multiclass classification using five machine learning algorithms.")


# Load trained models and scaler
@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.joblib"),
        "Decision Tree": joblib.load("model/decision_tree.joblib"),
        "KNN": joblib.load("model/knn.joblib"),
        "Gaussian Naive Bayes": joblib.load("model/gaussian_naive_bayes.joblib"),
        "SVM": joblib.load("model/svm.joblib")
    }

    scaler = joblib.load("model/scaler.joblib")

    return models, scaler


models, scaler = load_models()


# Expected feature columns
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


# Upload test dataset
st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing the features and Class column",
    type=["csv"]
)


if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(data.head())

    if "Class" not in data.columns:
        st.error("The uploaded CSV must contain a 'Class' column.")

    elif not all(col in data.columns for col in feature_columns):
        st.error("The uploaded CSV is missing one or more required features.")

    else:

        X = data[feature_columns]
        y = data["Class"]

        selected_model = st.selectbox(
            "Select Machine Learning Model",
            list(models.keys())
        )

        if st.button("Evaluate Model"):

            model = models[selected_model]

            # Apply scaling only to models that require it
            if selected_model in [
                "Logistic Regression",
                "KNN",
                "SVM"
            ]:
                X_input = scaler.transform(X)
            else:
                X_input = X

            # Predictions
            y_pred = model.predict(X_input)
            y_prob = model.predict_proba(X_input)

            # Evaluation metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(
                y, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(
                y, y_pred, average="weighted", zero_division=0
            )
            f1 = f1_score(
                y, y_pred, average="weighted", zero_division=0
            )
            mcc = matthews_corrcoef(y, y_pred)

            try:
                auc = roc_auc_score(
                    y,
                    y_prob,
                    labels=model.classes_,
                    multi_class="ovr",
                    average="weighted"
                )
            except ValueError:
                auc = None

            # Display results
            st.subheader("Evaluation Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("Precision", f"{precision:.4f}")
            col3.metric("Recall", f"{recall:.4f}")

            col4, col5, col6 = st.columns(3)

            col4.metric("F1-Score", f"{f1:.4f}")
            col5.metric(
                "AUC",
                f"{auc:.4f}" if auc is not None else "Unavailable"
            )
            col6.metric("MCC", f"{mcc:.4f}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y,
                y_pred,
                labels=model.classes_
            )

            fig, ax = plt.subplots(figsize=(9, 6))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                ax=ax
            )

            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")

            st.pyplot(fig)
            plt.close(fig)

            # Classification Report
            st.subheader("Classification Report")

            report = classification_report(
                y,
                y_pred,
                zero_division=0
            )

            st.text(report)
