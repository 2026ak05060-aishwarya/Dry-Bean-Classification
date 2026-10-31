
# Dry Bean Classification

## Project Overview

This project focuses on multiclass classification of dry bean varieties using machine learning algorithms.

The objective is to classify dry beans into one of seven categories based on their physical measurements. Five machine learning algorithms are implemented, trained, and evaluated using different performance metrics.

A Streamlit web application is developed to allow users to upload test data, select a trained model, and view its classification performance.

## Dataset Description

- **Dataset:** Dry Bean Dataset
- **File:** `Dry_Bean_Dataset.xlsx`
- **Original records:** 13,611
- **Features:** 16 numerical features
- **Target variable:** Class
- **Number of classes:** 7

The dataset contains physical measurements of dry beans, such as area, perimeter, major axis length, minor axis length, aspect ratio, eccentricity, and other shape-related features.

The seven bean varieties are:

- DERMASON
- SIRA
- SEKER
- HOROZ
- CALI
- BARBUNYA
- BOMBAY

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset from the Excel file using Pandas.
2. Checked for missing values and duplicate records.
3. Removed duplicate records.
4. Separated the input features and target variable.
5. Divided the dataset into training and testing sets using an 80:20 split.
6. Applied StandardScaler to the features for Logistic Regression, KNN, and SVM.

The train-test split was performed using stratified sampling with `random_state=42`.

## Machine Learning Algorithms

The following algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Model Evaluation

The models were evaluated using the following performance metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)

### Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9192 | 0.9934 | 0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree | 0.8966 | 0.9363 | 0.8965 | 0.8966 | 0.8964 | 0.8750 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Gaussian Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| SVM | 0.9210 | 0.9933 | 0.9214 | 0.9210 | 0.9211 | 0.9045 |

## Observations

- **Logistic Regression:** Achieved good classification performance with an accuracy of approximately 91.92%. It also obtained the highest AUC score.
- **Decision Tree:** Provided an accuracy of approximately 89.66%. Its performance was comparatively lower than the other models.
- **KNN:** Achieved approximately 91.55% accuracy, showing good classification performance.
- **Gaussian Naive Bayes:** Achieved approximately 76.30% accuracy, the lowest among the five models.
- **SVM:** Achieved the highest accuracy of approximately 92.10%, along with the highest precision, recall, F1-score, and MCC.

## Best Performing Model

Based on the evaluation results, **Support Vector Machine (SVM)** performed best overall, achieving the highest accuracy of approximately 92.10%.

Logistic Regression obtained a slightly higher AUC score. However, SVM performed better across the other reported evaluation metrics.

## Project Structure

```text
Dry-Bean-Classification/
│
├── README.md
├── app.py
├── requirements.txt
├── data_preprocessing.py
│
├── test_data.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── scaler.joblib
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Dry-Bean-Classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

### 4. Use the application

1. Upload the test dataset in CSV format.
2. Select one of the five trained machine learning models.
3. View the evaluation metrics.
4. View the confusion matrix and classification report.

The application uses the previously trained and saved models from the `model/` directory. It does not retrain the models during application execution.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib
- Google Colab
