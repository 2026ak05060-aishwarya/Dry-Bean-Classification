
import os
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

# Load dataset
data = pd.read_excel(BASE_DIR / "Dry_Bean_Dataset.xlsx")

# Remove duplicates
data = data.drop_duplicates()

# Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model directory
MODEL_DIR.mkdir(exist_ok=True)

# Save scaler
joblib.dump(scaler, MODEL_DIR / "scaler.joblib", compress=3)

# Save test dataset
test_data = X_test.copy()
test_data["Class"] = y_test

test_data.to_csv(BASE_DIR / "test_data.csv", index=False)

print("Data preprocessing completed successfully.")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Features:", X.shape[1])
print("Classes:", y.nunique())
