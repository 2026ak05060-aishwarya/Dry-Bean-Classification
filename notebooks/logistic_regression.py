
import sys
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_preprocessing import (
    X_train_scaled,
    y_train,
    MODEL_DIR
)

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

joblib.dump(
    model,
    MODEL_DIR / "logistic_regression.joblib",
    compress=3
)

print("Logistic Regression trained and saved successfully.")
