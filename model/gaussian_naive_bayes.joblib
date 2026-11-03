
import sys
import joblib

from pathlib import Path
from sklearn.naive_bayes import GaussianNB

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_preprocessing import (
    X_train,
    y_train,
    MODEL_DIR
)

model = GaussianNB()

model.fit(X_train, y_train)

joblib.dump(
    model,
    MODEL_DIR / "gaussian_naive_bayes.joblib",
    compress=3
)

print("Gaussian Naive Bayes trained and saved successfully.")
