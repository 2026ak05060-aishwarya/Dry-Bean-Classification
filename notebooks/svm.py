
import sys
import joblib

from pathlib import Path
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_preprocessing import (
    X_train_scaled,
    y_train,
    MODEL_DIR
)

model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

model.fit(X_train_scaled, y_train)

joblib.dump(
    model,
    MODEL_DIR / "svm.joblib",
    compress=3
)

print("SVM trained and saved successfully.")
