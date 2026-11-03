
import sys
import joblib

from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_preprocessing import (
    X_train_scaled,
    y_train,
    MODEL_DIR
)

model = KNeighborsClassifier(
    n_neighbors=5
)

model.fit(X_train_scaled, y_train)

joblib.dump(
    model,
    MODEL_DIR / "knn.joblib",
    compress=3
)

print("KNN trained and saved successfully.")
