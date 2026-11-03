
import sys
import joblib

from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_preprocessing import (
    X_train,
    y_train,
    MODEL_DIR
)

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(
    model,
    MODEL_DIR / "decision_tree.joblib",
    compress=3
)

print("Decision Tree trained and saved successfully.")
