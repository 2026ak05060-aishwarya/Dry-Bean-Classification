
from sklearn.svm import SVC
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

os.makedirs("model", exist_ok=True)

# Train model
svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)

# Predictions
y_pred = svm.predict(X_test_scaled)
y_prob = svm.predict_proba(X_test_scaled)

# Evaluation
print("SUPPORT VECTOR MACHINE")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("AUC      :", roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted", labels=svm.classes_))
print("Precision:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, average="weighted", zero_division=0))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted", zero_division=0))
print("MCC      :", matthews_corrcoef(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred, labels=svm.classes_),
            annot=True, fmt="d", cmap="Reds",
            xticklabels=svm.classes_, yticklabels=svm.classes_)
plt.title("SVM - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save model
joblib.dump(svm, "model/svm.joblib")
