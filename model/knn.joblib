
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

os.makedirs("model", exist_ok=True)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predictions
y_pred = knn.predict(X_test_scaled)
y_prob = knn.predict_proba(X_test_scaled)

# Evaluation
print("K-NEAREST NEIGHBORS")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("AUC      :", roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted", labels=knn.classes_))
print("Precision:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, average="weighted", zero_division=0))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted", zero_division=0))
print("MCC      :", matthews_corrcoef(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred, labels=knn.classes_),
            annot=True, fmt="d", cmap="Oranges",
            xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.title("KNN - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save model
joblib.dump(knn, "model/knn.joblib")
