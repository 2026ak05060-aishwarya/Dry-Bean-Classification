
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

os.makedirs("model", exist_ok=True)

# Train model
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predictions
y_pred = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)

# Evaluation
print("GAUSSIAN NAIVE BAYES")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("AUC      :", roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted", labels=nb.classes_))
print("Precision:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, average="weighted", zero_division=0))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted", zero_division=0))
print("MCC      :", matthews_corrcoef(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred, labels=nb.classes_),
            annot=True, fmt="d", cmap="Purples",
            xticklabels=nb.classes_, yticklabels=nb.classes_)
plt.title("Naive Bayes - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save model
joblib.dump(nb, "model/naive_bayes.joblib")
