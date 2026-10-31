
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Excel Dataset
df = pd.read_excel("Dry_Bean_Dataset.xlsx")

print("DATASET OVERVIEW")
print("---------------------------")
print("Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nDataset Information:")
df.info()
print("\nStatistical Summary:\n", df.describe())

# 2. Check Missing Values and Duplicates
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

print("\nShape After Cleaning:", df.shape)

# 3. Class Distribution
print("\nClass Distribution:\n", df["Class"].value_counts())

plt.figure(figsize=(10, 5))
sns.countplot(
    data=df,
    x="Class",
    order=df["Class"].value_counts().index
)
plt.title("Distribution of Dry Bean Classes")
plt.xlabel("Bean Class")
plt.ylabel("Number of Instances")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 4. Separate Features and Target
X = df.drop(columns=["Class"])
y = df["Class"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# 5. Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# 6. Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed")

# 7. Save Test Dataset as CSV for Streamlit
test_data = X_test.copy()
test_data["Class"] = y_test

test_data.to_csv("test_data.csv", index=False)

print("\nTest Dataset Saved Successfully")
print("Test Dataset Shape:", test_data.shape)
print("\nDataset Ready for Model Training!")
