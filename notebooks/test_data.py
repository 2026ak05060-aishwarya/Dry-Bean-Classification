import pandas as pd
from sklearn.model_selection import train_test_split

# Load Excel dataset
df = pd.read_excel("Dry_Bean_Dataset.xlsx")

# Remove duplicate records
df = df.drop_duplicates()

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Combine test features and target
test_data = X_test.copy()
test_data["Class"] = y_test

# Save as CSV
test_data.to_csv("test_data.csv", index=False)

print("Test data saved successfully!")
print(test_data.shape)
