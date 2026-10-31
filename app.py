import joblib

# Save the fitted scaler
scaler = joblib.load("model/scaler.joblib")

# Save held-out test data
test_data = X_test.copy()
test_data["Class"] = y_test

test_data.to_csv("test_data.csv", index=False)

print("Scaler and test dataset saved successfully.")
