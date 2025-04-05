import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load the government mock data
df = pd.read_csv("mock_government_records.csv")

# Create feature: hash of Aadhaar + PAN
df["Combined"] = df["Aadhaar"].astype(str) + df["pan"].astype(str)
df["HashFeature"] = df["Combined"].apply(hash)

# Train Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(df[["HashFeature"]])

# Save the model
joblib.dump(model, "fraud_model.pkl")
print("✅ Model trained and saved successfully.")
