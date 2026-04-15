import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = os.path.join("data", "creditcard.csv")
df = pd.read_csv(r"C:\Users\digit\OneDrive\Documents\sahil shukla\creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

fraud_idx = np.where(y == 1)[0]
normal_idx = np.where(y == 0)[0]

np.random.seed(42)
normal_sample_idx = np.random.choice(normal_idx, size=len(fraud_idx), replace=False)
undersample_idx = np.concatenate([fraud_idx, normal_sample_idx])

X_bal = X_scaled.iloc[undersample_idx]
y_bal = y.iloc[undersample_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight=None
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

os.makedirs("models", exist_ok=True)
joblib.dump(
    {"model": model, "scaler": scaler, "features": list(X.columns)},
    os.path.join("models", "fraud_model.pkl")
)

print("\nSaved trained model to models/fraud_model.pkl")