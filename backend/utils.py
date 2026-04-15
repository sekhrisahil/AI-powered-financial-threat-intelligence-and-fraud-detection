import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "fraud_model.pkl")

class FraudModel:
    def __init__(self):
        bundle = joblib.load(MODEL_PATH)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.features = bundle["features"]

    def predict_one(self, row_dict: dict):
        import pandas as pd
        df_row = pd.DataFrame([row_dict], columns=self.features)

        df_row_scaled = df_row.copy()
        df_row_scaled[["Time", "Amount"]] = self.scaler.transform(
            df_row[["Time", "Amount"]]
        )

        proba = self.model.predict_proba(df_row_scaled)[0][1]
        pred = int(proba >= 0.5)
        return pred, float(proba)

    def predict_batch(self, rows: list[dict]):
        import pandas as pd
        df_batch = pd.DataFrame(rows, columns=self.features)

        df_batch_scaled = df_batch.copy()
        df_batch_scaled[["Time", "Amount"]] = self.scaler.transform(
            df_batch[["Time", "Amount"]]
        )

        probas = self.model.predict_proba(df_batch_scaled)[:, 1]
        preds = (probas >= 0.5).astype(int)
        return preds.tolist(), probas.tolist()

fraud_model = FraudModel()