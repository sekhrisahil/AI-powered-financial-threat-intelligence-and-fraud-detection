from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import Transaction, BatchRequest, PredictionResponse
from backend.utils import fraud_model

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Detects potential fraudulent credit card transactions",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud API running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(txn: Transaction):
    pred, proba = fraud_model.predict_one(txn.dict())
    return {"is_fraud": pred, "probability": proba}

@app.post("/predict-batch")
def predict_fraud_batch(batch: BatchRequest):
    preds, probas = fraud_model.predict_batch([t.dict() for t in batch.transactions])
    return {"is_fraud": preds, "probability": probas}