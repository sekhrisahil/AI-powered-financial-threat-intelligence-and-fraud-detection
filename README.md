# Credit Card Fraud Detection App

## 1. Create venv + install
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Train model
```powershell
python train_model.py
```

This reads data/creditcard.csv and creates models/fraud_model.pkl

## 3. Run backend (FastAPI)
```powershell
uvicorn backend.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs

## 4. Run dashboard (Streamlit) in a second terminal
```powershell
streamlit run frontend/app.py --server.port 8501
```

Open http://127.0.0.1:8501
