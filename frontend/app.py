import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import os

st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    layout="wide"
)

API_BASE = "http://127.0.0.1:8000"

st.title("💳 Credit Card Fraud Detection Dashboard")
st.caption("Real-time fraud scoring using Machine Learning (Random Forest)")

st.header("1. Dataset Overview")

DATA_PATH = os.path.join("data", "creditcard.csv")
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    fraud_count = int((df["Class"] == 1).sum()) if "Class" in df.columns else 0
    legit_count = int((df["Class"] == 0).sum()) if "Class" in df.columns else 0
    total = len(df)

    fraud_pct = (fraud_count / total) * 100 if total > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{total}")
    c2.metric("Fraud Cases", f"{fraud_count}")
    c3.metric("Fraud %", f"{fraud_pct:.4f}%")

    if "Class" in df.columns:
        status_df = pd.DataFrame({
            "Class": ["Legit", "Fraud"],
            "Count": [legit_count, fraud_count]
        })
        fig_bar = px.bar(status_df, x="Class", y="Count", title="Fraud vs Legit Count")
        st.plotly_chart(fig_bar, use_container_width=True)

    if "Amount" in df.columns and "Class" in df.columns:
        st.subheader("Transaction Amount Distribution (Legit vs Fraud)")
        df_sample = df.sample(n=min(5000, len(df)), random_state=42)
        fig_violin = px.violin(
            df_sample,
            x="Class",
            y="Amount",
            box=True,
            points="outliers",
            color="Class",
            title="Amount by Class (0=Legit, 1=Fraud)"
        )
        st.plotly_chart(fig_violin, use_container_width=True)
else:
    st.warning("Dataset not found at data/creditcard.csv. Skipping overview section.")

st.header("2. Test Single Transaction")

with st.form("single_txn_form"):
    st.write("Enter transaction features:")

    time_val = st.number_input("Time", value=10000.0)
    amount_val = st.number_input("Amount", value=50.0)

    v_cols = {}
    for v in [f"V{i}" for i in range(1, 29)]:
        v_cols[v] = st.number_input(v, value=0.0, step=0.01)

    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    txn_payload = {
        "Time": time_val,
        "Amount": amount_val,
        **v_cols
    }

    try:
        resp = requests.post(f"{API_BASE}/predict", json=txn_payload, timeout=10)
        if resp.status_code == 200:
            out = resp.json()
            st.success(f"Prediction: {'FRAUD' if out['is_fraud']==1 else 'LEGIT'}")
            st.metric("Fraud Probability", f"{out['probability']:.4f}")
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

st.header("3. Bulk Scoring (Upload CSV)")

uploaded_file = st.file_uploader(
    "Upload CSV with columns: Time,V1,...,V28,Amount (NO Class required)",
    type=["csv"]
)

if uploaded_file is not None:
    new_df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(new_df.head())

    batch_payload = {"transactions": new_df.to_dict(orient="records")}

    try:
        resp = requests.post(f"{API_BASE}/predict-batch", json=batch_payload, timeout=20)
        if resp.status_code == 200:
            result = resp.json()
            new_df["is_fraud_pred"] = result["is_fraud"]
            new_df["fraud_probability"] = result["probability"]

            st.subheader("Scored Results")
            st.dataframe(new_df.head(50))

            pred_counts = new_df["is_fraud_pred"].value_counts().rename({0: "Legit", 1: "Fraud"})
            pred_df = pd.DataFrame({
                "Prediction": pred_counts.index,
                "Count": pred_counts.values
            })
            fig_pred = px.bar(
                pred_df, x="Prediction", y="Count",
                title="Predicted Fraud vs Legit (Uploaded File)"
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            csv_out = new_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Scored CSV",
                data=csv_out,
                file_name="scored_transactions.csv",
                mime="text/csv",
            )
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")