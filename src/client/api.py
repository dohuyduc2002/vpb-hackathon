import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import traceback

from train_simple_hybrid import (
    SimplifiedHybridFraudDetector,
    load_and_preprocess_encoded_csv,
)

MODEL_DIR = os.environ.get("MODEL_DIR", "./simple_hybrid_model")
try:
    model = SimplifiedHybridFraudDetector()
    model.load_model(MODEL_DIR)
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

app = FastAPI(title="Hybrid Fraud Detection API")


def categorize_risk_level(probability):
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"


def analyze_risk_factors(predictions, features, feature_names, top_n=10):
    high_risk_indices = np.where(predictions > 0.5)[0]
    result = []
    for idx in high_risk_indices[:top_n]:
        factors = []
        f = features.iloc[idx]
        if f.get("is_night", 0) == 1:
            factors.append("Transaction at unusual hours (night)")
        if f.get("is_weekend", 0) == 1:
            factors.append("Weekend transaction")
        if f.get("high_amount", 0) == 1:
            factors.append("High transaction amount")
        if f.get("user_fraud_risk", 0) > 0.1:
            factors.append(f"User has fraud history ({f['user_fraud_risk']:.1%})")
        if f.get("merchant_fraud_risk", 0) > 0.1:
            factors.append(
                f"Merchant has fraud history ({f['merchant_fraud_risk']:.1%})"
            )
        if f.get("amount_vs_user_avg", 0) > 3:
            factors.append("Amount much higher than user average")
        if f.get("amount_vs_merchant_avg", 0) > 3:
            factors.append("Amount much higher than merchant average")
        result.append(
            {
                "transaction_id": int(idx),
                "fraud_probability": float(predictions[idx]),
                "risk_factors": factors,
            }
        )
    return result


from io import StringIO


@app.post("/infer/csv")
async def infer_csv(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
    try:
        content = await file.read()
        file_stream = StringIO(content.decode("utf-8"))
        df = pd.read_csv(file_stream)

        # Chuẩn hóa Amount
        if "Amount" in df.columns:
            df["Amount"] = df["Amount"].replace("[\$,]", "", regex=True).astype(float)

        # Map 'Is Fraud?' sang 0/1
        if "Is Fraud?" in df.columns:
            df["Is Fraud?"] = df["Is Fraud?"].map({"Yes": 1, "No": 0})
            df["Is Fraud?"] = df["Is Fraud?"].fillna(0)

        # Convert "Time" (HH:MM) sang "Hour" (int)
        if "Time" in df.columns and "Hour" not in df.columns:
            df["Hour"] = df["Time"].astype(str).str.split(":").str[0].astype(int)

        # Nếu pipeline yêu cầu cột "Is_Fraud" thay cho "Is Fraud?"
        if "Is Fraud?" in df.columns and "Is_Fraud" not in df.columns:
            df = df.rename(columns={"Is Fraud?": "Is_Fraud"})

        # Tiếp tục pipeline
        enhanced_df = model.create_graph_features(df)
        features = model.prepare_features(enhanced_df)
        fraud_prob, binary_pred = model.predict(df)
        df["fraud_probability"] = fraud_prob
        df["risk_level"] = [categorize_risk_level(p) for p in fraud_prob]
        total = len(df)
        n_high = int((df["risk_level"] == "HIGH").sum())
        n_medium = int((df["risk_level"] == "MEDIUM").sum())
        n_low = int((df["risk_level"] == "LOW").sum())
        risk_analysis = analyze_risk_factors(
            fraud_prob, features, model.feature_names, top_n=10
        )
        return {
            "total_transactions": total,
            "high_risk": n_high,
            "medium_risk": n_medium,
            "low_risk": n_low,
            "top_10_high_risk_analysis": risk_analysis,
            "predictions": df[["fraud_probability", "risk_level"]].to_dict(
                orient="records"
            ),
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
