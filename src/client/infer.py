import torch
from torch_geometric.nn import GAE
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from input_preprocess import preprocess_single
from model.gat_encoder import GATEncoderWithEdgeAttrs
from model.edge_classifier import EdgeMLPClassifier
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

import os
from dotenv import load_dotenv
from kafka_schema import send_to_kafka, getidx

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL = "#fraud-alerts"

slack_client = WebClient(token=SLACK_BOT_TOKEN)


def notify_slack(message: str):
    try:
        slack_client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
    except SlackApiError as e:
        print(f"Slack error: {e.response['error']}")


app = FastAPI()


class FraudDetectionInput(BaseModel):
    user: str
    card: str
    year: int
    month: int
    day: int
    time: str
    amount: str
    use_chip: Optional[str] = None
    merchant_name: Optional[int] = None
    merchant_city: Optional[str] = None
    merchant_state: Optional[str] = None
    zip: Optional[float] = None
    mcc: int
    errors: Optional[str] = None
    is_fraud: Optional[str] = None


# Load models and preprocess objects
ckpt = torch.load("fraud_gnn_model.pt", map_location="cpu", weights_only=False)
node_mapping = ckpt["node_mapping"]
scaler = ckpt["scaler"]
label_encoders = ckpt["label_encoders"]

encoder = GATEncoderWithEdgeAttrs(
    in_channels=len(node_mapping), hidden_channels=64, edge_attr_dim=7
)
encoder.load_state_dict(ckpt["encoder_state_dict"])
model = GAE(encoder)

classifier = EdgeMLPClassifier(emb_dim=64, hidden_dim=64, num_classes=2)
classifier.load_state_dict(ckpt["classifier_state_dict"])

encoder.eval()
classifier.eval()


@app.post("/predict")
def predict(input: FraudDetectionInput):
    sample = input.dict()
    edge_index, edge_attr = preprocess_single(
        sample, node_mapping, scaler, label_encoders
    )
    x = torch.eye(len(node_mapping))
    with torch.no_grad():
        z, processed_edge_attr = encoder(x, edge_index, edge_attr)
        out = classifier(z, edge_index, processed_edge_attr)
        pred = torch.argmax(out, dim=1).item()
    result = "Yes" if pred == 1 else "No"

    avro_record = {
        "idx" : getidx(),
        "user": str(sample.get("user")),
        "card": str(sample.get("card")),
        "year": int(sample.get("year")),
        "month": int(sample.get("month")),
        "day": int(sample.get("day")),
        "time": sample.get("time"),
        "amount": sample.get("amount"),
        "use_chip": sample.get("use_chip"),
        "merchant_name": sample.get("merchant_name"),
        "merchant_city": sample.get("merchant_city"),
        "merchant_state": sample.get("merchant_state"),
        "zip": sample.get("zip"),
        "mcc": int(sample.get("mcc")),
        "errors": sample.get("errors"),
        "is_fraud": result,
    }

    send_to_kafka(avro_record)

    if result == "Yes":
        msg = (
            f"*FRAUD DETECTED!*\n"
            f"User: {avro_record['user']}\n"
            f"Card: {avro_record['card']}\n"
            f"Amount: {avro_record['amount']}\n"
            f"Time: {avro_record['year']}-{avro_record['month']:02}-{avro_record['day']:02} {avro_record['time']}\n"
            f"Merchant: {avro_record.get('merchant_name', '')}"
        )
        notify_slack(msg)

    return {"prediction": result, "probabilities": out.softmax(dim=1).tolist()}


@app.get("/")
def root():
    return {"status": "ok"}
