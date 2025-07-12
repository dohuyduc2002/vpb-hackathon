import os
from fastapi import FastAPI
from client.schema import RawItem
import pickle
import pandas as pd
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import StringSerializer
from confluent_kafka import SerializingProducer
import json

app = FastAPI()

# ----- LOAD CONFIG FROM ENV -----
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "fraud_predict")
SCHEMA_REGISTRY_URL = os.environ.get(
    "SCHEMA_REGISTRY_URL", "http://schema-registry:8081"
)
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL", "#fraud-alerts")

# ----- LOAD ARTIFACTS -----
with open("client/preprocess.pkl", "rb") as f:
    preprocess = pickle.load(f)
with open("client/model.pkl", "rb") as f:
    model = pickle.load(f)

# ----- SLACK CLIENT -----
slack_client = WebClient(token=SLACK_BOT_TOKEN)


def notify_slack(payload: dict):
    try:
        message = (
            "🚨 *Fraud detected!*\n"
            f"```{json.dumps(payload, indent=2, ensure_ascii=False)}```"
        )
        slack_client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
    except SlackApiError as e:
        print(f"Slack notify error: {e.response['error']}")


# ----- AVRO SCHEMA -----
avro_schema_str = """
{
  "namespace": "fraud.predict",
  "name": "Prediction",
  "type": "record",
  "fields": [
    {"name": "user", "type": "string"},
    {"name": "card", "type": "string"},
    {"name": "year", "type": "string"},
    {"name": "month", "type": "string"},
    {"name": "day", "type": "string"},
    {"name": "time", "type": "string"},
    {"name": "amount", "type": "string"},
    {"name": "use_chip", "type": ["null", "string"], "default": null},
    {"name": "merchant_name", "type": ["null", "string"], "default": null},
    {"name": "merchant_city", "type": ["null", "string"], "default": null},
    {"name": "merchant_state", "type": ["null", "string"], "default": null},
    {"name": "zip", "type": ["null", "string"], "default": null},
    {"name": "mcc", "type": "string"},
    {"name": "errors", "type": ["null", "string"], "default": null},
    {"name": "prediction", "type": "string"}
  ]
}
"""

# ----- KAFKA PRODUCER SETUP -----
schema_registry_conf = {"url": SCHEMA_REGISTRY_URL}
schema_registry_client = SchemaRegistryClient(schema_registry_conf)

avro_serializer = AvroSerializer(
    schema_registry_client=schema_registry_client,
    schema_str=avro_schema_str,
    to_dict=lambda obj, ctx: obj,
)

string_serializer = StringSerializer("utf_8")

producer_conf = {
    "bootstrap.servers": KAFKA_BROKER,
    "key.serializer": string_serializer,
    "value.serializer": avro_serializer,
    "schema.registry.url": SCHEMA_REGISTRY_URL,
}
producer = SerializingProducer(producer_conf)


def send_to_kafka(payload: dict):
    try:
        user_key = payload.get("user")
        if not user_key:
            raise ValueError("Field 'user' must not be empty, required as Kafka key.")
        producer.produce(topic=KAFKA_TOPIC, key=user_key, value=payload)
        producer.flush()
    except Exception as e:
        print(f"Kafka Avro produce error: {e}")


@app.post("/predict")
def predict(data: RawItem):
    df = pd.DataFrame([data.dict()])
    X = preprocess.transform(df)
    pred_label = model.predict(X)[0]
    pred = "fraud" if pred_label == 1 else "not_fraud"
    sink_payload = data.dict()
    sink_payload["prediction"] = pred

    if pred == "fraud":
        notify_slack(sink_payload)

    try:
        send_to_kafka(sink_payload)
    except Exception as ex:
        # Log error, không fail API, chỉ cảnh báo
        print(f"Failed to produce to Kafka: {ex}")

    return {"prediction": pred}
