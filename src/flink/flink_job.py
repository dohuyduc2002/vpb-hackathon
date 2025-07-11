import os
import requests
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common.watermark_strategy import WatermarkStrategy
from pyflink.datastream.connectors import KafkaSource
from pyflink.common.serialization import ConfluentRegistryAvroDeserializationSchema

BOOTSTRAP_SERVERS = os.getenv("BOOTSTRAP_SERVERS", "localhost:9092")
SCHEMA_REGISTRY_URL = os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")

env = StreamExecutionEnvironment.get_execution_environment()


def build_kafka_source(topic, group_id, schema_registry_url):
    schema_subject = f"{topic}-value"
    avro_schema = ConfluentRegistryAvroDeserializationSchema(
        subject=schema_subject, schema_registry_url=schema_registry_url
    )
    return (
        KafkaSource.builder()
        .set_bootstrap_servers(BOOTSTRAP_SERVERS)
        .set_topics(topic)
        .set_group_id(group_id)
        .set_value_only_deserializer(avro_schema)
        .build()
    )


transaction_source = build_kafka_source(
    "transaction-topic", "flink-ds-transaction-group", SCHEMA_REGISTRY_URL
)

# Đọc vào DataStream
transactions = env.from_source(
    transaction_source, WatermarkStrategy.no_watermarks(), "Transactions"
)


def call_model_api(transaction):
    data = {
        "user": transaction.get("user"),
        "card": transaction.get("card"),
    }
    try:
        resp = requests.post("http://localhost:8000/predict", json=data, timeout=3)
        pred = resp.json().get("prediction")
        transaction["prediction"] = pred
        print("Predicted:", transaction)
    except Exception as e:
        print("API call failed:", e)


transactions.map(call_model_api)

env.execute("Transaction topic -> Model API predict")
