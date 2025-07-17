import pandas as pd
from pathlib import Path
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import StringSerializer
from confluent_kafka import SerializingProducer
import math

def clean_record(record):
    for k, v in record.items():
        if isinstance(v, float) and math.isnan(v):
            record[k] = None
        elif isinstance(v, str) and v.strip().lower() in {"nan", "", "na"}:
            record[k] = None
        elif pd.isna(v):
            record[k] = None
    return record

def get_avro_producer(avro_name):
    base_dir = Path(__file__).resolve().parent.parent.parent
    avro_dir = base_dir / "src" / "avro_schemas"
    with open(avro_dir / avro_name, "r") as f:
        schema_str = f.read()
    schema_registry_conf = {
        "url": "http://ad52692b7d0ae41cabfdcf8f99816daf-463135936.us-east-1.elb.amazonaws.com:8081"
    }
    schema_registry_client = SchemaRegistryClient(schema_registry_conf)
    avro_serializer = AvroSerializer(
        schema_registry_client,
        schema_str,
    )
    # Producer config
    producer_conf = {
        "bootstrap.servers": "af8d07cc8fe7b410dad5212f956d90cf-1672909144.us-east-1.elb.amazonaws.com:9094",
        "key.serializer": StringSerializer("utf_8"),
        "value.serializer": avro_serializer,
        "queue.buffering.max.messages": 100000,
        "queue.buffering.max.kbytes": 1048576,
        "message.timeout.ms": 120000,
        "request.timeout.ms": 60000,
        "retries": 5,
        "retry.backoff.ms": 1000,
    }
    return SerializingProducer(producer_conf)

def produce_user():
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    producer = get_avro_producer("user.avsc")
    df = pd.read_csv(data_dir / "user_cleaned.csv")
    for _, row in df.iterrows():
        record = clean_record(row.to_dict())
        producer.produce(topic="user-topic", key=str(record.get("idx")), value=record)
    producer.flush()
    print(f"Produced {len(df)} user records.")

def produce_card():
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    producer = get_avro_producer("card.avsc")
    df = pd.read_csv(data_dir / "card_cleaned.csv")
    for _, row in df.iterrows():
        record = clean_record(row.to_dict())
        producer.produce(topic="card-topic", key=str(record.get("user")), value=record)
    producer.flush()
    print(f"Produced {len(df)} card records.")

def produce_transaction(chunksize=10000):
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    producer = get_avro_producer("transaction.avsc")
    total = 0
    for chunk in pd.read_csv(data_dir / "transaction_cleaned.csv", chunksize=chunksize):
        for _, row in chunk.iterrows():
            record = clean_record(row.to_dict())
            delivered = False
            while not delivered:
                try:
                    producer.produce(
                        topic="transaction-topic", key=str(record.get("user")), value=record
                    )
                    producer.poll(0)
                    delivered = True
                except BufferError:
                    print("Queue full, waiting...")
                    producer.poll(1)
        producer.flush()
        total += len(chunk)
        print(f"Produced {total} transaction records so far...")

if __name__ == "__main__":
    produce_transaction()
    produce_user()
    produce_card()
