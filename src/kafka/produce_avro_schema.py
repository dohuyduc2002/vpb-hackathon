import pandas as pd
from pathlib import Path
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import StringSerializer
from confluent_kafka import SerializingProducer


def produce_to_kafka(csv_name, avro_name, topic, chunksize=None):
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    avro_dir = base_dir / "src" / "avro_schemas"

    with open(avro_dir / avro_name, "r") as f:
        schema_str = f.read()

    # Schema Registry config
    schema_registry_conf = {"url": "http://34.172.236.120:8081"}
    schema_registry_client = SchemaRegistryClient(schema_registry_conf)
    avro_serializer = AvroSerializer(
        schema_registry_client,
        schema_str,
    )

    # Producer config
    producer_conf = {
        "bootstrap.servers": "35.192.41.150:9094",
        "key.serializer": StringSerializer("utf_8"),
        "value.serializer": avro_serializer,
        "queue.buffering.max.messages": 100000,
        "queue.buffering.max.kbytes": 1048576,
        "message.timeout.ms": 120000,
        "request.timeout.ms": 60000,
        "retries": 5,
        "retry.backoff.ms": 1000,
    }
    producer = SerializingProducer(producer_conf)

    if chunksize:  
        total = 0
        for chunk in pd.read_csv(data_dir / csv_name, chunksize=chunksize):
            for _, row in chunk.iterrows():
                record = row.to_dict()
                for k, v in record.items():
                    if pd.isna(v):
                        record[k] = None
                delivered = False
                while not delivered:
                    try:
                        producer.produce(
                            topic=topic, key=str(record.get("user")), value=record
                        )
                        producer.poll(0)
                        delivered = True
                    except BufferError:
                        print("Queue full, waiting...")
                        producer.poll(1)
                    except Exception as e:
                        print(f"Error: {e}")
                        break
            producer.flush()
            total += len(chunk)
            print(f"Produced {total} records so far...")
    else:  
        df = pd.read_csv(data_dir / csv_name)
        for _, row in df.iterrows():
            record = row.to_dict()
            for k, v in record.items():
                if pd.isna(v):
                    record[k] = None
                producer.produce(topic=topic, key=str(record.get("User")), value=record)
        producer.flush()
        print(f"Produced {len(df)} records.")


if __name__ == "__main__":
    produce_to_kafka(
        csv_name="transaction_cleaned.csv",
        avro_name="transaction.avsc",
        topic="transaction-topic",
        chunksize=10000,  
    )

    produce_to_kafka(
        csv_name="user_cleaned.csv", avro_name="user.avsc", topic="user-topic"
    )

    produce_to_kafka(
        csv_name="card_cleaned.csv", avro_name="card.avsc", topic="card-topic"
    )
