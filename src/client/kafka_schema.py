from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import StringSerializer
from confluent_kafka import SerializingProducer
import random

# "bootstrap.servers": "kafka-cluster-0-kafka-bootstrap:9092",
# "schema.registry.url": "http://schema-registry-svc:8081",
KAFKA_TOPIC = "transaction-topic"
_IDX_MIN = 26_000_001
_IDX_MAX = 10**9 
used_idx_set = set()
TRANSACTION_AVRO_SCHEMA_STR = """
{
  "type": "record",
  "namespace": "com.example",
  "name": "Transaction",
  "fields": [
    {"name": "idx", "type": "int"},
    {"name": "user", "type": "string"},
    {"name": "card", "type": "string"},
    {"name": "year", "type": "int"},
    {"name": "month", "type": "int"},
    {"name": "day", "type": "int"},
    {"name": "time", "type": "string"},
    {"name": "amount", "type": "string"},
    {"name": "use_chip", "type": ["null", "string"], "default": null},
    {"name": "merchant_name", "type": ["null", "long"], "default": null},
    {"name": "merchant_city", "type": ["null", "string"], "default": null},
    {"name": "merchant_state", "type": ["null", "string"], "default": null},
    {"name": "zip", "type": ["null", "double"], "default": null},
    {"name": "mcc", "type": "int"},
    {"name": "errors", "type": ["null", "string"], "default": null},
    {"name": "is_fraud", "type": "string"}
  ]
}
"""

schema_registry_conf = {"url": "http://schema-registry-svc:8081"}
schema_registry_client = SchemaRegistryClient(schema_registry_conf)


def transaction_to_dict(obj, ctx):
    return obj


avro_serializer = AvroSerializer(
    schema_registry_client, TRANSACTION_AVRO_SCHEMA_STR, transaction_to_dict
)

producer_config = {
    "bootstrap.servers": "kafka-cluster-0-kafka-bootstrap:9092",
    "key.serializer": StringSerializer("utf_8"),
    "value.serializer": avro_serializer,
    "queue.buffering.max.messages": 100000,
    "queue.buffering.max.kbytes": 1048576,
    "message.timeout.ms": 120000,
    "request.timeout.ms": 60000,
    "retries": 5,
    "retry.backoff.ms": 1000,
}

producer = SerializingProducer(producer_config)

def send_to_kafka(record: dict):
    producer.produce(topic=KAFKA_TOPIC, key=record.get("user"), value=record)
    producer.flush()


def getidx():
    while True:
        idx = random.randint(_IDX_MIN, _IDX_MAX)
        if idx not in used_idx_set:
            used_idx_set.add(idx)
            return idx
