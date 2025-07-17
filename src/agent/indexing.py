import os
from minio import Minio
import io
from fastavro import reader
from pymilvus import connections, Collection
import openai
import time

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "bronze-layer")

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "transaction_embedding")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BATCH_SIZE = 100  # Số lượng record mỗi batch, có thể chỉnh lên 200, 500, 1000 tùy bạn


def get_avro_files_from_minio():
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    avro_files = []
    objects = minio_client.list_objects(
        MINIO_BUCKET, prefix="topics/transaction-topic/", recursive=True
    )
    for obj in objects:
        if obj.object_name.endswith(".avro"):
            avro_files.append(obj.object_name)
    return avro_files


def read_avro_from_minio(minio_client, bucket, avro_path):
    response = minio_client.get_object(bucket, avro_path)
    data = io.BytesIO(response.read())
    response.close()
    response.release_conn()
    records = []
    for record in reader(data):
        records.append(record)
    return records


def get_embeddings_from_openai(texts):
    openai.api_key = OPENAI_API_KEY
    try:
        response = openai.Embedding.create(input=texts, model="text-embedding-3-small")
        return [item["embedding"] for item in response["data"]]
    except Exception as e:
        print(f"Error from OpenAI API: {e}")
        # Trả về vector 0 nếu fail, tránh crash script
        return [[0.0] * 1536 for _ in texts]  


def connect_to_milvus_collection():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    if COLLECTION_NAME not in [c.name for c in Collection.list()]:
        raise Exception(
            f"Collection '{COLLECTION_NAME}' does not exist! Please create it first."
        )
    collection = Collection(COLLECTION_NAME)
    return collection


def index_avro_minio_to_milvus():
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    avro_files = get_avro_files_from_minio()
    print(f"Found {len(avro_files)} Avro files.")

    collection = connect_to_milvus_collection()
    fields = [f.name for f in collection.schema.fields if not f.auto_id]
    print(f"Insert fields: {fields}")

    for avro_path in avro_files:
        records = read_avro_from_minio(minio_client, MINIO_BUCKET, avro_path)
        print(f"{avro_path}: {len(records)} records")

        texts = []
        for rec in records:
            text = rec.get("content") or rec.get("message")
            if text and isinstance(text, str) and len(text.strip()) > 0:
                texts.append(text)

        num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            print(f"Processing batch {i//BATCH_SIZE + 1}/{num_batches} ...", end=" ")
            t0 = time.time()
            embeddings = get_embeddings_from_openai(batch_texts)
            entities = []
            if "embedding" in fields:
                entities.append(embeddings)
            if "raw_text" in fields:
                entities.append(batch_texts)
            collection.insert(entities)
            t1 = time.time()
            print(f"Inserted {len(batch_texts)} records in {t1-t0:.1f}s")
        print(f"Done for file: {avro_path}")


if __name__ == "__main__":
    index_avro_minio_to_milvus()
