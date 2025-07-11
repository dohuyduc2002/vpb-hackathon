import os
from minio import Minio
import io
from fastavro import reader
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import openai

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "bronze-layer")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "transaction_embedding")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_avro_files_from_minio():
    """Tìm tất cả file avro từ các partition."""
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    avro_files = []
    # List partition folders
    objects = minio_client.list_objects(
        MINIO_BUCKET, prefix="topics/transaction-topic/", recursive=True
    )
    for obj in objects:
        if obj.object_name.endswith(".avro"):
            avro_files.append(obj.object_name)
    return avro_files


def read_avro_from_minio(minio_client, bucket, avro_path):
    """Trả về list records từ file avro trên MinIO."""
    response = minio_client.get_object(bucket, avro_path)
    data = io.BytesIO(response.read())
    response.close()
    response.release_conn()
    records = []
    for record in reader(data):
        records.append(record)
    return records


def get_embedding_from_openai(text):
    openai.api_key = OPENAI_API_KEY
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response["data"][0]["embedding"]


def ensure_milvus_collection(dim):
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    if COLLECTION_NAME in [c.name for c in Collection.list()]:
        collection = Collection(COLLECTION_NAME)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="raw_text", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(
            fields, description="Transaction embedding collection"
        )
        collection = Collection(COLLECTION_NAME, schema)
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

    dim = 1536  
    collection = ensure_milvus_collection(dim)

    # Với mỗi file avro
    for avro_path in avro_files:
        records = read_avro_from_minio(minio_client, MINIO_BUCKET, avro_path)
        print(f"{avro_path}: {len(records)} records")
        embeddings, texts = [], []
        for rec in records:
            text = rec.get("content") or rec.get("message")
            if text and isinstance(text, str) and len(text.strip()) > 0:
                embedding = get_embedding_from_openai(text)
                embeddings.append(embedding)
                texts.append(text)
        if embeddings:
            entities = [embeddings, texts]
            collection.insert(entities)
            print(f"Inserted {len(embeddings)} records from {avro_path}")
        else:
            print(f"No valid text in {avro_path}")


if __name__ == "__main__":
    index_avro_minio_to_milvus()
