from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import json
from minio import Minio
import pandas as pd
import chromadb
from chromadb.config import Settings
import boto3
from datetime import datetime, timedelta

# --- Config ---
MINIO_ENDPOINT = "minio.minio.svc.cluster.local:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"
BUCKET = "stream-bucket"
MINIO_PREFIX = "etl/intermediate/"
CHROMA_COLLECTION = "fraud_titanv2_indexing"
LOCAL_PARQUET_DIR = "/tmp/parquet_files"
CHROMA_HOST = "chroma-chromadb.chroma.svc.cluster.local"
CHROMA_PORT = 8000

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BEDROCK_REGION = "us-east-1"
BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v2:0"

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def get_titan_v2_embedding(text):
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=BEDROCK_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    body = json.dumps({"inputText": text})
    response = client.invoke_model(
        modelId=BEDROCK_EMBED_MODEL,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    embedding = json.loads(response["body"].read())["embedding"]
    return embedding  # 1024 dimension


def download_parquet_files(**kwargs):
    if not os.path.exists(LOCAL_PARQUET_DIR):
        os.makedirs(LOCAL_PARQUET_DIR, exist_ok=True)
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    objects = minio_client.list_objects(BUCKET, prefix=MINIO_PREFIX, recursive=True)
    downloaded_files = []
    for obj in objects:
        if obj.object_name.endswith(".parquet"):
            data = minio_client.get_object(BUCKET, obj.object_name).read()
            fname = obj.object_name.split("/")[-1]
            local_path = os.path.join(LOCAL_PARQUET_DIR, fname)
            with open(local_path, "wb") as f:
                f.write(data)
            downloaded_files.append(local_path)
    kwargs["ti"].xcom_push(key="downloaded_files", value=downloaded_files)


def index_parquet_files_as_documents(**kwargs):
    downloaded_files = kwargs["ti"].xcom_pull(
        key="downloaded_files", task_ids="download_parquet_files"
    )
    if not downloaded_files:
        return

    # Connect to ChromaDB
    chroma_client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(allow_reset=True),
    )
    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

    indexed_files = 0
    for fpath in downloaded_files:
        fname = os.path.basename(fpath)
        table_name = fname.split("_")[0]
        df = pd.read_parquet(fpath)
        file_text = df.to_csv(index=False)
        text_to_embed = file_text[:1500]
        embedding = get_titan_v2_embedding(text_to_embed)

        collection.add(
            documents=[file_text],
            embeddings=[embedding],
            metadatas=[
                {
                    "table": table_name,
                    "file_name": fname,
                    "row_count": len(df),
                    "columns": ",".join(df.columns),
                }
            ],
            ids=[fname],
        )
        indexed_files += 1


with DAG(
    "boto3_index_parquet_files_to_chromadb",
    default_args=default_args,
    description="Indexing job from MinIO Parquet to ChromaDb",
    schedule="0 0 * * *",
    start_date=datetime(2024, 7, 1),
    catchup=False,
    tags=["parquet", "minio", "chromadb", "boto3"],
) as dag:
    download_task = PythonOperator(
        task_id="download_parquet_files",
        python_callable=download_parquet_files,
    )
    index_task = PythonOperator(
        task_id="index_parquet_files_as_documents",
        python_callable=index_parquet_files_as_documents,
    )

    download_task >> index_task

    download_task.doc_md = "### Download Task: Download Parquet files from MinIO bucket."
    index_task.doc_md = "### Index Task: Read intermediate files from MinIO, generate embeddings using Amazon Titan and index into ChromaDB."

    dag.doc_md = """
        ### Boto3 Indexing DAG
        This DAG downloads Parquet files from MinIO, extracts text, generates embeddings using Amazon Titan
        and indexes them into ChromaDB using XCom."""
