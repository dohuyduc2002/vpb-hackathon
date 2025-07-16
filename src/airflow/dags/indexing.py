from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from minio import Minio
import pandas as pd
import os
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.bedrock import BedrockEmbedding
import chromadb

MINIO_ENDPOINT = "minio.minio.svc.cluster.local:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"
BUCKET = "stream-bucket"
MINIO_PREFIX = "etl/intermediate/"
CHROMA_COLLECTION = "fraud_knowledge_base"
BEDROCK_REGION = "us-east-1"
BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
LOCAL_PARQUET_DIR = "/tmp/parquet_files"  
CHROMA_HOST = "chroma-chromadb.chroma.svc.cluster.local"
CHROMA_PORT = 8000
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

def download_parquet_files(**kwargs):
    if not os.path.exists(LOCAL_PARQUET_DIR):
        os.makedirs(LOCAL_PARQUET_DIR, exist_ok=True)
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    objects = minio_client.list_objects(
        BUCKET, prefix=MINIO_PREFIX, recursive=True
    )
    downloaded_files = []
    for obj in objects:
        if obj.object_name.endswith(".parquet"):
            data = minio_client.get_object(BUCKET, obj.object_name).read()
            fname = obj.object_name.split("/")[-1]
            local_path = os.path.join(LOCAL_PARQUET_DIR, fname)
            with open(local_path, "wb") as f:
                f.write(data)
            print(f"Downloaded {obj.object_name} -> {local_path}")
            downloaded_files.append(local_path)
    print(f"Done! Downloaded {len(downloaded_files)} parquet files to {LOCAL_PARQUET_DIR}")
    kwargs["ti"].xcom_push(key="downloaded_files", value=downloaded_files)

def index_parquet_files_as_documents(**kwargs):
    downloaded_files = kwargs["ti"].xcom_pull(key="downloaded_files", task_ids="download_parquet_files")
    if not downloaded_files:
        print("No files to index!")
        return

    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = chroma_client.get_or_create_collection("fraud_knowledge_base")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = BedrockEmbedding(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=BEDROCK_REGION, 
        model_id=BEDROCK_EMBED_MODEL
    )
    indexed_files = 0
    for fpath in downloaded_files:
        fname = os.path.basename(fpath)
        table_name = fname.split("_")[0]
        df = pd.read_parquet(fpath)
        file_text = df.to_csv(index=False)
        doc = Document(
            text=file_text,
            metadata={
                "table": table_name,
                "file_name": fname,
                "row_count": len(df),
                "columns": list(df.columns),
            },
        )
        VectorStoreIndex.from_documents(
            [doc], storage_context=storage_context, embed_model=embed_model
        )
        print(f"Indexed {fname} ({len(df)} rows, table: {table_name})")
        indexed_files += 1

        os.remove(fpath)
    print(f"Done! Total {indexed_files} parquet files indexed as documents.")

with DAG(
    "load_and_index_parquet_files_as_documents",
    schedule="0 0 * * *",  
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["parquet", "minio", "chromadb"],
) as dag:
    load_task = PythonOperator(
        task_id="download_parquet_files", 
        python_callable=download_parquet_files,
    )
    index_task = PythonOperator(
        task_id="index_parquet_docs_to_chroma",
        python_callable=index_parquet_files_as_documents,
    )

    load_task >> index_task
