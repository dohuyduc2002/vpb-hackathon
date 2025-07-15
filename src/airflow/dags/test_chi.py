import pandas as pd
from minio import Minio
from fastavro import reader
from io import BytesIO
from clickhouse_connect import get_client
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

MINIO_ENDPOINT = "minio.minio.svc.cluster.local:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"
BUCKET = "stream-bucket"

CLICKHOUSE_HOST = "clickhouse-ch-cluster.clickhouse.svc.cluster.local"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = "ducdh"
CLICKHOUSE_PASSWORD = "test_password"
CLICKHOUSE_DB = "default"
CLICKHOUSE_TABLE = "test"

def read_all_avro_in_topic(bucket, topic, minio_client, usecols=None, filter_func=None):
    objects = minio_client.list_objects(
        bucket, prefix=f"topics/{topic}/", recursive=True
    )
    dfs = []
    for obj in objects:
        if obj.object_name.endswith(".avro") and "/partition=" in obj.object_name:
            data = minio_client.get_object(bucket, obj.object_name).read()
            with BytesIO(data) as bio:
                records = list(reader(bio))
                if records:
                    df = pd.DataFrame(records)
                    if usecols:
                        df = df[[col for col in usecols if col in df.columns]]
                    if filter_func:
                        df = filter_func(df)
                    if not df.empty:
                        dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=usecols if usecols else [])

def etl_and_insert_to_clickhouse():
    # MinIO client
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    usecols = ["user", "is_fraud"]
    df_transaction = read_all_avro_in_topic(
        BUCKET,
        "transaction-topic",
        minio_client,
        usecols=usecols,
    )

    # Kết nối ClickHouse
    client = get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DB,
        secure=False,
    )

    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {CLICKHOUSE_TABLE} (
            user String,
            is_fraud String
        ) ENGINE = MergeTree() ORDER BY user
        """
    )

    client.insert_df(CLICKHOUSE_TABLE, df_transaction)
    print("Insert vào ClickHouse thành công!")

with DAG(
    "minio_to_clickhouse_test_table",
    start_date=datetime(2025, 7, 20),
    schedule_interval="0 0 * * *",  # mỗi ngày lúc 00:00
    catchup=False,
    tags=["minio", "clickhouse", "test"],
) as dag:
    etl_task = PythonOperator(
        task_id="etl_and_insert_to_clickhouse",
        python_callable=etl_and_insert_to_clickhouse,
    )
