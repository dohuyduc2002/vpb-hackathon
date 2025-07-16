import io
import pandas as pd
from minio import Minio
from fastavro import reader
from io import BytesIO
from clickhouse_connect import get_client
from datetime import datetime, timedelta
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator

logging.basicConfig(level=logging.INFO)

MINIO_ENDPOINT = "minio.minio.svc.cluster.local:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"
BUCKET = "stream-bucket"
CLICKHOUSE_HOST = "clickhouse-stream.clickhouse.svc.cluster.local"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = "ducdh"
CLICKHOUSE_PASSWORD = "test_password"
CLICKHOUSE_DB = "default"
CLICKHOUSE_TABLE = "fraud_report"

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


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


def df_to_minio_parquet(df, minio_client, bucket, object_key):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    minio_client.put_object(
        bucket,
        object_key,
        buf,
        length=buf.getbuffer().nbytes,
        content_type="application/octet-stream",
    )
    return object_key


def df_from_minio_parquet(minio_client, bucket, object_key):
    data = minio_client.get_object(bucket, object_key).read()
    buf = io.BytesIO(data)
    df = pd.read_parquet(buf)
    return df


def extract(**kwargs):
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    trans_cols = ["user", "is_fraud"]
    filter_fraud = lambda df: df[df["is_fraud"] == "Yes"]
    df_transaction = read_all_avro_in_topic(
        BUCKET,
        "transaction-topic",
        minio_client,
        usecols=trans_cols,
        filter_func=filter_fraud,
    )
    user_cols = ["idx", "person", "current_age", "total_debt"]
    df_user = read_all_avro_in_topic(
        BUCKET, "user-topic", minio_client, usecols=user_cols
    )
    card_cols = ["user", "card_number", "credit_limit"]
    df_card = read_all_avro_in_topic(
        BUCKET, "card-topic", minio_client, usecols=card_cols
    )

    execution_date = kwargs["ds_nodash"] 
    key_tran = f"etl/intermediate/transaction_{execution_date}.parquet"
    key_user = f"etl/intermediate/user_{execution_date}.parquet"
    key_card = f"etl/intermediate/card_{execution_date}.parquet"

    df_to_minio_parquet(df_transaction, minio_client, BUCKET, key_tran)
    df_to_minio_parquet(df_user, minio_client, BUCKET, key_user)
    df_to_minio_parquet(df_card, minio_client, BUCKET, key_card)

    ti = kwargs["ti"]
    ti.xcom_push(key="transaction_key", value=key_tran)
    ti.xcom_push(key="user_key", value=key_user)
    ti.xcom_push(key="card_key", value=key_card)


def transform(**kwargs):
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    ti = kwargs["ti"]
    key_tran = ti.xcom_pull(key="transaction_key", task_ids="extract_task")
    key_user = ti.xcom_pull(key="user_key", task_ids="extract_task")
    key_card = ti.xcom_pull(key="card_key", task_ids="extract_task")

    df_transaction = df_from_minio_parquet(minio_client, BUCKET, key_tran)
    df_user = df_from_minio_parquet(minio_client, BUCKET, key_user)
    df_card = df_from_minio_parquet(minio_client, BUCKET, key_card)

    # Join 
    df_join = pd.merge(df_transaction, df_card, on="user", how="left")
    df_join = pd.merge(df_join, df_user, left_on="user", right_on="idx", how="left")

    # groupby & aggregate
    result = df_join.groupby("user", as_index=False).agg(
        count_fraud=("user", "size"),
        person=("person", "first"),
        current_age=("current_age", "first"),
        total_debt=("total_debt", "first"),
        card_number=("card_number", "first"),
        credit_limit=("credit_limit", "first"),
    )

    execution_date = kwargs["ds_nodash"]
    result_key = f"etl/intermediate/fraud_report_{execution_date}.parquet"
    df_to_minio_parquet(result, minio_client, BUCKET, result_key)
    ti.xcom_push(key="result_key", value=result_key)


def load(**kwargs):
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    ti = kwargs["ti"]
    result_key = ti.xcom_pull(key="result_key", task_ids="transform_task")
    result = df_from_minio_parquet(minio_client, BUCKET, result_key)

    result["user"] = result["user"].astype("uint32")
    result['count_fraud'] = result['count_fraud'].astype('uint32')
    result['person'] = result['person'].astype(str)
    result['current_age'] = result['current_age'].astype('Int32')

    result['total_debt'] = result['total_debt'].astype(str)
    result.loc[result['total_debt'].isin(['nan', 'None', 'NaN']), 'total_debt'] = None

    result['card_number'] = result['card_number'].astype(str)
    result.loc[result['card_number'].isin(['nan', 'None', 'NaN']), 'card_number'] = None

    result['credit_limit'] = result['credit_limit'].astype(str)
    result.loc[result['credit_limit'].isin(['nan', 'None', 'NaN']), 'credit_limit'] = None

    # convert NaN in string to None
    result = result.where(pd.notnull(result), None)

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
            user UInt32,
            count_fraud UInt32,
            person String,
            current_age Nullable(Int32),
            total_debt Nullable(String),
            card_number Nullable(String),
            credit_limit Nullable(String)
        ) ENGINE = MergeTree() ORDER BY user
        """
    )
    client.insert_df(CLICKHOUSE_TABLE, result)


with DAG(
    dag_id="minio_to_clickhouse_fraud_report",
    default_args=default_args,
    description="ETL from MinIO Avro (multi-file) to ClickHouse",
    schedule="0 0 * * *",  
    start_date=datetime(2024, 7, 1),
    catchup=False,
    tags=["minio", "clickhouse", "fraud"],
) as dag:

    extract_task = PythonOperator(
        task_id="extract_task",
        python_callable=extract,
    )

    transform_task = PythonOperator(
        task_id="transform_task",
        python_callable=transform,
    )

    load_task = PythonOperator(
        task_id="load_task",
        python_callable=load,
    )

    extract_task >> transform_task >> load_task

    # Documentation
    extract_task.doc_md = "### Extract Task: Read all avro files from MinIO, save DataFrame to as 1 intermediate file MinIO (Parquet)."
    transform_task.doc_md = "### Transform Task: Read intermediate files MinIO, join/aggregate and save aggregated as parquet."
    load_task.doc_md = (
        "### Load Task: Read aggregated parquet from MinIO and insert to ClickHouse."
    )
    dag.doc_md = """
        ### MinIO to ClickHouse Fraud Report DAG (multi-file ETL, using MinIO object key to parse XCom between tasks)
        """
