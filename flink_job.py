from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings

env = StreamExecutionEnvironment.get_execution_environment()
settings = EnvironmentSettings.in_streaming_mode()
table_env = StreamTableEnvironment.create(env, environment_settings=settings)

JARS_PATH = "/opt/flink/lib"

# table_env.get_config().set(
#     "pipeline.jars",
#     ";".join(
#         [
#             f"file://{JARS_PATH}/avro-1.12.0.jar",
#             f"file://{JARS_PATH}/flink-avro-1.20.0.jar",
#             f"file://{JARS_PATH}/flink-connector-kafka-3.4.0-1.20.jar",
#             f"file://{JARS_PATH}/flink-java-1.20.0.jar",
#             f"file://{JARS_PATH}/flink-parquet-1.20.0.jar",
#             f"file://{JARS_PATH}/flink-python-1.20.0.jar",
#             f"file://{JARS_PATH}/flink-s3-fs-hadoop-1.20.0.jar",
#             f"file://{JARS_PATH}/kafka-avro-serializer-7.5.0.jar",
#             f"file://{JARS_PATH}/kafka-clients-3.9.0.jar",
#             f"file://{JARS_PATH}/kafka-schema-registry-client-7.5.0.jar",
#         ]
#     ),
# )

# Tạo bảng Kafka source
table_env.execute_sql("""
CREATE TABLE transactions (
    user_id STRING,
    transaction_id STRING,
    amount DOUBLE,
    is_fraud INT
) WITH (
    'connector' = 'kafka',
    'topic' = 'transaction-topic',
    'properties.bootstrap.servers' = '34.171.132.202:9094',
    'properties.group.id' = 'pyflink-sql-group',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'json',
    'json.fail-on-missing-field' = 'false',
    'json.ignore-parse-errors' = 'true'
)
""")

# Sink 1: Count fraud
table_env.execute_sql("""
CREATE TABLE minio_sink_count (
    user_id STRING,
    fraud_count BIGINT
) WITH (
    'connector' = 'filesystem',
    'path' = 's3a://silver-layer/fraud_count/',
    'format' = 'avro'
)
""")

table_env.execute_sql("""
INSERT INTO minio_sink_count
SELECT 
    user_id,
    COUNT(*) AS fraud_count
FROM transactions
WHERE is_fraud = 1
GROUP BY user_id
""")

table_env.execute_sql("""
INSERT INTO minio_sink_all
SELECT user_id, transaction_id, amount, is_fraud
FROM transactions
""")
