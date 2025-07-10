from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
import os


env = StreamExecutionEnvironment.get_execution_environment()
settings = EnvironmentSettings.in_streaming_mode()
table_env = StreamTableEnvironment.create(env, environment_settings=settings)

JARS_PATH = "/opt/flink/lib"
BOOTSTRAP_SERVERS = os.getenv("BOOTSTRAP_SERVERS")
SCHEMA_REGISTRY_URL = os.getenv("SCHEMA_REGISTRY_URL")

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

# Transaction Table
table_env.execute_sql(
    f"""
CREATE TABLE transactions (
    `user` BIGINT,
    card BIGINT,
    `year` INT,
    `month` INT,
    `day` INT,
    `time` STRING,
    amount STRING,
    use_chip STRING,
    merchant_name BIGINT,
    merchant_city STRING,
    merchant_state STRING,
    zip DOUBLE,
    mcc INT,
    errors STRING,
    is_fraud STRING
) WITH (
    'connector' = 'kafka',
    'topic' = 'transaction-topic',
    'properties.bootstrap.servers' = '{BOOTSTRAP_SERVERS}',
    'properties.group.id' = 'pyflink-sql-group-tx',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'avro-confluent',
    'avro-confluent.schema-registry.url' = '{SCHEMA_REGISTRY_URL}'
)
"""
)

# User Table
table_env.execute_sql(
    f"""
CREATE TABLE users (
    idx INT,
    person STRING,
    current_age INT,
    retirement_age INT,
    birth_year INT,
    birth_month INT,
    gender STRING,
    address STRING,
    apartment DOUBLE,
    city STRING,
    state STRING,
    zipcode INT,
    latitude DOUBLE,
    longitude DOUBLE,
    per_capita_income__zipcode STRING,
    yearly_income__person STRING,
    total_debt STRING,
    fico_score INT,
    num_credit_cards INT
) WITH (
    'connector' = 'kafka',
    'topic' = 'user-topic',
    'properties.bootstrap.servers' = '{BOOTSTRAP_SERVERS}',
    'properties.group.id' = 'pyflink-sql-group-user',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'avro-confluent',
    'avro-confluent.schema-registry.url' = '{SCHEMA_REGISTRY_URL}'
)
"""
)

# Card Table
table_env.execute_sql(
    f"""
CREATE TABLE cards (
    `user` BIGINT,
    card_index BIGINT,
    card_brand STRING,
    card_type STRING,
    card_number BIGINT,
    expires STRING,
    cvv INT,
    has_chip STRING,
    cards_issued BIGINT,
    credit_limit STRING,
    acct_open_date STRING,
    year_pin_last_changed INT,
    card_on_dark_web STRING
) WITH (
    'connector' = 'kafka',
    'topic' = 'card-topic',
    'properties.bootstrap.servers' = '{BOOTSTRAP_SERVERS}',
    'properties.group.id' = 'pyflink-sql-group-card',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'avro-confluent',
    'avro-confluent.schema-registry.url' = '{SCHEMA_REGISTRY_URL}'
)
"""
)

# Sink ra MinIO
table_env.execute_sql(
    """
CREATE TABLE minio_sink_fraud (
    `user` BIGINT,
    card BIGINT,
    person STRING,
    current_age INT,
    total_debt STRING,
    num_credit_cards INT,
    yearly_income__person STRING,
    fico_score INT,
    card_number BIGINT,
    credit_limit STRING,
    cards_issued BIGINT,
    card_on_dark_web STRING
) WITH (
    'connector' = 'filesystem',
    'path' = 's3a://silver-layer/final_fraud/',
    'format' = 'avro'
)
"""
)

# Join & Sink
table_env.execute_sql(
    """
INSERT INTO minio_sink_fraud
SELECT 
    t.`user`,
    t.card,
    u.person,
    u.current_age,
    u.total_debt,
    u.num_credit_cards,
    u.yearly_income__person,
    u.fico_score,
    c.card_number,
    c.credit_limit,
    c.cards_issued,
    c.card_on_dark_web
FROM transactions t
LEFT JOIN cards c ON t.`user` = c.`user`
LEFT JOIN users u ON t.`user` = u.idx
WHERE t.is_fraud = 'Yes'
"""
)

# INSERT INTO minio_sink_fraud
# SELECT
#     t.`user`,
#     COUNT(*) AS count_fraud,
#     u.person,
#     u.current_age,
#     u.total_debt,
#     c.card_number,
#     c.credit_limit
# FROM transactions t
# LEFT JOIN cards c ON t.`user` = c.`user`
# LEFT JOIN users u ON t.`user` = u.idx
# WHERE t.is_fraud = 'Yes'
# GROUP BY
#     t.`user`, u.person, u.current_age, u.total_debt, c.card_number, c.credit_limit
