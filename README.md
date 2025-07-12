# vpb-hackathon
This readme will demonstrate how to set up a real-time fraud detection system using Kafka, Flink, Milvus, ClickHouse, and MinIO on a EKS cluster. The system will process credit card transactions in real-time, detect fraudulent activities, and store the data for further analysis.

## Install EKS cluster
Firstly, login to your AWS account through AWS CLI, ensure that your IAM account got these permission 
```bash
AmazonEKSClusterPolicy
AmazonEKS_CNI_Policy
AmazonEKSDashboardConsoleReadOnly
AmazonEKSComputePolicy
AmazonEKSBlockStoragePolicy
AmazonEBSCSIDriverPolicy
```

After that navigate to `terraform` directory to provision EKS cluster, this will provision EKS cluster with full acess to S3 and EC2.

## Download data
Firstly you have to download data from kaggle with kaggle cli, be sure to login with your kaggle account
```bash
#!/bin/bash
kaggle datasets download ealtman2019/credit-card-transactions
```
I'm also cleaning data to standardize the data before produce it to streaming, you can find the code in `src/streaming/clean_data.py`

## Build docker images
1. Kafka connect image
```bash
docker build --no-cache\
    -f Dockerfile.kafka_connect \
    -t microwave1005/kafka-connect-s3:0.0.2 \
    --push \
    .
```
2. API image
```bash

```

## Instalation
Be sure to following this order to install all components correctly, you can run this script to install all components
### Install Nginx Ingress Controller
```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer \
  --set controller.service.externalTrafficPolicy=Cluster \
  --set controller.resources.requests.cpu=100m \
  --set controller.resources.requests.memory=90Mi \
  --set controller.config.proxy-body-size="5120G" \
  --set controller.config.proxy-connect-timeout="600" \
  --set controller.config.proxy-send-timeout="600" \
  --set controller.config.proxy-read-timeout="600"
```

After Nginx controller got external IP, modify the `/etc/hosts` file to point the domain to the external IP of Nginx controller
```bash

34.132.171.2 dbeaver.ducdh.com kafka.ducdh.com milvus-example.local milvus-attu.local minio.ducdh.com console.minio.ducdh.com
```

### Install Strimzi Kafka Operator
```bash
helm repo add strimzi https://strimzi.io/charts/
helm install strimzi strimzi/strimzi-kafka-operator \
  --create-namespace \
  --namespace kafka 
```
### Install Kafka infra
```bash
helm upgrade --install kafka-infra ./helm/kafka -n kafka --create-namespace
```
### Install Minio
```bash
helm repo add minio https://charts.min.io/
helm upgrade --install minio minio/minio \
  --namespace minio \
  --create-namespace \
  --set mode=standalone \
  --set rootUser=minio \
  --set rootPassword=minio123 \
  --set persistence.size=10Gi \
  --set service.type=ClusterIP \
  --set resources.requests.memory=2Gi \
  --set ingress.enabled=true \
  --set ingress.ingressClassName=nginx \
  --set ingress.hosts[0]=minio.ducdh.com \
  --set consoleIngress.enabled=true \
  --set consoleIngress.ingressClassName=nginx \
  --set consoleIngress.hosts[0]=console.minio.ducdh.com 
```
Create minio bucket
```bash
mc alias set localMinio http://minio.ducdh.com minio minio123
mc mb localMinio/stream-bucket
mc mb localMinio/milvus-bucket
mc mb localMinio/flink-data
```

After these steps, produce streaming data to kafka topics with this command, this will produce roughly 26 million records to transaction topic, 6000 records to card topic and 2000 record to user topic, so it will take a while to finish, you can check the progress in Kafka UI. **You can continue to the next step while waiting for this command to finish.**
```bash
bash ./scripts/produce_data.sh
```

### Clickhouse
```bash
k create namespace clickhouse
# After namespace is created, you can install ClickHouse Operator this script
bash ./clickhouse/install.sh
# After ClickHouse Operator is installed, you can install ClickHouse cluster with this command
k apply -f clickhouse/clickhouse.yaml
```

### DBeaver
```bash
helm upgrade --install dbeaver ./helm/dbeaver -n clickhouse 
```
After installing DBeaver and create user, connect to ClickHouse with the provided credentials in `clickhouse/clickhouse.yaml` to create these tables in ClickHouse for Kafka Connect to consume data from Kafka topics and write to ClickHouse.
1. card_topic table
```sql
CREATE TABLE card_topic
(
    idx Int32,
    user Int64,
    card_index Int64,
    card_brand String,
    card_type String,
    card_number Int64,
    expires String,
    cvv Int32,
    has_chip String,
    cards_issued Int64,
    credit_limit String,
    acct_open_date String,
    year_pin_last_changed Int32,
    card_on_dark_web String
)
ENGINE = MergeTree()
ORDER BY idx;
```
2. user_topic table
```sql
CREATE TABLE user_topic
(
    idx                         Int32,
    person                      String,
    current_age                 Int32,
    retirement_age              Int32,
    birth_year                  Int32,
    birth_month                 Int32,
    gender                      String,
    address                     String,
    apartment                   Nullable(Float64),
    city                        String,
    state                       String,
    zipcode                     Int32,
    latitude                    Float64,
    longitude                   Float64,
    per_capita_income__zipcode  String,
    yearly_income__person       String,
    total_debt                  String,
    fico_score                  Int32,
    num_credit_cards            Int32
)
ENGINE = MergeTree
ORDER BY idx;

```

3. transaction_topic table
```sql
CREATE TABLE transaction_topic
(
    idx             Int32,
    user            Int64,
    card            Int64,
    year            Int32,
    month           Int32,
    day             Int32,
    time            String,
    amount          String,
    use_chip        Nullable(String),
    merchant_name   Nullable(Int64),
    merchant_city   Nullable(String),
    merchant_state  Nullable(String),
    zip             Nullable(Float64),
    mcc             Int32,
    errors          Nullable(String),
    is_fraud        String
)
ENGINE = MergeTree
ORDER BY idx;
```

### Install Kafka Connect and Kafka Connector
```bash
k apply -f kafka/
```
This will create a Kafka Connect cluster and Kafka Connectors to consume data from Kafka topics and write to ClickHouse, MinIO. 

### Milvus 
```bash
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm upgrade --install milvus milvus/milvus \
  --namespace milvus \
  --create-namespace \
  --set image.all.tag=v2.6.0-rc1 \
  --set cluster.enabled=false \
  --set minio.enabled=false \
  --set externalS3.enabled=true \
  --set externalS3.host=minio.minio.svc.cluster.local \
  --set externalS3.port=9000 \
  --set externalS3.accessKey=minio \
  --set externalS3.secretKey=minio123 \
  --set externalS3.bucketName=milvus-bucket \
  --set externalS3.useSSL=false \
  --set standalone.messageQueue=woodpecker \
  --set woodpecker.enabled=true \
  --set streaming.enabled=true \
  --set ingress.enabled=true \
  --set ingress.ingressClassName=nginx \
  --set standalone.persistence.persistentVolumeClaim.size=5Gi \
  --set attu.enabled=true \
  --set attu.ingress.enabled=true \
  --set attu.ingress.ingressClassName=nginx \
  --set pulsarv3.enabled=false
```

### Airflow
```bash
helm upgrade --install airflow apache-airflow/airflow \
 --namespace airflow \
 --create-namespace \
 --set flower.enabled=true \
 --set workers.persistence.size=5Gi \
 --set triggerer.persistence.size=5Gi \
 --set ingress.web.enabled=true \
 --set ingress.web.hosts[0]=airflow.ducdh.com \
 --set ingress.web.ingressClassName=nginx
```



kcat -b 34.172.236.120:9094 \
     -t user-topic \
     -r http://34.59.250.15:8081 \
     -s value=avro \
     -C -c 5