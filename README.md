# vpb-hackathon
This readme will demonstrate how to set up a real-time fraud detection system using Kafka, Flink, Milvus, ClickHouse, and MinIO on a EKS cluster. The system will process credit card transactions in real-time, detect fraudulent activities, and store the data for further analysis.

## Install EKS cluster
Firstly, login to your AWS account through AWS CLI, for POC, I'm using `AdministratorAccess` policy. After that navigate to `terraform` directory to provision EKS cluster, this will provision EKS cluster with full acess to S3 and EC2. To be able to provision EKS cluster, you have to configure your AWS credentials with `aws configure` command and login with your AWS IAM account.
```bash
cd terraform
terraform init #This will download all required providers, in this case I'm using another EKS module to provision EKS cluster
terraform apply
```
After the EKS cluster is created, add context to your kubeconfig file to access the EKS cluster with kubectl
```bash
aws eks update-kubeconfig --region us-east-1 --name fsds
```
You also need to crate storage class EBS GP3 for other components to use
```bash
kubectl apply -f ebs-sc.yaml
```
## Download data
Firstly you have to download data from kaggle with kaggle cli, be sure to login with your kaggle account
```bash
#!/bin/bash
kaggle datasets download ealtman2019/credit-card-transactions
```
I'm also cleaning data to standardize the data before produce it to streaming, you can find the code in `src/streaming/clean_data.py`, this script will create 3 cleaned files in `data` folder
## Build docker images
1. Kafka connect image
```bash
docker build --no-cache\
    -f Dockerfile.kafka_connect \
    -t microwave1005/kafka-connect-s3:0.0.1 \
    --push \
    .
```
2. API image
```bash
docker build --no-cache \
    -f Dockerfile.api \
    -t microwave1005/fraud-model-api:0.0.1 \
    --push \
    .
```

3. Airflow image
```bash
docker build --no-cache \
    -f Dockerfile.airflow \
    -t microwave1005/airflow-custom:0.0.3 \
    --push \
    .
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

After Nginx controller got external IP, modify the `/etc/hosts` file to point the domain to the external IP of Nginx controller, you can get the external IP with this command
```bash
nslookup a332765274bd34e71b8b558cbe23c12a-1675950531.us-east-1.elb.amazonaws.com

3.217.98.32 dbeaver.ducdh.com kafka.ducdh.com milvus-example.local milvus-attu.local minio.ducdh.com console.minio.ducdh.com
```
### Install Strimzi Kafka Operator
In this case, I'm using Strimzi Kafka Operator to manage Kafka cluster with customized CRDs which is useful for managing Kafka Connect, Kafka Connectors, and other Kafka related resources.
```bash
helm repo add strimzi https://strimzi.io/charts/
helm install strimzi strimzi/strimzi-kafka-operator \
  --create-namespace \
  --namespace kafka 
```
### Install Kafka infra
After Strimzi is installed, I have created a custom Kafka infrastructure with 1 Kafka broker, 1 Kafka Controler using Kraft, and 3 topics which is `transaction-topic`, `card-topic`, and `user-topic`. For the UI, I'm using Kafka Provectus and connect to the broker. In chart, I'm also change the bootstrap server and Confluent Kafka Schema Registry to use the external IP to producer messages from Python script to the EKS cluster. 
```bash
helm upgrade --install kafka-infra ./helm/kafka -n kafka --create-namespace
```
### Install Minio
I'm using MinIO as object storage for this POC to store data from Kafka Connect which will sink messages from Kafka topic to MinIO.
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
```
In this repository, I'm producing data as Avro format to Kafka topics, so you need to create a schema registry to register the Avro schemas for each topic, and then register the schemas to the schema registry. After these steps, produce streaming data to kafka topics with this command, this will produce roughly 26 million records to transaction topic, 6000 records to card topic and 2000 record to user topic, so it will take a while to finish, you can check the progress in Kafka UI. 
**You can continue to the next step while waiting for this command to finish.**
```bash
cd src/producer
bash run.sh
```
### Clickhouse
Because we are producing streaming data, so ClickHouse is the suitable OLAP database to store and analyze data. In this POC, I'm using ClickHouse Operator to manage ClickHouse cluster and ClickHouse tables. You can install ClickHouse Operator with this command
```bash
k create namespace clickhouse
# After namespace is created, you can install ClickHouse Operator this script
bash ./clickhouse/install.sh
# After ClickHouse Operator is installed, you can install ClickHouse cluster with this command
k apply -f clickhouse/clickhouse.yaml -n clickhouse
```

### DBeaver
For end users to interact with ClickHouse, I'm using DBeaver as a database client with my custom helm chart.
```bash
helm upgrade --install dbeaver ./helm/dbeaver -n clickhouse 
```
After installing DBeaver and create user, connect to ClickHouse with the provided credentials in `clickhouse/clickhouse.yaml` to create these tables in ClickHouse for Kafka Connect to consume data from Kafka topics and write to ClickHouse.
1. card_topic table
```sql
CREATE TABLE card_topic
(
    idx Int32,
    user String,
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
    user            String,
    card            String,
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
After create table in ClickHouse and bucket in MinIO, I'm installing 1 Kafka Connect cluster with 6 connectors to consume data from Kafka topics and sink to ClickHouse and MinIO. The Kafka Connect cluster will be configured to use the Confluent Schema Registry to register the Avro schemas for each topic, each message will be serialized as `Avro` format for value and `String` format for key.
```bash
k apply -f kafka/
```

### Airflow
For ETL process, I'm using Airflow to orchestrate the ETL process which load data from Minio to ClickHouse. This will produce intermediate data as `parquet` format in MinIO and then load it to ClickHouse.
```bash
helm repo add apache-airflow https://airflow.apache.org
helm upgrade --install airflow apache-airflow/airflow \
  --namespace airflow \
  --create-namespace \
  --set defaultAirflowRepository=microwave1005/airflow-custom \
  --set defaultAirflowTag=0.0.17 \
  --set workers.persistence.size=5Gi \
  --set triggerer.persistence.size=5Gi \
  --set ingress.web.enabled=true \
  --set ingress.web.hosts[0]=airflow.ducdh.com \
  --set ingress.web.ingressClassName=nginx
```

kubectl create secret generic aws-cred \
  --namespace airflow \
  --from-literal=AWS_ACCESS_KEY_ID=AKIAQQABDTYWKRDN2ZMR \
  --from-literal=AWS_SECRET_ACCESS_KEY=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

## Model description 


k port-forward svc/airflow-api-server 8080:8080
kubectl delete all --all -n airflow --grace-period=0 --force
k delete pvc --all -n airflow

docker build --no-cache \
    -f Dockerfile.airflow \
    -t microwave1005/airflow-custom:0.0.17 \
    --push \
    .


kcat -b 34.172.236.120:9094 \
     -t user-topic \
     -r http://34.59.250.15:8081 \
     -s value=avro \
     -C -c 5


helm repo add chroma https://amikos-tech.github.io/chromadb-chart/
helm install chroma chroma/chromadb \
  --namespace chroma \
  --create-namespace \
  --set ingress.enabled=true \
  --set ingress.className=nginx \
  --set ingress.hosts[0].host=chroma.ducdh.com \
  --set ingress.hosts[0].paths[0].path=/ \
  --set ingress.hosts[0].paths[0].pathType=ImplementationSpecific


kubectl create secret generic aws-cred \
  --namespace airflow \
  --from-literal=AWS_ACCESS_KEY_ID=AKIAQQABDTYWKRDN2ZMR \
  --from-literal=AWS_SECRET_ACCESS_KEY=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
