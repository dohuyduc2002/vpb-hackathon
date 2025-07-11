# vpb-hackathon

download kaggle data

```
#!/bin/bash
kaggle datasets download ealtman2019/credit-card-transactions
```

docker buildx build --no-cache\
    --platform linux/amd64 \
    -f Dockerfile.kafka_connect \
    -t microwave1005/kafka-connect-s3:0.0.1 \
    --push \
    .

# Install Nginx Ingress Controller
```bash
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
# Install Strimzi Kafka Operator
```
helm repo add strimzi https://strimzi.io/charts/
helm install strimzi strimzi/strimzi-kafka-operator \
  --create-namespace \
  --namespace kafka 
```
## Install Kafka infra
helm upgrade --install kafka-infra ./helm/kafka -n kafka --create-namespace
## Install Minio
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
mc mb localMinio/bronze-layer
mc mb localMinio/milvus-bucket
mc mb localMinio/flink-data
mc mb localMinio/silver-layer
```

## Instal flink jar
```bash
wget https://repo1.maven.org/maven2/org/apache/flink/flink-python/1.20.0/flink-python-1.20.0.jar
wget https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.9.0/kafka-clients-3.9.0.jar
wget https://repo1.maven.org/maven2/org/apache/flink/flink-streaming-java_2.12/1.20.0/flink-streaming-java_2.12-1.20.0.jar
wget https://repo1.maven.org/maven2/org/apache/flink/flink-table-api-java-bridge_2.12/1.20.0/flink-table-api-java-bridge_2.12-1.20.0.jar
wget https://repo1.maven.org/maven2/org/apache/flink/flink-java/1.20.0/flink-java-1.20.0.jar
wget https://repo1.maven.org/maven2/org/apache/flink/flink-connector-kafka/3.4.0-1.20/flink-connector-kafka-3.4.0-1.20.jar
wget https://repo1.maven.org/maven2/org/apache/avro/avro/1.12.0/avro-1.12.0.jar
wget https://repo1.maven.org/maven2/org/apache/flink/flink-avro/1.20.0/flink-avro-1.20.0.jar
wget https://repo1.maven.org/maven2/org/apache/flink/flink-parquet/1.20.0/flink-parquet-1.20.0.jar
wget https://repo1.maven.org/maven2/org/apache/flink/flink-s3-fs-hadoop/1.20.0/flink-s3-fs-hadoop-1.20.0.jar
wget https://repo1.maven.org/maven2/org/apache/flink/flink-avro-confluent-registry/1.20.0/flink-avro-confluent-registry-1.20.0.jar

wget https://packages.confluent.io/maven/io/confluent/kafka-schema-registry-client/7.5.0/kafka-schema-registry-client-7.5.0.jar
wget https://packages.confluent.io/maven/io/confluent/kafka-avro-serializer/7.5.0/kafka-avro-serializer-7.5.0.jar
```
## Cert Manager
```bash
helm repo add jetstack https://charts.jetstack.io 
helm install \
  cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.18.2 \
  --set crds.enabled=true
```

## Flink Operator
```bash
helm repo add flink-operator-repo https://downloads.apache.org/flink/flink-kubernetes-operator-1.12.1/
helm install flink-operator flink-operator-repo/flink-kubernetes-operator \
  --namespace flink \
  --create-namespace
```

# Milvus 
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
  --set ingress.host[1]=milvus.ducdh.com \
  --set standalone.persistence.persistentVolumeClaim.size=5Gi \
  --set attu.enabled=true \
  --set attu.ingress.enabled=true \
  --set attu.ingress.ingressClassName=nginx \
  --set attu.ingress.host[1]=attu.milvus.ducdh.com \
  --set pulsarv3.enabled=false
```
## Clickhouse
```bash
helm repo add hyperdx https://hyperdxio.github.io/helm-charts
helm upgrade --install clickhouse hyperdx/hdx-oss-v2 \
  --namespace clickhouse \
  --create-namespace \
  --set hyperdx.ingress.enabled=true \
  --set hyperdx.ingress.host=clickhouse.ducdh.com \
  --set clickhouse.persistence.dataSize=5Gi \
  --set clickhouse.persistence.logSize=1Gi \
  --set clickhouse.persistence.storageClass=standard-rwo \
  --set mongodb.persistence.dataSize=1Gi \
  --set global.storageClassName=standard-rwo 
```
## Airflow
```bash
helm upgrade --install airflow apache-airflow/airflow \
 --namespace airflow \
 --create-namespace \
 --set flower.enabled=true \
 --set workers.persistence.size=5Gi \
 --set triggerer.persistence.size=5Gi \
 --set ingress.web.enabled=true \
 --set ingress.web.hosts[0]=airflow.ducdh.com \
 --set ingress.web.ingressClassName=nginx \
 --set ingress.flower.enabled=true \
 --set ingress.flower.hosts[0]=flower.ducdh.com \
 --set ingress.flower.ingressClassName=nginx 
```
