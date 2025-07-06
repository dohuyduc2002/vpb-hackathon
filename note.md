aws iam list-attached-user-policies --user-name fsds

aws eks update-kubeconfig --region ap-southeast-1 --name eks-fsds
aws iam list-attached-role-policies --role-name main-eks-node-group-20250705160337935100000002

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


helm repo add strimzi https://strimzi.io/charts/
helm install strimzi strimzi/strimzi-kafka-operator \
  --create-namespace \
  --namespace kafka 

helm repo add kafka-ui https://provectus.github.io/kafka-ui-charts

helm upgrade --install kafka-ui kafka-ui/kafka-ui \
  --namespace kafka \
  --set envs.config.KAFKA_CLUSTERS_0_NAME=kafka-cluster-0 \
  --set envs.config.KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS=kafka-cluster-1-kafka-bootstrap:9092 \
  --set ingress.enabled=true \
  --set ingress.host=kafka.ducdh.com \
  --set ingress.ingressClassName=nginx


k apply -f kafka_cluster.yaml -n kafka

helm upgrade --install airflow apache-airflow/airflow \
 --namespace airflow \
 --create-namespace \
 --set flower.enabled=true \
 --set workers.persistence.size=10Gi \
 --set triggerer.persistence.size=10Gi \
 --set ingress.web.enabled=true \
 --set ingress.web.hosts[0]=airflow.ducdh.com \
 --set ingress.web.ingressClassName=nginx \
 --set ingress.flower.enabled=true \
 --set ingress.flower.hosts[0]=flower.ducdh.com \
 --set ingress.flower.ingressClassName=nginx 
 


helm repo add jetstack https://charts.jetstack.io --force-update
helm install \
  cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.18.2 \
  --set crds.enabled=true

helm repo add flink-operator-repo https://downloads.apache.org/flink/flink-kubernetes-operator-1.12.0/

helm install flink-operator flink-operator-repo/flink-kubernetes-operator \
  --namespace flink \
  --create-namespace



helm repo add spark https://apache.github.io/spark-kubernetes-operator
helm install spark spark/spark-kubernetes-operator \
  --namespace spark \
  --create-namespace 

helm repo add minio https://charts.min.io/

helm install minio minio/minio \
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


mc alias set localMinio http://minio.ducdh.com minio minio123
mc mb localMinio/bronze-layer
mc mb localMinio/mlflow

mc cp --recursive ./data localMinio/bronze-layer
mc ls --recursive localMinio/bronze-layer


git clone https://github.com/confluentinc/cp-helm-charts.git
helm install cp-helm-charts