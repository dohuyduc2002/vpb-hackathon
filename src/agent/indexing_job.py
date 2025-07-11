import requests
import json

AIRFLOW_BASE_URL = "http://localhost:8080"  # đổi lại thành IP/host thực tế nếu cần
DAG_ID = "minio_to_milvus_indexing"  # Tên DAG đã setup
USERNAME = "airflow"  # hoặc tài khoản airflow của bạn
PASSWORD = "airflow"  # hoặc mật khẩu thực tế


# Gọi REST API để trigger DAG
def trigger_dag():
    url = f"{AIRFLOW_BASE_URL}/api/v1/dags/{DAG_ID}/dagRuns"
    payload = {
        "conf": {},  # Có thể truyền tham số vào DAG nếu muốn
    }
    resp = requests.post(url, json=payload, auth=(USERNAME, PASSWORD))
    if resp.status_code in (200, 201):
        print("DAG triggered successfully:", resp.json())
    else:
        print("Error triggering DAG:", resp.status_code, resp.text)


if __name__ == "__main__":
    trigger_dag()
