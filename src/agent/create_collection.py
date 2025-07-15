from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

MILVUS_HOST = "127.0.0.1"  # Due to milvus ingress use gRPC, using k port-forward svc/milvus -n milvus 19530:19530
MILVUS_PORT = "19530"

# Định nghĩa thông tin cho từng collection
COLLECTIONS = [
    {"name": "user_embedding", "desc": "User embedding collection"},
    {"name": "card_embedding", "desc": "Card embedding collection"},
    {"name": "transaction_embedding", "desc": "Transaction embedding collection"},
]

EMBEDDING_DIM = 1536


def create_collection_if_not_exists(collection_name, description, embedding_dim):
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    if collection_name in utility.list_collections():
        print(f"Collection '{collection_name}' đã tồn tại.")
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        FieldSchema(name="raw_text", dtype=DataType.VARCHAR, max_length=65535),
    ]
    schema = CollectionSchema(fields, description=description)

    collection = Collection(collection_name, schema)
    print(f"Đã tạo collection '{collection_name}' thành công!")


if __name__ == "__main__":
    for col in COLLECTIONS:
        create_collection_if_not_exists(col["name"], col["desc"], EMBEDDING_DIM)
