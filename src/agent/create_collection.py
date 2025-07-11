from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

MILVUS_HOST = "localhost"  # hoặc endpoint Milvus thật
MILVUS_PORT = "19530"
COLLECTION_NAME = "transaction_embedding"  # Đây chính là MILVUS_COLLECTION


def create_milvus_collection():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    # Kiểm tra collection đã tồn tại chưa
    if COLLECTION_NAME in [c.name for c in Collection.list()]:
        print(f"Collection {COLLECTION_NAME} đã tồn tại.")
        return

    # Định nghĩa schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="raw_text", dtype=DataType.VARCHAR, max_length=65535),
        # Bạn có thể thêm các field metadata khác nếu cần (vd: timestamp, transaction_id...)
    ]
    schema = CollectionSchema(fields, description="Transaction embedding collection")

    # Tạo collection
    collection = Collection(COLLECTION_NAME, schema)
    print(f"Đã tạo collection {COLLECTION_NAME} thành công!")


if __name__ == "__main__":
    create_milvus_collection()
