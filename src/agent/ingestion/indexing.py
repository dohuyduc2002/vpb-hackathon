from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from app.config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, EMBED_DIM

vector_store = MilvusVectorStore(
    uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
    collection_name=COLLECTION_NAME,
    dim=EMBED_DIM,
)


def index_data(data_dir="data/raw"):
    documents = SimpleDirectoryReader(data_dir).load_data()
    VectorStoreIndex.from_documents(documents, vector_store=vector_store)
    print(f"Indexed {len(documents)} documents into Milvus.")


if __name__ == "__main__":
    index_data()
