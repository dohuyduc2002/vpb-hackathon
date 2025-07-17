from fastapi import FastAPI
from pydantic import BaseModel
from app.config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, EMBED_DIM
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.openai import OpenAI

vector_store = MilvusVectorStore(
    uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
    collection_name=COLLECTION_NAME,
    dim=EMBED_DIM,
)
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2), similarity_top_k=4
)

app = FastAPI(title="RAG Agent API (LlamaIndex)")


class AskRequest(BaseModel):
    query: str


@app.post("/ask")
def ask(request: AskRequest):
    response = query_engine.query(request.query)
    return {
        "answer": str(response),
        "context": [node.text for node in getattr(response, "source_nodes", [])],
    }
