import chromadb
from dotenv import load_dotenv
import os
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock_converse import BedrockConverse

load_dotenv()
# --- Cấu hình AWS credentials và region ---
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_session_token = None
aws_region = "us-east-1"

bedrock_embedding = BedrockEmbedding(
    region_name=aws_region,
    model_id="amazon.titan-embed-text-v2:0",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
)

# --- Khởi tạo Chroma VectorStore ---
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection("bedrock_nova_demo")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- Index sample docs vào Chroma ---
docs = [
    Document(text="The capital of Vietnam is Hanoi.", metadata={"country": "Vietnam", "topic": "capital"}),
    Document(text="Pho is a famous Vietnamese noodle soup.", metadata={"country": "Vietnam", "topic": "food"}),
    Document(text="The capital of Japan is Tokyo.", metadata={"country": "Japan", "topic": "capital"}),
]
index = VectorStoreIndex.from_documents(
    docs, storage_context=storage_context, embed_model=bedrock_embedding
)
print("Đã index xong dữ liệu vào Chroma sử dụng embedding Bedrock.")

# --- Khởi tạo LLM với BedrockConverse Nova ---
llm = BedrockConverse(
    model="us.amazon.nova-lite-v1:0",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name=aws_region,
)

# --- Test completion trực tiếp với Nova LLM ---
response = llm.complete("Paul Graham is ")
print("\nKết quả Nova completion:")
print(response)

# --- QueryEngine kết hợp retrieval + Nova LLM ---
query_engine = index.as_query_engine(llm=llm)
question = "What is the capital of Vietnam?"
response2 = query_engine.query(question)
print("\nKết quả truy vấn RAG (Nova LLM trả lời):")
print(response2)
