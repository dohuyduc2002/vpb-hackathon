import streamlit as st
import requests
import os
from dotenv import load_dotenv

from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core import SQLDatabase
from sqlalchemy import create_engine
import chromadb

load_dotenv()

API_FQDN = "http://fraud-model-api.api.svc.cluster.local:8000"
CHROMA_HOST = "chroma-chromadb.chroma.svc.cluster.local"
CHROMA_PORT = 8000
COLLECTION_NAME = "fraud_titanv2_indexing"
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.environ.get("AWS_SESSION_TOKEN")
AWS_REGION = "us-east-1"
CLICKHOUSE_URL = "clickhouse+http://ducdh:test_password@clickhouse-stream.clickhouse.svc.cluster.local:8123/default"


def predict_fraud_api(sample):
    try:
        response = requests.post(f"{API_FQDN}/predict", json=sample, timeout=10)
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction")
            probs = result.get("probabilities")
            confidence = probs[0][1]
            return prediction, confidence
        else:
            st.error(f"Lỗi từ model API: {response.text}")
            return None, None
    except Exception as e:
        st.error(f"Không thể gọi model API: {str(e)}")
        return None, None


st.set_page_config(
    page_title="Fraud Detection & Chat System", page_icon="", layout="wide"
)


def main():
    st.title("Fraud Detection & Chat System")
    tab1, tab2 = st.tabs(["Fraud Detection", "Chat with Agent"])

    # --- Fraud Detection Tab ---
    with tab1:
        st.header("Transaction Fraud Detection")
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            with col1:
                user = st.text_input("User ID", value="user1")
                card = st.text_input("Card ID", value="card1")
                year = st.text_input("Year", value="2023")
                month = st.text_input("Month", value="5")
                day = st.text_input("Day", value="20")
                time = st.text_input("Time", value="12:00")
            with col2:
                amount = st.text_input("Amount", value="1200")
                use_chip = st.text_input("Use Chip", value="Chip Transaction")
                merchant_name = st.text_input(
                    "Merchant ID", value="3189517333335617109"
                )
                merchant_city = st.text_input("Merchant City", value="Riverside")
                merchant_state = st.text_input("Merchant State", value="CA")
                zip_code = st.text_input("ZIP Code", value="92505.0")
            col3, col4 = st.columns(2)
            with col3:
                mcc = st.text_input("MCC Code", value="7538")
            with col4:
                errors = st.text_input(
                    "Transaction Errors", value="Insufficient Balance"
                )

            submitted = st.form_submit_button("🔍 Predict Fraud", type="primary")
            if submitted:
                sample = {
                    "user": user,
                    "card": card,
                    "year": year,
                    "month": month,
                    "day": day,
                    "time": time,
                    "amount": amount,
                    "use_chip": use_chip,
                    "merchant_name": merchant_name,
                    "merchant_city": merchant_city,
                    "merchant_state": merchant_state,
                    "zip": zip_code,
                    "mcc": mcc,
                    "errors": errors,
                }

                with st.spinner("Analyzing transaction..."):
                    result, confidence = predict_fraud_api(sample)

                if result is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        if result == "Yes":
                            st.error("FRAUD DETECTED!")
                        else:
                            st.success(
                                f"Transaction appears legitimate (Confidence: {confidence:.2%})"
                            )
                    with col2:
                        st.metric("Prediction", result)

                    with st.expander("Transaction Details"):
                        st.json(sample)

    with tab2:
        st.header("Chat with Agent (RAG: Chroma + ClickHouse + Nova LLM)")
        @st.cache_resource(show_spinner="🔌 Initialising Bedrock models…")
        def get_bedrock_models():
            embed = BedrockEmbedding(
                aws_access_key_id     = AWS_ACCESS_KEY_ID,
                aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
                region_name           = AWS_REGION,
                model_name            = "amazon.titan-embed-text-v2:0",
            )
            llm = BedrockConverse(
                model                = "us.amazon.nova-lite-v1:0",
                aws_access_key_id     = AWS_ACCESS_KEY_ID,
                aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
                region_name           = AWS_REGION,
            )
            # fallback cho các constructor chưa support embed_model/llm
            Settings.embed_model = embed
            Settings.llm         = llm
            return embed, llm

        @st.cache_resource(show_spinner="📚 Building Chroma index…")
        def get_vector_index():
            embed, _ = get_bedrock_models()

            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            try:
                collection = chroma_client.get_collection(name=COLLECTION_NAME)
            except Exception:
                collection = chroma_client.create_collection(name=COLLECTION_NAME)

            vector_store    = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex.from_vector_store(
                vector_store    = vector_store,
                storage_context = storage_context,
                embed_model     = embed,          # truyền cục bộ
            )
            return index

        @st.cache_resource(show_spinner="🤖 Spinning up Router engine…")
        def get_auto_engine():
            embed, llm = get_bedrock_models()
            index      = get_vector_index()

            # Vector query engine
            vector_engine = index.as_query_engine(llm=llm)

            # NL-to-SQL engine (ClickHouse)
            sql_db  = SQLDatabase(create_engine(CLICKHOUSE_URL))
            sql_eng = NLSQLTableQueryEngine(
                sql_database=sql_db, llm=llm, embed_model=embed
            )

            sql_tool = QueryEngineTool(
                query_engine=sql_eng,
                metadata=ToolMetadata(
                    name="sql",
                    description="Answer questions from ClickHouse",
                ),
            )

            vector_tool = QueryEngineTool(
                query_engine=vector_engine,
                metadata=ToolMetadata(
                    name="vector",
                    description="Answer questions from document KB",
                ),
            )

            return RouterQueryEngine(
                selector=LLMSingleSelector.from_defaults(), query_engine_tools=[sql_tool, vector_tool]
            )

        try:
            router_engine = get_auto_engine()  
            if "chat_history_tab2" not in st.session_state:
                st.session_state.chat_history_tab2 = []

            st.subheader("Ask anything about transactions (RAG/SQL):")
            user_input = st.text_input(
                "Type your message:",
                placeholder="e.g. Show me suspicious transactions or Count fraud by user",
                key="chat_input_tab2",
            )

            if user_input:
                with st.spinner("Agent is reasoning and searching..."):
                    response = router_engine.query(user_input)

                # Lưu & hiển thị lịch sử
                st.session_state.chat_history_tab2.append(
                    {"role": "user", "content": user_input}
                )
                st.session_state.chat_history_tab2.append(
                    {"role": "assistant", "content": str(response)}
                )

            # Render chat transcript
            for msg in st.session_state.chat_history_tab2:
                speaker = "You" if msg["role"] == "user" else "Agent"
                st.markdown(f"**{speaker}:** {msg['content']}")

            # Hiện truy vấn SQL (nếu có)
            if (
                user_input
                and hasattr(response, "metadata")
                and "sql_query" in getattr(response, "metadata", {})
            ):
                with st.expander("SQL Query Used"):
                    st.code(response.metadata["sql_query"])

        except Exception as e:
            st.error(f"Error initializing chat system: {e}")
            st.info("Make sure ChromaDB, ClickHouse, and credentials are properly set.")

if __name__ == "__main__":
    main()
