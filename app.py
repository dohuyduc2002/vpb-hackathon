import streamlit as st
import torch
import numpy as np
import requests
import json
from datetime import datetime
from torch_geometric.nn import GAE
from input_preprocess import preprocess_single
from model.gat_encoder import GATEncoderWithEdgeAttrs
from model.edge_classifier import EdgeMLPClassifier

try:
    from llama_index.core import Document, VectorStoreIndex, StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.bedrock import BedrockEmbedding
    from llama_index.llms.bedrock_converse import BedrockConverse
    import chromadb
    CHAT_AVAILABLE = True
except ImportError:
    CHAT_AVAILABLE = False

st.set_page_config(
    page_title="Fraud Detection & Chat System",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the fraud detection model"""
    try:
        ckpt = torch.load("fraud_gnn_model.pt", map_location="cpu")
        node_mapping = ckpt["node_mapping"]
        scaler = ckpt["scaler"]
        label_encoders = ckpt["label_encoders"]
        
        encoder = GATEncoderWithEdgeAttrs(
            in_channels=len(node_mapping), 
            hidden_channels=64, 
            edge_attr_dim=7
        )
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        model = GAE(encoder)
        
        classifier = EdgeMLPClassifier(emb_dim=64, hidden_dim=64, num_classes=2)
        classifier.load_state_dict(ckpt["classifier_state_dict"])
        
        encoder.eval()
        classifier.eval()
        
        return encoder, classifier, node_mapping, scaler, label_encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None

def predict_fraud(sample, encoder, classifier, node_mapping, scaler, label_encoders):
    """Predict fraud for a given transaction sample"""
    try:
        edge_index, edge_attr = preprocess_single(sample, node_mapping, scaler, label_encoders)
        x = torch.eye(len(node_mapping))
        
        with torch.no_grad():
            z, processed_edge_attr = encoder(x, edge_index, edge_attr)
            out = classifier(z, edge_index, processed_edge_attr)
            pred = torch.argmax(out, dim=1).item()
            confidence = torch.softmax(out, dim=1).max().item()
        
        result = "Yes" if pred == 1 else "No"
        return result, confidence
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def send_slack_webhook(webhook_url, transaction_data, prediction_result):
    """Send notification to Slack webhook"""
    try:
        message = {
            "text": f"FRAUD ALERT",
            "attachments": [
                {
                    "color": "danger",
                    "fields": [
                        {"title": "User", "value": transaction_data["user"], "short": True},
                        {"title": "Card", "value": transaction_data["card"], "short": True},
                        {"title": "Amount", "value": f"${transaction_data['amount']}", "short": True},
                        {"title": "Merchant", "value": transaction_data["merchant_name"], "short": True},
                        {"title": "Location", "value": f"{transaction_data['merchant_city']}, {transaction_data['merchant_state']}", "short": True},
                        {"title": "Prediction", "value": prediction_result, "short": True}
                    ]
                }
            ]
        }
        
        response = requests.post(webhook_url, json=message)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error sending Slack notification: {str(e)}")
        return False

def main():
    st.title("Fraud Detection & Chat System")

    tab1, tab2 = st.tabs(["Fraud Detection", "Chat with Agent"])
    
    with tab1:
        st.header("Transaction Fraud Detection")
        encoder, classifier, node_mapping, scaler, label_encoders = load_model()
        
        if encoder is None:
            st.error("Failed to load model. Please check if fraud_gnn_model.pt exists.")
            return

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
                merchant_name = st.text_input("Merchant ID", value="3189517333335617109")
                merchant_city = st.text_input("Merchant City", value="Riverside")
                merchant_state = st.text_input("Merchant State", value="CA")
                zip_code = st.text_input("ZIP Code", value="92505.0")
            
            col3, col4 = st.columns(2)
            with col3:
                mcc = st.text_input("MCC Code", value="7538")
            with col4:
                errors = st.text_input("Transaction Errors", value="Insufficient Balance")

            st.subheader("Slack Notification Settings")
            webhook_url = st.text_input("Slack Webhook URL (optional)", 
                                       placeholder="https://hooks.slack.com/services/...")

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
                    "errors": errors
                }

                with st.spinner("Analyzing transaction..."):
                    result, confidence = predict_fraud(sample, encoder, classifier, 
                                                     node_mapping, scaler, label_encoders)
                
                if result is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if result == "Yes":
                            st.error(f"FRAUD DETECTED!")
                        else:
                            st.success(f"Transaction appears legitimate (Confidence: {confidence:.2%})")
                    
                    with col2:
                        st.metric("Prediction", result)

                    if result == "Yes" and webhook_url:
                        with st.spinner("Sending Slack notification..."):
                            success = send_slack_webhook(webhook_url, sample, result)
                            if success:
                                st.success("Slack notification sent successfully!")
                            else:
                                st.error("Failed to send Slack notification")
                    
                    with st.expander("Transaction Details"):
                        st.json(sample)
    
    with tab2:
        st.header("Chat with Agent")
        
        if not CHAT_AVAILABLE:
            st.error("""
            Chat functionality requires additional packages. Install them with:
            ```bash
            pip install llama_index chromadb
            ```
            """)
            return
        
        try:
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            collection_name = "transaction_logs"
            try:
                collection = chroma_client.get_collection(name=collection_name)
            except:
                collection = chroma_client.create_collection(name=collection_name)
            
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create or load index
            @st.cache_resource
            def load_chat_index():
                try:
                    index = VectorStoreIndex.from_vector_store(
                        vector_store=vector_store,
                        storage_context=storage_context
                    )
                    return index
                except:
                    index = VectorStoreIndex([], storage_context=storage_context)
                    return index
            
            index = load_chat_index()
            
            st.subheader("Query Transaction Logs")

            if st.button("Add Sample Transaction Logs"):
                sample_logs = [
                    "Transaction ID 12345: User user1 made a $500 purchase at Amazon on 2023-05-20",
                    "Transaction ID 12346: Suspicious activity detected for card card1 at gas station",
                    "Transaction ID 12347: Large transaction $2000 flagged for manual review",
                    "Transaction ID 12348: International transaction blocked for security reasons"
                ]
                
                documents = [Document(text=log) for log in sample_logs]
                index.insert_nodes(documents)
                st.success("Sample logs added to database!")
            
            # Query interface
            query = st.text_input("Ask about transaction logs:", 
                                 placeholder="e.g. 'Show me suspicious transactions' or 'What happened with user1?'")
            
            if query:
                with st.spinner("Searching logs..."):
                    query_engine = index.as_query_engine()
                    response = query_engine.query(query)
                    
                    st.subheader("Response:")
                    st.write(response.response)
                    
                    # Show source documents if available
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        with st.expander("Source Documents"):
                            for i, node in enumerate(response.source_nodes):
                                st.write(f"**Source {i+1}:**")
                                st.write(node.text)
                                st.write("---")
        
        except Exception as e:
            st.error(f"Error initializing chat system: {str(e)}")
            st.info("Make sure ChromaDB is properly installed and accessible.")

if __name__ == "__main__":
    main()
