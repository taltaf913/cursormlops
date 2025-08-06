import streamlit as st
import requests
import json
import os
from typing import List, Dict, Any
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="RAG Agentic AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

class RAGClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def query(self, query: str, top_k: int = 5, temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """Send query to RAG system"""
        try:
            payload = {
                "query": query,
                "top_k": top_k,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            response = requests.post(f"{self.base_url}/query", json=payload, timeout=30)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def upload_documents(self, files: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        """Upload documents to the RAG system"""
        try:
            files_data = []
            for file in files:
                files_data.append(("files", (file.name, file.getvalue(), file.type)))
            
            data = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
            
            response = requests.post(
                f"{self.base_url}/upload-documents",
                files=files_data,
                data=data,
                timeout=60
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def list_documents(self) -> Dict[str, Any]:
        """List all documents in the vector store"""
        try:
            response = requests.get(f"{self.base_url}/documents", timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document from the vector store"""
        try:
            response = requests.delete(f"{self.base_url}/documents/{document_id}", timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def clear_documents(self) -> Dict[str, Any]:
        """Clear all documents from the vector store"""
        try:
            response = requests.post(f"{self.base_url}/clear-documents", timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# Initialize client
rag_client = RAGClient(API_BASE_URL)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Agentic AI</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Configuration
        st.subheader("API Settings")
        api_url = st.text_input("API Base URL", value=API_BASE_URL, key="api_url")
        
        # Health check
        if st.button("🔍 Check API Health"):
            with st.spinner("Checking API health..."):
                health = rag_client.health_check()
                if health.get("status") == "healthy":
                    st.success("✅ API is healthy")
                else:
                    st.error(f"❌ API Error: {health.get('message', 'Unknown error')}")
        
        st.divider()
        
        # Query Parameters
        st.subheader("Query Parameters")
        top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        max_tokens = st.slider("Max Tokens", min_value=100, max_value=2000, value=1000, step=100)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Document Management", "📊 Analytics"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        document_management()
    
    with tab3:
        analytics_interface()

def chat_interface():
    st.markdown('<h2 class="sub-header">💬 Chat with RAG AI</h2>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = rag_client.query(
                    query=prompt,
                    top_k=st.session_state.get("top_k", 5),
                    temperature=st.session_state.get("temperature", 0.7),
                    max_tokens=st.session_state.get("max_tokens", 1000)
                )
                
                if "error" in response:
                    st.error(f"Error: {response['error']}")
                else:
                    # Display answer
                    st.markdown(response["answer"])
                    
                    # Display sources if available
                    if response.get("sources"):
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(response["sources"]):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(f"*File: {source['metadata'].get('filename', 'Unknown')}*")
                                st.markdown(f"*Content: {source['content'][:200]}...*")
                                st.divider()
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

def document_management():
    st.markdown('<h2 class="sub-header">📁 Document Management</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['txt', 'md', 'html', 'pdf'],
            accept_multiple_files=True,
            help="Supported formats: TXT, MD, HTML, PDF"
        )
        
        if uploaded_files:
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
            with col1_2:
                chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
            
            if st.button("🚀 Upload and Process"):
                with st.spinner("Processing documents..."):
                    response = rag_client.upload_documents(
                        files=uploaded_files,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    if "error" in response:
                        st.error(f"Upload failed: {response['error']}")
                    else:
                        st.success(f"✅ {response['message']}")
                        st.info(f"📊 Processed {response['document_count']} documents")
                        for doc_name in response['document_names']:
                            st.write(f"• {doc_name}")
    
    with col2:
        st.subheader("📋 Document List")
        
        if st.button("🔄 Refresh Documents"):
            with st.spinner("Loading documents..."):
                documents_response = rag_client.list_documents()
                
                if "error" in documents_response:
                    st.error(f"Error loading documents: {documents_response['error']}")
                else:
                    documents = documents_response.get("documents", [])
                    
                    if not documents:
                        st.info("No documents found")
                    else:
                        st.write(f"📚 Total documents: {len(documents)}")
                        
                        for doc in documents:
                            with st.expander(f"📄 {doc.get('metadata', {}).get('filename', 'Unknown')}"):
                                st.write(f"**ID:** {doc.get('id', 'N/A')}")
                                st.write(f"**Type:** {doc.get('metadata', {}).get('file_type', 'N/A')}")
                                st.write(f"**Preview:** {doc.get('content_preview', 'N/A')}")
                                
                                if st.button(f"🗑️ Delete", key=f"delete_{doc.get('id')}"):
                                    delete_response = rag_client.delete_document(doc.get('id'))
                                    if "error" in delete_response:
                                        st.error(f"Delete failed: {delete_response['error']}")
                                    else:
                                        st.success("Document deleted successfully")
                                        st.rerun()
    
    # Clear all documents
    st.divider()
    st.subheader("⚠️ Danger Zone")
    
    if st.button("🗑️ Clear All Documents", type="secondary"):
        if st.checkbox("I understand this will delete all documents"):
            with st.spinner("Clearing documents..."):
                response = rag_client.clear_documents()
                if "error" in response:
                    st.error(f"Clear failed: {response['error']}")
                else:
                    st.success("All documents cleared successfully")
                    st.rerun()

def analytics_interface():
    st.markdown('<h2 class="sub-header">📊 Analytics & Insights</h2>', unsafe_allow_html=True)
    
    # Get system stats
    with st.spinner("Loading analytics..."):
        health = rag_client.health_check()
        documents_response = rag_client.list_documents()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if health.get("status") == "healthy":
            st.metric("System Status", "🟢 Healthy")
        else:
            st.metric("System Status", "🔴 Unhealthy")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if "error" not in documents_response:
            doc_count = len(documents_response.get("documents", []))
            st.metric("Total Documents", doc_count)
        else:
            st.metric("Total Documents", "Error")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("API Version", "1.0.0")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed analytics
    st.subheader("📈 Detailed Analytics")
    
    if "error" not in documents_response:
        documents = documents_response.get("documents", [])
        
        if documents:
            # File type distribution
            file_types = {}
            for doc in documents:
                file_type = doc.get('metadata', {}).get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 File Type Distribution")
                for file_type, count in file_types.items():
                    st.write(f"• {file_type}: {count} documents")
            
            with col2:
                st.subheader("📋 Recent Documents")
                recent_docs = documents[-5:]  # Show last 5 documents
                for doc in recent_docs:
                    st.write(f"• {doc.get('metadata', {}).get('filename', 'Unknown')}")
        else:
            st.info("No documents found. Upload some documents to see analytics.")
    else:
        st.error(f"Error loading analytics: {documents_response['error']}")

if __name__ == "__main__":
    main()