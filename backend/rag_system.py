import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with Azure OpenAI and ChromaDB"""
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.chroma_persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        
        if not all([self.azure_openai_endpoint, self.azure_openai_api_key, self.azure_openai_deployment_name]):
            raise ValueError("Missing required Azure OpenAI configuration")
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",  # You may need to adjust this
            openai_api_version=self.azure_openai_api_version,
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_api_key,
        )
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=self.azure_openai_deployment_name,
            openai_api_version=self.azure_openai_api_version,
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_api_key,
            temperature=0.7,
            max_tokens=1000,
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name="rag_documents",
            embedding_function=self.embeddings,
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize retrieval QA chain
        self.qa_chain = self._create_qa_chain()
        
        logger.info("RAG system initialized successfully")
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create the QA chain with custom prompt"""
        prompt_template = """You are a helpful AI assistant that answers questions based on the provided context. 
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        try:
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vector_store.add_documents(texts)
            logger.info(f"Added {len(texts)} document chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def query(self, query: str, top_k: int = 5, temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            # Update LLM parameters
            self.llm.temperature = temperature
            self.llm.max_tokens = max_tokens
            
            # Update retriever
            self.qa_chain.retriever.search_kwargs["k"] = top_k
            
            # Get response
            result = self.qa_chain({"query": query})
            
            # Extract sources
            sources = []
            if result.get("source_documents"):
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)
            
            return {
                "answer": result["result"],
                "sources": sources,
                "metadata": {
                    "query": query,
                    "top_k": top_k,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def update_vector_store(self) -> None:
        """Update the vector store (called after adding new documents)"""
        try:
            # Recreate the QA chain with updated vector store
            self.qa_chain = self._create_qa_chain()
            logger.info("Vector store updated successfully")
        except Exception as e:
            logger.error(f"Error updating vector store: {e}")
            raise
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the vector store"""
        try:
            collection = self.chroma_client.get_collection("rag_documents")
            results = collection.get()
            
            documents = []
            if results.get("metadatas"):
                for i, metadata in enumerate(results["metadatas"]):
                    doc_info = {
                        "id": results["ids"][i] if results.get("ids") else f"doc_{i}",
                        "metadata": metadata,
                        "content_preview": results["documents"][i][:100] + "..." if results.get("documents") else ""
                    }
                    documents.append(doc_info)
            
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a specific document from the vector store"""
        try:
            collection = self.chroma_client.get_collection("rag_documents")
            collection.delete(ids=[document_id])
            logger.info(f"Deleted document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def clear_documents(self) -> None:
        """Clear all documents from the vector store"""
        try:
            collection = self.chroma_client.get_collection("rag_documents")
            collection.delete(where={})
            logger.info("Cleared all documents from vector store")
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        try:
            collection = self.chroma_client.get_collection("rag_documents")
            count = collection.count()
            return {
                "total_documents": count,
                "collection_name": "rag_documents",
                "persist_directory": self.chroma_persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise