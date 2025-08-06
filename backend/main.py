from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uvicorn
from dotenv import load_dotenv
import logging

from rag_system import RAGSystem
from document_processor import DocumentProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Agentic AI API",
    description="Retrieval-Augmented Generation API using Azure OpenAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = None
document_processor = None

@app.on_event("startup")
async def startup_event():
    global rag_system, document_processor
    try:
        rag_system = RAGSystem()
        document_processor = DocumentProcessor()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    temperature: float = 0.7
    max_tokens: int = 1000

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class DocumentUploadResponse(BaseModel):
    message: str
    document_count: int
    document_names: List[str]

@app.get("/")
async def root():
    return {"message": "RAG Agentic AI API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "rag_system_ready": rag_system is not None}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.query(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-documents", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """
    Upload and process documents for the RAG system
    """
    if not document_processor:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    try:
        document_names = []
        for file in files:
            if file.filename:
                content = await file.read()
                document_processor.process_document(
                    content=content.decode('utf-8'),
                    filename=file.filename,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                document_names.append(file.filename)
        
        # Update the RAG system with new documents
        if rag_system:
            rag_system.update_vector_store()
        
        return DocumentUploadResponse(
            message=f"Successfully processed {len(document_names)} documents",
            document_count=len(document_names),
            document_names=document_names
        )
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """
    List all documents in the vector store
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        documents = rag_system.list_documents()
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a specific document from the vector store
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        success = rag_system.delete_document(document_id)
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-documents")
async def clear_all_documents():
    """
    Clear all documents from the vector store
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        rag_system.clear_documents()
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host=host, port=port)