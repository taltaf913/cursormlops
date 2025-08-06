# 🤖 RAG Agentic AI System

A comprehensive Retrieval-Augmented Generation (RAG) system built with FastAPI backend and Streamlit frontend, powered by Azure OpenAI.

## 🚀 Features

- **🔍 Intelligent Document Processing**: Support for TXT, MD, HTML, and PDF files
- **🧠 Azure OpenAI Integration**: Leverage Azure's powerful language models
- **📊 Vector Database**: ChromaDB for efficient document storage and retrieval
- **💬 Interactive Chat Interface**: Modern Streamlit UI with chat history
- **📁 Document Management**: Upload, view, and manage your knowledge base
- **📈 Analytics Dashboard**: Monitor system health and document statistics
- **🔧 Configurable Parameters**: Adjust chunk size, temperature, and more

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   Azure OpenAI  │
│   Frontend      │◄──►│   Backend       │◄──►│   Services      │
│   (Port 8501)   │    │   (Port 8000)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   ChromaDB      │
                       │   Vector Store  │
                       └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- Azure OpenAI account with API access
- Azure OpenAI deployment for text generation and embeddings

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-agentic-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your Azure OpenAI credentials:
   ```env
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name_here
   ```

## 🚀 Quick Start

### Option 1: Start Backend Only
```bash
python start_backend.py
```

### Option 2: Start Frontend Only
```bash
python start_frontend.py
```

### Option 3: Start Both (Recommended)
```bash
# Terminal 1 - Backend
python start_backend.py

# Terminal 2 - Frontend
python start_frontend.py
```

## 📖 Usage

### 1. Access the Application
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 2. Upload Documents
1. Navigate to the "Document Management" tab
2. Upload your documents (TXT, MD, HTML, PDF)
3. Configure chunk size and overlap
4. Click "Upload and Process"

### 3. Start Chatting
1. Go to the "Chat" tab
2. Ask questions about your uploaded documents
3. View sources and references

### 4. Monitor Analytics
- Check system health
- View document statistics
- Monitor file type distribution

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | Required |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Required |
| `AZURE_OPENAI_API_VERSION` | API version | 2024-02-15-preview |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Model deployment name | Required |
| `CHROMA_PERSIST_DIRECTORY` | Vector DB storage path | ./chroma_db |
| `API_HOST` | Backend host | 0.0.0.0 |
| `API_PORT` | Backend port | 8000 |
| `STREAMLIT_PORT` | Frontend port | 8501 |

### Query Parameters

- **Top K**: Number of relevant documents to retrieve (1-10)
- **Temperature**: Response creativity (0.0-2.0)
- **Max Tokens**: Maximum response length (100-2000)

## 📚 API Endpoints

### Core Endpoints
- `GET /` - API status
- `GET /health` - Health check
- `POST /query` - Query the RAG system
- `POST /upload-documents` - Upload documents
- `GET /documents` - List documents
- `DELETE /documents/{id}` - Delete document
- `POST /clear-documents` - Clear all documents

### Example API Usage
```bash
# Query the system
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 5}'

# Upload documents
curl -X POST "http://localhost:8000/upload-documents" \
  -F "files=@document.txt" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200"
```

## 🏗️ Project Structure

```
rag-agentic-ai/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── rag_system.py        # Core RAG functionality
│   └── document_processor.py # Document processing
├── frontend/
│   └── streamlit_app.py     # Streamlit UI
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
├── start_backend.py        # Backend startup script
├── start_frontend.py       # Frontend startup script
└── README.md               # This file
```

## 🔍 Troubleshooting

### Common Issues

1. **Azure OpenAI Connection Error**
   - Verify your endpoint and API key
   - Check deployment name exists
   - Ensure API version is correct

2. **Document Upload Fails**
   - Check file format is supported
   - Verify file size isn't too large
   - Ensure backend is running

3. **Chat Not Working**
   - Verify backend is healthy
   - Check documents are uploaded
   - Review API logs for errors

### Logs
- Backend logs: Check terminal output
- Frontend logs: Check browser console
- Vector DB: Check `./chroma_db` directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Azure OpenAI for powerful language models
- LangChain for RAG framework
- ChromaDB for vector storage
- FastAPI for robust API framework
- Streamlit for beautiful UI components

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review API documentation at `/docs`

---

**Happy RAG-ing! 🤖✨**