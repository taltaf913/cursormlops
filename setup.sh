#!/bin/bash

# RAG Agentic AI Setup Script
echo "🚀 Setting up RAG Agentic AI System..."
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p chroma_db
mkdir -p sample_documents

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your Azure OpenAI credentials:"
    echo "   - AZURE_OPENAI_ENDPOINT"
    echo "   - AZURE_OPENAI_API_KEY"
    echo "   - AZURE_OPENAI_DEPLOYMENT_NAME"
else
    echo "✅ .env file already exists"
fi

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x start_backend.py
chmod +x start_frontend.py
chmod +x test_setup.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📚 Next steps:"
echo "1. Edit .env file with your Azure OpenAI credentials"
echo "2. Start the backend: python start_backend.py"
echo "3. Start the frontend: python start_frontend.py"
echo "4. Test the setup: python test_setup.py"
echo ""
echo "🌐 Access points:"
echo "   - Frontend: http://localhost:8501"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "🐳 For Docker deployment:"
echo "   docker-compose up -d"
echo ""
echo "Happy RAG-ing! 🤖✨"