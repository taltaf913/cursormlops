#!/usr/bin/env python3
"""
Startup script for the RAG Agentic AI Streamlit frontend
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Start the Streamlit frontend"""
    port = int(os.getenv("STREAMLIT_PORT", 8501))
    
    print(f"🎨 Starting RAG Agentic AI Frontend...")
    print(f"📍 Streamlit will be available at: http://localhost:{port}")
    print(f"🔗 Backend API: http://localhost:8000")
    print("=" * 50)
    
    # Start Streamlit
    subprocess.run([
        "streamlit", "run", "frontend/streamlit_app.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    main()