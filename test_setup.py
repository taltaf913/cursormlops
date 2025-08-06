#!/usr/bin/env python3
"""
Test script to verify RAG Agentic AI system setup
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment_variables():
    """Test if required environment variables are set"""
    print("🔍 Testing environment variables...")
    
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY", 
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        print("✅ All required environment variables are set")
        return True

def test_backend_health():
    """Test backend health endpoint"""
    print("\n🔍 Testing backend health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend is healthy: {data}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running or not accessible")
        return False
    except Exception as e:
        print(f"❌ Backend health check error: {e}")
        return False

def test_frontend_access():
    """Test frontend accessibility"""
    print("\n🔍 Testing frontend access...")
    
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            return True
        else:
            print(f"❌ Frontend access failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Frontend is not running or not accessible")
        return False
    except Exception as e:
        print(f"❌ Frontend access error: {e}")
        return False

def test_rag_query():
    """Test RAG query functionality"""
    print("\n🔍 Testing RAG query...")
    
    try:
        payload = {
            "query": "Hello, this is a test query",
            "top_k": 3,
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        response = requests.post(
            "http://localhost:8000/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ RAG query successful")
            print(f"📝 Response: {data.get('answer', 'No answer')[:100]}...")
            return True
        else:
            print(f"❌ RAG query failed: {response.status_code}")
            print(f"📝 Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ RAG query error: {e}")
        return False

def test_api_documentation():
    """Test API documentation access"""
    print("\n🔍 Testing API documentation...")
    
    try:
        response = requests.get("http://localhost:8000/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API documentation is accessible")
            return True
        else:
            print(f"❌ API documentation access failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API documentation error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 RAG Agentic AI System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Backend Health", test_backend_health),
        ("Frontend Access", test_frontend_access),
        ("RAG Query", test_rag_query),
        ("API Documentation", test_api_documentation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG system is ready to use.")
        print("\n📚 Next steps:")
        print("1. Open http://localhost:8501 in your browser")
        print("2. Upload some documents")
        print("3. Start chatting with your RAG AI!")
    else:
        print("⚠️  Some tests failed. Please check the configuration and try again.")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure both backend and frontend are running")
        print("2. Check your Azure OpenAI configuration")
        print("3. Verify all environment variables are set correctly")

if __name__ == "__main__":
    main()