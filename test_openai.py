#!/usr/bin/env python3
"""Test script to verify OpenAI API connectivity."""

import sys
import os

# Add the qa_analysis_service to the path
sys.path.append('qa_analysis_service')

try:
    from app.config import settings
    from app.services.llm_client import LLMClient, LLMClientError
    
    print("🧪 Testing OpenAI API Configuration")
    print("=" * 50)
    
    # Check configuration
    print(f"API Key present: {bool(settings.openai_api_key and len(settings.openai_api_key) > 10)}")
    print(f"Model: {settings.openai_model}")
    print(f"Temperature: {settings.openai_temperature}")
    
    if not settings.openai_api_key or len(settings.openai_api_key) < 10:
        print("❌ ERROR: OpenAI API key not set or too short")
        sys.exit(1)
    
    # Test LLM client
    print(f"\n🔧 Testing LLM Client...")
    try:
        llm_client = LLMClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.openai_temperature
        )
        print("✅ LLM Client created successfully")
        
        # Test simple completion
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond with JSON only."},
            {"role": "user", "content": "Return this JSON: {\"test\": \"success\", \"status\": \"working\"}"}
        ]
        
        print("🚀 Testing API call...")
        response = llm_client.chat_completion_json(test_messages)
        print(f"✅ API Response: {response}")
        
        if response.get("test") == "success":
            print("🎉 OpenAI API is working correctly!")
        else:
            print("⚠️  API responded but with unexpected format")
            
    except LLMClientError as e:
        print(f"❌ LLM Client Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the project root and dependencies are installed")
    sys.exit(1) 