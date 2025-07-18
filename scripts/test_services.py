#!/usr/bin/env python3
"""Test script to verify all services are working correctly."""
import requests
import json
import time
import sys
import os


def test_service_health(url: str, service_name: str) -> bool:
    """Test if a service is healthy.
    
    Args:
        url: Service health endpoint URL.
        service_name: Name of the service for logging.
        
    Returns:
        bool: True if service is healthy.
    """
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name} is healthy")
            return True
        else:
            print(f"❌ {service_name} returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {service_name} is not responding: {e}")
        return False


def test_chat_data_service() -> bool:
    """Test the Chat Data Service."""
    print("\n🧪 Testing Chat Data Service...")
    
    # Test health
    if not test_service_health("http://localhost:8001/", "Chat Data Service"):
        return False
    
    # Test getting transcripts
    try:
        response = requests.get("http://localhost:8001/transcripts", timeout=10)
        if response.status_code == 200:
            transcripts = response.json()
            print(f"✅ Retrieved {len(transcripts)} sample transcripts")
            return True
        else:
            print(f"❌ Failed to get transcripts: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing Chat Data Service: {e}")
        return False


def test_rag_service() -> bool:
    """Test the RAG Service v2."""
    print("\n🧪 Testing RAG Service v2...")
    
    # Test health
    if not test_service_health("http://localhost:8002/health", "RAG Service"):
        return False
    
    # Test retrieving chunks
    try:
        test_request = {
            "question": "How do I filter unpaid invoices?",
            "k": 6
        }
        
        response = requests.post(
            "http://localhost:8002/retrieve-chunks", 
            json=test_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retrieved {len(result['chunks'])} KB chunks")
            if result['chunks']:
                print(f"✅ Top chunk: {result['chunks'][0]['content'][:50]}...")
                print(f"✅ Confidence: {result['chunks'][0]['confidence']}")
            return True
        else:
            print(f"❌ Failed to retrieve chunks: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing RAG Service: {e}")
        return False


def test_qa_analysis_service() -> bool:
    """Test the QA Analysis Service with 3-stage algorithm."""
    print("\n🧪 Testing QA Analysis Service (3-Stage Algorithm)...")
    
    # Test health
    if not test_service_health("http://localhost:8000/health", "QA Analysis Service"):
        return False
    
    # Test analysis with sample conversation
    try:
        test_request = {
            "conversation": {
                "id": 12345,
                "type": "chat",
                "messages": [
                    {
                        "id": "msg_001",
                        "role": "customer",
                        "content": "Hey, I just saw an alert about unpaid invoices—where do I check them?",
                        "timestamp": "2024-01-15T08:50:00Z"
                    },
                    {
                        "id": "msg_002",
                        "role": "agent",
                        "content": "Sure—head over to Billing → Invoices and you'll see all your bills.",
                        "timestamp": "2024-01-15T08:51:00Z"
                    },
                    {
                        "id": "msg_003",
                        "role": "customer",
                        "content": "In 'Invoices' I only see paid ones. How do I filter unpaid?",
                        "timestamp": "2024-01-15T08:53:00Z"
                    },
                    {
                        "id": "msg_004",
                        "role": "agent",
                        "content": "There's a 'Status' dropdown up top—select 'Unpaid.'",
                        "timestamp": "2024-01-15T08:54:00Z"
                    },
                    {
                        "id": "msg_005",
                        "role": "customer",
                        "content": "I need to disable 2FA just this once—how?",
                        "timestamp": "2024-01-15T09:12:00Z"
                    },
                    {
                        "id": "msg_006",
                        "role": "agent",
                        "content": "In Security Settings there's a 'Disable 2FA' toggle.",
                        "timestamp": "2024-01-15T09:13:00Z"
                    },
                    {
                        "id": "msg_007",
                        "role": "customer",
                        "content": "I don't see that toggle in Security Settings.",
                        "timestamp": "2024-01-15T09:15:00Z"
                    },
                    {
                        "id": "msg_008",
                        "role": "agent",
                        "content": "You might not have permission—ask your admin to disable it.",
                        "timestamp": "2024-01-15T09:16:00Z"
                    }
                ]
            },
            "integratedKbId": "kb_12345"
        }
        
        response = requests.post(
            "http://localhost:8000/aiqa/analysis/analyze", 
            json=test_request,
            timeout=60  # Increased timeout for LLM calls
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis completed for conversation {result['conversationId']}")
            print(f"✅ Conversation type: {result['conversationType']}")
            print(f"✅ Number of question ratings: {len(result['questionRatings'])}")
            print(f"✅ Overall accuracy: {result['overallAccuracy']}/5.0")
            
            print("\n--- 3-Stage Analysis Details ---")
            
            # Show details for each question rating
            if result['questionRatings']:
                for i, rating in enumerate(result['questionRatings']):
                    print(f"\n[Question Rating #{i+1}]")
                    print(f"  🔄 Stage 1 - Rewritten Question: {rating['aiRewrittenQuestion']}")
                    print(f"  👤 Agent's Answer: {rating['agentAnswer']}")
                    print(f"  🤖 Stage 2 - AI Suggested Answer: {rating['aiSuggestedAnswer']}")
                    print(f"  📊 Stage 3 - AI Score: {rating['aiScore']}/5.0")
                    print(f"  📝 Stage 3 - AI Rationale: {rating['aiRationale']}")
            else:
                print("No question ratings were generated from the conversation.")
            
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing QA Analysis Service: {e}")
        return False


def main():
    """Run all service tests."""
    print("🧪 AI Quality Assurance Service Test Suite (3-Stage Algorithm)")
    print("=" * 50)
    
    print("Waiting for services to start up...")
    time.sleep(2)
    
    all_passed = True
    
    # Test each service
    services = [
        ("Chat Data Service", test_chat_data_service),
        ("RAG Service v2", test_rag_service),
        ("QA Analysis Service", test_qa_analysis_service)
    ]
    
    for service_name, test_function in services:
        if not test_function():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Services are working correctly.")
        sys.exit(0)
    else:
        print("💥 Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main() 