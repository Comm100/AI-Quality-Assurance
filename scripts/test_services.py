#!/usr/bin/env python3
"""Test script to verify all services are working correctly."""
import requests
import json
import time
import sys


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
    """Test the RAG Service."""
    print("\n🧪 Testing RAG Service...")
    
    # Test health
    if not test_service_health("http://localhost:8002/health", "RAG Service"):
        return False
    
    # Test generating answer
    try:
        test_request = {
            "question": "How do I enable email notifications?",
            "context": "User needs help with settings"
        }
        
        response = requests.post(
            "http://localhost:8002/generate-answer", 
            json=test_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generated answer: {result['answer'][:50]}...")
            print(f"✅ Confidence: {result['confidence']}")
            return True
        else:
            print(f"❌ Failed to generate answer: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing RAG Service: {e}")
        return False


def test_qa_analysis_service() -> bool:
    """Test the QA Analysis Service."""
    print("\n🧪 Testing QA Analysis Service...")
    
    # Test health
    if not test_service_health("http://localhost:8000/health", "QA Analysis Service"):
        return False
    
    # Test analysis with sample transcript
    try:
        sample_transcript = """
Agent: Hello! How can I help you today?
Customer: Hi, I'm having trouble with email notifications.
Agent: I'd be happy to help with that. What specific issue are you experiencing?
Customer: I can't find where to enable notifications.
Agent: Go to Settings > Notifications > Email Settings and check the notification boxes.
Customer: That worked! Thank you.
        """.strip()
        
        test_request = {
            "transcript": sample_transcript,
            "transcript_id": "test_001",
            "metadata": {"test": True}
        }
        
        response = requests.post(
            "http://localhost:8000/analyze", 
            json=test_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis completed for {result['total_questions']} questions")
            print(f"✅ Overall score: {result['overall_score']:.2f}")
            print(f"✅ Processing time: {result['processing_time_ms']}ms")
            
            # Show first Q&A pair details
            if result['qa_pairs']:
                first_qa = result['qa_pairs'][0]
                print(f"✅ First Q&A accuracy: {first_qa['accuracy_score']:.2f}")
            
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
    print("🧪 AI Quality Assurance Service Test Suite")
    print("=" * 50)
    
    print("Waiting for services to start up...")
    time.sleep(2)
    
    all_passed = True
    
    # Test each service
    services = [
        ("Chat Data Service", test_chat_data_service),
        ("RAG Service", test_rag_service),
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