"""Tests for the QA Analysis Service."""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from app.models.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    Conversation,
    Message,
    QuestionRating,
    ConversationThread,
    AIAnswers,
    AIAnswer,
    RAGResponse,
    KBChunk
)
from app.services.analysis_service import AnalysisService
from app.services.rag_client import RAGClient
from app.services.llm_client import LLMClient


@pytest.fixture
def mock_rag_client():
    """Create a mock RAG client."""
    client = Mock(spec=RAGClient)
    # Mock response for retrieve_chunks
    client.retrieve_chunks.return_value = RAGResponse(
        question="How do I filter unpaid invoices?",
        chunks=[
            KBChunk(
                content="On the Invoices page, use the Status dropdown to choose Unpaid.",
                source="billing/invoices.md",
                confidence=0.95
            )
        ],
        formatted_chunks=[
            "On the Invoices page, use the Status dropdown to choose Unpaid. (source: billing/invoices.md)"
        ]
    )
    return client


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock(spec=LLMClient)
    
    # Mock Stage 1 response (thread segmentation)
    client.chat_completion_json.side_effect = [
        # First call: Stage 1 segmentation
        {
            "threads": [
                {
                    "qid": "T1",
                    "question": "How do I filter unpaid invoices?",
                    "answer": "Use the Status dropdown and select Unpaid."
                }
            ]
        },
        # Second call: Stage 2 AI answers
        {
            "ai_suggested_answer": {
                "answer": "Select Unpaid in Status filter.",
                "context": "[1]"
            },
            "ai_detailed_answer": {
                "answer": "On the Invoices page, use the Status dropdown to choose Unpaid to filter unpaid invoices.",
                "context": "[1]"
            }
        },
        # Third call: Stage 3 scoring
        {
            "ai_score": 4.5,
            "ai_rational": "Agent provided correct navigation but could be more specific about the 'Unpaid' option.",
            "kb_verify": ["On the Invoices page, use the Status dropdown to choose Unpaid. (source: billing/invoices.md)"]
        }
    ]
    
    return client


@pytest.fixture
def analysis_service(mock_rag_client, mock_llm_client):
    """Create an analysis service with mocked dependencies."""
    return AnalysisService(rag_client=mock_rag_client, llm_client=mock_llm_client)


@pytest.fixture
def sample_conversation():
    """Create a sample conversation for testing."""
    return Conversation(
        id=12345,
        type="chat",
        messages=[
            Message(
                id="1",
                role="customer",
                content="How do I filter unpaid invoices?",
                timestamp=datetime.utcnow()
            ),
            Message(
                id="2",
                role="agent",
                content="Use the Status dropdown and select Unpaid.",
                timestamp=datetime.utcnow()
            )
        ]
    )


def test_stage1_segment_conversation(analysis_service, sample_conversation):
    """Test Stage 1: Conversation segmentation."""
    threads = analysis_service._stage1_segment_conversation(sample_conversation)
    
    assert len(threads) == 1
    assert threads[0].question == "How do I filter unpaid invoices?"
    assert threads[0].answer == "Use the Status dropdown and select Unpaid."
    assert threads[0].qid == "T1"


def test_stage2_generate_ai_answers(analysis_service):
    """Test Stage 2: AI answer generation."""
    question = "How do I filter unpaid invoices?"
    kb_id = "kb_001"
    
    kb_chunks, ai_answers = analysis_service._stage2_generate_ai_answers(question, kb_id)
    
    # Check KB chunks
    assert len(kb_chunks) == 1
    assert "Status dropdown" in kb_chunks[0]
    
    # Check AI answers
    assert isinstance(ai_answers, AIAnswers)
    assert ai_answers.suggested.answer == "Select Unpaid in Status filter."
    assert ai_answers.detailed.answer == "On the Invoices page, use the Status dropdown to choose Unpaid to filter unpaid invoices."


def test_stage3_score_agent_answer(analysis_service):
    """Test Stage 3: Agent answer scoring."""
    question = "How do I filter unpaid invoices?"
    agent_answer = "Use the Status dropdown and select Unpaid."
    ai_answers = AIAnswers(
        short=AIAnswer(answer="Select Unpaid in Status filter.", context="[1]"),
        long=AIAnswer(answer="On the Invoices page, use the Status dropdown to choose Unpaid.", context="[1]")
    )
    kb_chunks = ["On the Invoices page, use the Status dropdown to choose Unpaid. (source: billing/invoices.md)"]
    
    rating = analysis_service._stage3_score_agent_answer(question, agent_answer, ai_answers, kb_chunks)
    
    assert rating.aiScore == 4.5
    assert "correct navigation" in rating.aiRationale
    assert len(rating.kbVerify) == 1


def test_analyze_conversation_full_flow(analysis_service, sample_conversation):
    """Test the complete analysis flow."""
    request = AnalysisRequest(
        conversation=sample_conversation,
        integratedKbId="kb_001"
    )
    
    response = analysis_service.analyze_conversation(request)
    
    assert isinstance(response, AnalysisResponse)
    assert response.conversationId == 12345
    assert response.conversationType == "chat"
    assert len(response.questionRatings) == 1
    
    rating = response.questionRatings[0]
    assert rating.aiRewrittenQuestion == "How do I filter unpaid invoices?"
    assert rating.agentAnswer == "Use the Status dropdown and select Unpaid."
    assert rating.aiScore == 4.5
    assert response.overallAccuracy == 4.5


def test_fallback_when_llm_fails(mock_rag_client):
    """Test fallback behavior when LLM client fails."""
    # Create a failing LLM client
    failing_llm = Mock(spec=LLMClient)
    failing_llm.chat_completion_json.side_effect = Exception("LLM API error")
    
    service = AnalysisService(rag_client=mock_rag_client, llm_client=failing_llm)
    
    conversation = Conversation(
        id=1,
        type="chat",
        messages=[
            Message(id="1", role="customer", content="Test question?", timestamp=datetime.utcnow()),
            Message(id="2", role="agent", content="Test answer.", timestamp=datetime.utcnow())
        ]
    )
    
    # Should fall back to simple extraction
    threads = service._stage1_segment_conversation(conversation)
    assert len(threads) == 1
    assert threads[0].question == "Test question?"


def test_empty_conversation():
    """Test handling of empty conversations."""
    service = AnalysisService()
    
    empty_conv = Conversation(
        id=1,
        type="chat",
        messages=[]
    )
    
    request = AnalysisRequest(
        conversation=empty_conv,
        integratedKbId="kb_001"
    )
    
    response = service.analyze_conversation(request)
    assert len(response.questionRatings) == 0
    assert response.overallAccuracy == 0.0


def test_out_of_scope_scoring():
    """Test handling of out-of-scope answers (score = -1)."""
    mock_llm = Mock(spec=LLMClient)
    mock_llm.chat_completion_json.side_effect = [
        {"threads": [{"qid": "T1", "question": "Q1", "answer": "A1"}]},
                    {"ai_suggested_answer": {"answer": "AI1", "context": ""}, "ai_detailed_answer": {"answer": "AI1 Long", "context": ""}},
        {"ai_score": -1, "ai_rational": "Out of scope", "kb_verify": []},
        {"threads": [{"qid": "T2", "question": "Q2", "answer": "A2"}]},
                    {"ai_suggested_answer": {"answer": "AI2", "context": ""}, "ai_detailed_answer": {"answer": "AI2 Long", "context": ""}},
        {"ai_score": 5, "ai_rational": "Perfect", "kb_verify": []}
    ]
    
    service = AnalysisService(llm_client=mock_llm)
    
    # Create conversation with multiple Q&A pairs
    conversation = Conversation(
        id=1,
        type="chat",
        messages=[
            Message(id="1", role="customer", content="Q1", timestamp=datetime.utcnow()),
            Message(id="2", role="agent", content="A1", timestamp=datetime.utcnow()),
            Message(id="3", role="customer", content="Q2", timestamp=datetime.utcnow()),
            Message(id="4", role="agent", content="A2", timestamp=datetime.utcnow())
        ]
    )
    
    # Need to mock multiple analyze calls
    with patch.object(service, '_stage1_segment_conversation') as mock_stage1:
        mock_stage1.return_value = [
            ConversationThread(qid="T1", question="Q1", answer="A1"),
            ConversationThread(qid="T2", question="Q2", answer="A2")
        ]
        
        request = AnalysisRequest(conversation=conversation, integratedKbId="kb_001")
        response = service.analyze_conversation(request)
        
        # Overall accuracy should only include score >= 0 (exclude -1)
        assert response.overallAccuracy == 5.0  # Only the score of 5 is counted 