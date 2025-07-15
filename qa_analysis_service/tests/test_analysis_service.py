"""Tests for the QA Analysis Service."""
import pytest
from unittest.mock import Mock, patch

from app.models.analysis import AnalysisRequest, RAGResponse
from app.services.analysis_service import AnalysisService
from app.services.rag_client import RAGClient


class TestAnalysisService:
    """Test cases for AnalysisService."""
    
    @pytest.fixture
    def mock_rag_client(self):
        """Create a mock RAG client."""
        mock_client = Mock(spec=RAGClient)
        mock_client.generate_answer.return_value = RAGResponse(
            question="Test question",
            answer="This is a test reference answer.",
            confidence=0.9,
            sources=["Test Source"]
        )
        return mock_client
    
    @pytest.fixture
    def analysis_service(self, mock_rag_client):
        """Create an analysis service with mocked dependencies."""
        return AnalysisService(rag_client=mock_rag_client)
    
    @pytest.fixture
    def sample_transcript(self):
        """Sample chat transcript for testing."""
        return """
Agent: Hello! How can I help you today?
Customer: Hi, I'm having trouble with email notifications.
Agent: I'd be happy to help with that. What specific issue are you experiencing?
Customer: I can't find where to enable notifications.
Agent: Go to Settings > Notifications > Email Settings and check the notification boxes.
Customer: That worked! Thank you.
Agent: You're welcome! Anything else?
Customer: No, that's all.
Agent: Have a great day!
        """.strip()
    
    def test_segment_transcript_success(self, analysis_service, sample_transcript):
        """Test successful transcript segmentation."""
        # Test the private method directly for unit testing
        qa_pairs = analysis_service._segment_transcript(sample_transcript)
        
        # Should extract Q&A pairs
        assert len(qa_pairs) >= 1
        
        # Check first Q&A pair
        question, answer = qa_pairs[0]
        assert "trouble with email notifications" in question
        assert "help with that" in answer
    
    def test_is_likely_question(self, analysis_service):
        """Test question identification logic."""
        # These should be identified as questions
        assert analysis_service._is_likely_question("How do I enable notifications?")
        assert analysis_service._is_likely_question("I'm having trouble with settings")
        assert analysis_service._is_likely_question("Can you help me with this issue?")
        assert analysis_service._is_likely_question("What should I do next?")
        
        # These should NOT be identified as questions
        assert not analysis_service._is_likely_question("Hi")
        assert not analysis_service._is_likely_question("Thank you")
        assert not analysis_service._is_likely_question("Goodbye")
    
    def test_analyze_transcript_success(self, analysis_service, sample_transcript):
        """Test successful transcript analysis."""
        request = AnalysisRequest(
            transcript=sample_transcript,
            transcript_id="test_001"
        )
        
        result = analysis_service.analyze_transcript(request)
        
        # Verify response structure
        assert result.transcript_id == "test_001"
        assert result.total_questions >= 1
        assert len(result.qa_pairs) == result.total_questions
        assert 0.0 <= result.overall_score <= 1.0
        assert result.processing_time_ms is not None
        
        # Verify Q&A pairs
        for qa_pair in result.qa_pairs:
            assert qa_pair.question
            assert qa_pair.agent_answer
            assert qa_pair.reference_answer
            assert 0.0 <= qa_pair.accuracy_score <= 1.0
            assert 0.0 <= qa_pair.confidence_score <= 1.0
    
    def test_analyze_transcript_empty_input(self, analysis_service):
        """Test analysis with empty transcript."""
        request = AnalysisRequest(
            transcript="",
            transcript_id="test_empty"
        )
        
        result = analysis_service.analyze_transcript(request)
        
        # Should handle empty input gracefully
        assert result.total_questions == 0
        assert len(result.qa_pairs) == 0
        assert result.overall_score == 0.0
    
    def test_assess_accuracy(self, analysis_service):
        """Test accuracy assessment logic."""
        question = "How do I enable notifications?"
        agent_answer = "Go to Settings > Notifications and enable email alerts."
        reference_answer = "Navigate to Settings > Notifications > Email Settings and check notification boxes."
        
        accuracy, confidence, feedback = analysis_service._assess_accuracy(
            question, agent_answer, reference_answer
        )
        
        # Should have reasonable scores for similar answers
        assert 0.0 <= accuracy <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert isinstance(feedback, str)
        assert len(feedback) > 0
    
    def test_calculate_overall_score(self, analysis_service):
        """Test overall score calculation."""
        from app.models.analysis import QuestionAnswerPair
        
        qa_pairs = [
            QuestionAnswerPair(
                question="Test 1",
                agent_answer="Answer 1",
                reference_answer="Ref 1",
                accuracy_score=0.8,
                confidence_score=0.9,
                feedback="Good"
            ),
            QuestionAnswerPair(
                question="Test 2", 
                agent_answer="Answer 2",
                reference_answer="Ref 2",
                accuracy_score=0.6,
                confidence_score=0.7,
                feedback="OK"
            )
        ]
        
        overall_score = analysis_service._calculate_overall_score(qa_pairs)
        
        # Should be weighted average, closer to higher confidence scores
        assert 0.0 <= overall_score <= 1.0
        assert overall_score > 0.6  # Should be above the lower score due to weighting
    
    def test_rag_client_error_handling(self, analysis_service, sample_transcript):
        """Test handling of RAG client errors."""
        from app.services.rag_client import RAGClientError
        
        # Mock RAG client to raise an error
        analysis_service.rag_client.generate_answer.side_effect = RAGClientError("Service unavailable")
        
        request = AnalysisRequest(
            transcript=sample_transcript,
            transcript_id="test_error"
        )
        
        result = analysis_service.analyze_transcript(request)
        
        # Should handle error gracefully with fallback
        assert result.total_questions >= 0
        for qa_pair in result.qa_pairs:
            assert qa_pair.accuracy_score == 0.5  # Fallback score
            assert qa_pair.confidence_score == 0.0  # No confidence without RAG
            assert "RAG service unavailable" in qa_pair.feedback 