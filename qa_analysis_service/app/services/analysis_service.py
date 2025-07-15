"""Core QA analysis service for processing chat transcripts."""
import logging
import re
import time
from typing import List, Tuple, Optional
from datetime import datetime

from ..models.analysis import (
    AnalysisRequest, 
    AnalysisResponse, 
    QuestionAnswerPair
)
from .rag_client import RAGClient, RAGClientError


logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for performing comprehensive QA analysis on chat transcripts."""
    
    def __init__(self, rag_client: Optional[RAGClient] = None):
        """Initialize the analysis service.
        
        Args:
            rag_client: RAG client for getting reference answers. If None, creates default client.
        """
        self.rag_client = rag_client or RAGClient()
    
    def analyze_transcript(self, request: AnalysisRequest) -> AnalysisResponse:
        """Analyze a chat transcript and return QA analysis results.
        
        Args:
            request: The analysis request containing the transcript.
            
        Returns:
            AnalysisResponse: The analysis results.
        """
        start_time = time.time()
        
        logger.info(f"Starting analysis for transcript: {request.transcript_id}")
        
        try:
            # Step 1: Segment transcript into Q&A pairs
            qa_pairs = self._segment_transcript(request.transcript)
            logger.info(f"Extracted {len(qa_pairs)} Q&A pairs from transcript")
            
            # Step 2: Process each Q&A pair
            analyzed_pairs = []
            for question, agent_answer in qa_pairs:
                analyzed_pair = self._analyze_qa_pair(question, agent_answer)
                analyzed_pairs.append(analyzed_pair)
            
            # Step 3: Calculate overall score
            overall_score = self._calculate_overall_score(analyzed_pairs)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(f"Analysis completed in {processing_time_ms}ms with overall score: {overall_score}")
            
            return AnalysisResponse(
                transcript_id=request.transcript_id,
                qa_pairs=analyzed_pairs,
                overall_score=overall_score,
                total_questions=len(analyzed_pairs),
                analysis_timestamp=datetime.utcnow(),
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"Analysis failed for transcript {request.transcript_id}: {e}")
            raise
    
    def _segment_transcript(self, transcript: str) -> List[Tuple[str, str]]:
        """Segment a transcript into question-answer pairs.
        
        Args:
            transcript: The raw transcript text.
            
        Returns:
            List of tuples containing (question, answer) pairs.
        """
        lines = [line.strip() for line in transcript.split('\n') if line.strip()]
        qa_pairs = []
        
        current_question = None
        current_answer = []
        
        for line in lines:
            # Skip empty lines
            if not line:
                continue
            
            # Check if line is from customer (likely a question)
            if line.startswith('Customer:'):
                # If we have a previous question and answer, save it
                if current_question and current_answer:
                    answer_text = ' '.join(current_answer)
                    qa_pairs.append((current_question, answer_text))
                
                # Start new question
                current_question = line.replace('Customer:', '').strip()
                current_answer = []
            
            # Check if line is from agent (likely an answer)
            elif line.startswith('Agent:'):
                if current_question:  # Only collect answer if we have a question
                    answer_text = line.replace('Agent:', '').strip()
                    current_answer.append(answer_text)
        
        # Don't forget the last Q&A pair
        if current_question and current_answer:
            answer_text = ' '.join(current_answer)
            qa_pairs.append((current_question, answer_text))
        
        # Filter out pairs where question might not be a real question
        filtered_pairs = []
        for question, answer in qa_pairs:
            if self._is_likely_question(question):
                filtered_pairs.append((question, answer))
        
        return filtered_pairs
    
    def _is_likely_question(self, text: str) -> bool:
        """Determine if a text is likely a question that needs analysis.
        
        Args:
            text: The text to evaluate.
            
        Returns:
            bool: True if the text is likely a question needing analysis.
        """
        # Simple heuristics to identify questions
        question_indicators = [
            '?',  # Explicit question mark
            'how',
            'what',
            'where',
            'when',
            'why',
            'which',
            'who',
            'can you',
            'could you',
            'would you',
            'help',
            'problem',
            'issue',
            'trouble',
            'need',
            'want'
        ]
        
        text_lower = text.lower()
        
        # Must have more than just greeting/thanks
        if len(text.split()) < 3:
            return False
        
        # Check for question indicators
        for indicator in question_indicators:
            if indicator in text_lower:
                return True
        
        return False
    
    def _analyze_qa_pair(self, question: str, agent_answer: str) -> QuestionAnswerPair:
        """Analyze a single question-answer pair.
        
        Args:
            question: The customer question.
            agent_answer: The agent's response.
            
        Returns:
            QuestionAnswerPair: The analyzed Q&A pair with accuracy assessment.
        """
        try:
            # Get reference answer from RAG service
            rag_response = self.rag_client.generate_answer(question)
            reference_answer = rag_response.answer
            
            # Assess accuracy by comparing agent answer with reference
            accuracy_score, confidence_score, feedback = self._assess_accuracy(
                question, agent_answer, reference_answer
            )
            
            return QuestionAnswerPair(
                question=question,
                agent_answer=agent_answer,
                reference_answer=reference_answer,
                accuracy_score=accuracy_score,
                confidence_score=confidence_score,
                feedback=feedback
            )
            
        except RAGClientError as e:
            logger.warning(f"RAG service unavailable, using fallback assessment: {e}")
            
            # Fallback assessment when RAG service is unavailable
            return QuestionAnswerPair(
                question=question,
                agent_answer=agent_answer,
                reference_answer="Reference answer unavailable - RAG service error",
                accuracy_score=0.5,  # Neutral score when we can't assess
                confidence_score=0.0,  # No confidence without reference
                feedback="Unable to assess accuracy - RAG service unavailable"
            )
    
    def _assess_accuracy(
        self, 
        question: str, 
        agent_answer: str, 
        reference_answer: str
    ) -> Tuple[float, float, str]:
        """Assess the accuracy of an agent's answer against a reference answer.
        
        Args:
            question: The customer question.
            agent_answer: The agent's response.
            reference_answer: The reference answer from RAG.
            
        Returns:
            Tuple of (accuracy_score, confidence_score, feedback).
        """
        # Simple accuracy assessment based on keyword overlap and length similarity
        
        # Normalize texts for comparison
        agent_words = set(re.findall(r'\w+', agent_answer.lower()))
        reference_words = set(re.findall(r'\w+', reference_answer.lower()))
        
        # Calculate word overlap
        common_words = agent_words.intersection(reference_words)
        total_unique_words = agent_words.union(reference_words)
        
        if len(total_unique_words) == 0:
            overlap_score = 0.0
        else:
            overlap_score = len(common_words) / len(total_unique_words)
        
        # Length similarity (penalize answers that are too short or too long)
        agent_length = len(agent_answer.split())
        reference_length = len(reference_answer.split())
        
        if reference_length == 0:
            length_score = 0.5
        else:
            length_ratio = min(agent_length, reference_length) / max(agent_length, reference_length)
            length_score = length_ratio
        
        # Combine scores (weighted average)
        accuracy_score = (overlap_score * 0.7) + (length_score * 0.3)
        
        # Confidence is higher when we have good overlap and similar length
        confidence_score = min(overlap_score + length_score * 0.5, 1.0)
        
        # Generate feedback
        if accuracy_score >= 0.8:
            feedback = "Excellent response - closely matches reference answer"
        elif accuracy_score >= 0.6:
            feedback = "Good response - covers most key points from reference"
        elif accuracy_score >= 0.4:
            feedback = "Adequate response - some alignment with reference answer"
        else:
            feedback = "Response differs significantly from reference answer"
        
        return accuracy_score, confidence_score, feedback
    
    def _calculate_overall_score(self, qa_pairs: List[QuestionAnswerPair]) -> float:
        """Calculate the overall quality score for all Q&A pairs.
        
        Args:
            qa_pairs: List of analyzed Q&A pairs.
            
        Returns:
            float: Overall score between 0.0 and 1.0.
        """
        if not qa_pairs:
            return 0.0
        
        # Calculate weighted average of accuracy scores
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for pair in qa_pairs:
            # Weight by confidence - more confident assessments have more impact
            weight = max(pair.confidence_score, 0.1)  # Minimum weight to avoid zero
            total_weighted_score += pair.accuracy_score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0 