"""Core QA analysis service implementing the 3-stage algorithm."""
import logging
from typing import List, Tuple, Optional
from datetime import datetime


from ..config import settings
from ..models.analysis import (
    AnalysisRequest, 
    AnalysisResponse, 
    QuestionRating,
    Message,
    Conversation,
    ConversationThread,
    AIAnswers,
    AIAnswer
)
from .rag_client import RAGClient, RAGClientError
from .llm_client import LLMClient, LLMClientError
from .prompt_builder import PromptBuilder


logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for performing comprehensive QA analysis using the 3-stage algorithm."""
    
    def __init__(self, rag_client: Optional[RAGClient] = None, llm_client: Optional[LLMClient] = None):
        """Initialize the analysis service.
        
        Args:
            rag_client: RAG client for retrieving KB chunks. If None, creates default client.
            llm_client: LLM client for OpenAI calls. If None, creates default client.
        """
        self.settings = settings
        self.rag_client = rag_client or RAGClient()
        self.llm_client = llm_client or LLMClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.openai_temperature
        )
        self.prompt_builder = PromptBuilder()
    
    def analyze_conversation(self, request: AnalysisRequest) -> AnalysisResponse:
        """Analyze a conversation using the 3-stage algorithm.
        
        Args:
            request: The analysis request containing the conversation and KB ID.
            
        Returns:
            AnalysisResponse: The analysis results with question ratings.
        """
        logger.info(f"Starting 3-stage analysis for conversation: {request.conversation.id}")
        
        try:
            # Stage 1: Extract Q&A threads from conversation
            threads = self._stage1_segment_conversation(request.conversation)
            logger.info(f"Stage 1 complete: Extracted {len(threads)} Q&A threads")
            
            # Process each thread through stages 2 and 3
            question_ratings = []
            scores = []
            
            for thread in threads:
                # Stage 2: Retrieve KB chunks and generate AI answers
                ai_answers = self._stage2_generate_ai_answers(
                    thread.question, 
                    request.integratedKbId
                )
                
                # Extract verified KB chunks from AI answers context
                verified_kb_chunks = self._extract_verified_kb_chunks(ai_answers)
                
                # Stage 3: Score the agent's answer using verified KB chunks
                rating = self._stage3_score_agent_answer(
                    thread.question,
                    thread.answer,
                    ai_answers,
                    verified_kb_chunks
                )
                
                question_ratings.append(rating)
                # Only include scores >= 0 in average (exclude out-of-scope -1 scores)
                if rating.aiScore >= 0:
                    scores.append(rating.aiScore)
            
            logger.info(f"Analysis completed for conversation {request.conversation.id}")
            logger.info(f"Processed {len(scores)} in-scope Q&A pairs with scores: {[round(s, 1) for s in scores]}")
            
            return AnalysisResponse(
                conversationId=request.conversation.id,
                conversationType=request.conversation.type,
                analysisTime=datetime.utcnow(),
                questionRatings=question_ratings
            )
            
        except Exception as e:
            logger.error(f"Analysis failed for conversation {request.conversation.id}: {e}")
            raise
    
    def _stage1_segment_conversation(self, conversation: Conversation) -> List[ConversationThread]:
        """Stage 1: Segment conversation into Q&A threads using LLM.
        
        Args:
            conversation: The conversation to segment.
            
        Returns:
            List of Q&A threads.
        """
        logger.info("Stage 1: Segmenting conversation into threads")
        
        # Convert messages to transcript format
        transcript_lines = []
        for msg in conversation.messages:
            if msg.role == "customer":
                role_prefix = "CUST"
            elif msg.role == "agent":
                role_prefix = "AGT"
            else:
                continue  # Skip system messages
            
            # Extract time from timestamp (HH:MM format)
            time_str = msg.timestamp.strftime("%H:%M")
            transcript_lines.append(f"{role_prefix} {time_str} {msg.content}")
        
        transcript = "\n".join(transcript_lines)
        
        # If no valid messages found, return empty list
        if not transcript.strip():
            logger.info("No valid Q&A pairs found in conversation")
            return []
        
        # Use LLM to segment the transcript
        messages = self.prompt_builder.split_prompt(transcript)
        
        try:
            result = self.llm_client.chat_completion_json(messages)
            threads_data = result.get("threads", [])
            
            # Convert to ConversationThread objects
            threads = []
            for thread_data in threads_data:
                thread = ConversationThread(
                    qid=thread_data.get("qid", f"T{len(threads)+1}"),
                    question=thread_data["question"],
                    answer=thread_data["answer"]
                )
                threads.append(thread)
            
            return threads
            
        except LLMClientError as e:
            logger.error(f"Stage 1 LLM call failed: {e}")
            # Fallback to simple extraction
            return self._fallback_extract_qa_pairs(conversation.messages)
    
    def _fallback_extract_qa_pairs(self, messages: List[Message]) -> List[ConversationThread]:
        """Fallback method to extract Q&A pairs without LLM.
        
        Args:
            messages: List of messages in the conversation.
            
        Returns:
            List of Q&A threads.
        """
        threads = []
        pending_questions = []
        
        for i, message in enumerate(messages):
            if message.role == "customer":
                pending_questions.append(message.content)
            elif message.role == "agent" and pending_questions:
                # Combine all pending questions
                combined_question = " ".join(pending_questions)
                thread = ConversationThread(
                    question=combined_question,
                    answer=message.content
                )
                threads.append(thread)
                pending_questions = []
        
        return threads
    
    def _stage2_generate_ai_answers(self, question: str, kb_id: str) -> AIAnswers:
        """Stage 2: Retrieve KB chunks and generate AI answers.
        
        Args:
            question: The question to answer.
            kb_id: The knowledge base ID.
            
        Returns:
            AIAnswers: The AI-generated answers with context.
        """
        logger.info(f"Stage 2: Generating AI answers for question: {question[:50]}...")
        
        # Retrieve KB chunks from RAG service
        try:
            rag_response = self.rag_client.retrieve_chunks(question, k=6)
            kb_chunks = rag_response.formatted_chunks
            
            # Log KB chunks for this question (debug mode only)
            if self.settings.debug:
                logger.info("📚 KB CHUNKS RETRIEVED (DEBUG MODE):")
                logger.info(f"  Question: {question}")
                logger.info(f"  Number of chunks: {len(kb_chunks)}")
                for i, chunk in enumerate(kb_chunks):
                    logger.info(f"    Chunk {i+1}: {chunk[:100]}..." if len(chunk) > 100 else f"    Chunk {i+1}: {chunk}")
            else:
                logger.info(f"Retrieved {len(kb_chunks)} KB chunks for question")
                
        except RAGClientError as e:
            logger.warning(f"RAG service error: {e}")
            # Use empty chunks as fallback
            kb_chunks = []
            if self.settings.debug:
                logger.info("📚 KB CHUNKS: None (using empty fallback due to RAG error)")
            else:
                logger.info("Using empty KB chunks fallback due to RAG error")
        
        # Generate AI answers using LLM
        messages = self.prompt_builder.draft_prompt(question, kb_chunks)
        
        try:
            logger.info(f"Stage 2: Calling LLM with {len(messages)} messages")
            logger.debug(f"Stage 2: KB chunks count: {len(kb_chunks)}")
            result = self.llm_client.chat_completion_json(messages)
            logger.info(f"Stage 2: LLM response received: {result}")
            logger.debug(f"Stage 2: Response keys: {list(result.keys())}")
            
            # Parse AI answers
            suggested_data = result.get("ai_suggested_answer", {})
            detailed_data = result.get("ai_detailed_answer", {})
            
            ai_answers = AIAnswers(
                suggested=AIAnswer(
                    answer=suggested_data.get("answer", "I cannot answer this question"),
                    context=suggested_data.get("context", "")
                ),
                detailed=AIAnswer(
                    answer=detailed_data.get("answer", "I cannot answer this question"),
                    context=detailed_data.get("context", "")
                )
            )
            
            return ai_answers
            
        except LLMClientError as e:
            logger.error(f"Stage 2 LLM call failed: {e}")
            # Return default answers
            default_answers = AIAnswers(
                suggested=AIAnswer(answer="Unable to generate answer", context=""),
                detailed=AIAnswer(answer="Unable to generate answer due to technical error", context="")
            )
            return default_answers
    
    def _extract_verified_kb_chunks(self, ai_answers: AIAnswers) -> List[str]:
        """Extract verified KB chunks from AI answers context.
        
        Args:
            ai_answers: The AI-generated answers with context.
            
        Returns:
            List of verified KB chunks extracted from the context.
        """
        verified_chunks = []
        
        # Extract context from suggested answer
        if ai_answers.suggested.context:
            verified_chunks.append(f"Suggested Answer Context: {ai_answers.suggested.context}")
        
        # Extract context from detailed answer
        if ai_answers.detailed.context:
            verified_chunks.append(f"Detailed Answer Context: {ai_answers.detailed.context}")
        
        # If no context found, return empty list
        if not verified_chunks:
            logger.warning(f"No context found in AI answers for KB verification. Suggested context: '{ai_answers.suggested.context}', Detailed context: '{ai_answers.detailed.context}'")
        
        return verified_chunks
    
    def _stage3_score_agent_answer(self, question: str, agent_answer: str, 
                                   ai_answers: AIAnswers, kb_chunks: List[str]) -> QuestionRating:
        """Stage 3: Score the agent's answer using LLM.
        
        Args:
            question: The rewritten question.
            agent_answer: The agent's answer.
            ai_answers: The AI-generated answers.
            kb_chunks: The KB chunks used as evidence.
            
        Returns:
            QuestionRating with score and analysis.
        """
        logger.info("Stage 3: Scoring agent answer")
        
        # Prepare bundle for grading
        bundle = {
            "question": question,
            "agent": agent_answer,
            "ai_suggested": ai_answers.suggested.answer,
            "ai_detailed": ai_answers.detailed.answer,
            "kb_evidence": kb_chunks
        }
        
        # Get grading from LLM
        messages = self.prompt_builder.grade_prompt(bundle)
        
        try:
            result = self.llm_client.chat_completion_json(messages)
            
            ai_score = result.get("ai_score", 0)
            ai_rationale = result.get("ai_rational", result.get("ai_rationale", "Unable to generate rationale"))
            kb_verify = result.get("kb_verify", kb_chunks)
            
            return QuestionRating(
                aiRewrittenQuestion=question,
                agentAnswer=agent_answer,
                aiSuggestedAnswer=ai_answers.suggested.answer,
                aiLongAnswerInternal=ai_answers.detailed.answer,
                aiScore=float(ai_score),
                aiRationale=ai_rationale,
                kbVerifyInternal=kb_verify
            )
            
        except LLMClientError as e:
            logger.error(f"Stage 3 LLM call failed: {e}")
            # Return a default rating
            return QuestionRating(
                aiRewrittenQuestion=question,
                agentAnswer=agent_answer,
                aiSuggestedAnswer=ai_answers.suggested.answer,
                aiLongAnswerInternal=ai_answers.detailed.answer,
                aiScore=0.0,
                aiRationale="Unable to score due to technical error",
                kbVerifyInternal=kb_chunks
            ) 