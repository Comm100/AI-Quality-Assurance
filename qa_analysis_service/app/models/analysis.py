"""Pydantic models for QA analysis operations."""
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""
    
    id: str = Field(..., description="Unique message identifier")
    role: Literal["customer", "agent", "system"] = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="When the message was sent")


class Conversation(BaseModel):
    """A complete conversation with multiple messages."""
    
    id: int = Field(..., description="Conversation identifier")
    type: Literal["chat", "ticket"] = Field(..., description="Type of conversation")
    messages: List[Message] = Field(..., description="List of messages in the conversation")


class AnalysisRequest(BaseModel):
    """Request model for QA analysis."""
    
    conversation: Conversation = Field(..., description="Complete conversation data to analyze")
    integratedKbId: str = Field(..., description="Integrated knowledge base ID for reference answer generation")


# New model for Stage 1: Thread segmentation
class ConversationThread(BaseModel):
    """A segmented Q&A thread from the conversation."""
    
    qid: str = Field(default_factory=lambda: str(uuid4()), description="Unique thread identifier")
    question: str = Field(..., description="Rewritten/reformatted customer question")
    answer: str = Field(..., description="Corresponding agent answer")


# Enhanced models for Stage 2: AI answer generation
class AIAnswer(BaseModel):
    """AI-generated answer with context."""
    
    answer: str = Field(..., description="The generated answer")
    context: str = Field(..., description="KB context used to generate the answer")


class AIAnswers(BaseModel):
    """Both suggested and detailed AI-generated answers."""
    
    suggested: AIAnswer = Field(..., description="Suggested concise AI answer")
    detailed: AIAnswer = Field(..., description="Detailed AI answer with reasoning")


# Enhanced model for Stage 3: Scoring
class QuestionRating(BaseModel):
    """A single question rating with comprehensive AI analysis."""
    
    aiRewrittenQuestion: str = Field(..., description="AI-rewritten version of the customer question")
    agentAnswer: str = Field(..., description="The agent's response to the question")
    aiSuggestedAnswer: str = Field(..., description="AI-generated golden answer, concise and accurate")
    aiScore: float = Field(
        ..., 
        ge=0.0,
        le=5.0, 
        description="AI quality score between 0.0 and 5.0"
    )
    aiRationale: str = Field(..., description="AI explanation of the score and comparison")
    # Internal fields not exposed in API response
    aiLongAnswerInternal: str = Field("", exclude=True, description="Long AI-generated answer for internal scoring use")
    kbVerifyInternal: List[str] = Field(
        default_factory=list,
        exclude=True,
        description="KB chunks used for verification"
    )


class AnalysisResponse(BaseModel):
    """Response model for QA analysis results."""
    
    conversationId: int = Field(..., description="Original conversation identifier")
    conversationType: Literal["chat", "ticket"] = Field(..., description="Type of conversation analyzed")
    analysisTime: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the analysis was performed"
    )
    questionRatings: List[QuestionRating] = Field(
        ..., 
        description="List of question ratings with AI analysis"
    )
    overallAccuracy: float = Field(
        ..., 
        ge=0.0,
        le=5.0,
        description="Overall accuracy score averaged across all Q&A pairs"
    )


# RAG service models updated for KB chunks
class RAGRequest(BaseModel):
    """Request model for RAG service calls."""
    
    question: str = Field(..., description="The question to get KB chunks for")
    k: int = Field(default=6, description="Number of KB chunks to retrieve")


class KBChunk(BaseModel):
    """A knowledge base chunk with metadata."""
    
    content: str = Field(..., description="The chunk content")
    source: str = Field(..., description="Source document/file")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Relevance confidence score"
    )


class RAGResponse(BaseModel):
    """Response model from RAG service."""
    
    question: str = Field(..., description="The original question")
    chunks: List[KBChunk] = Field(..., description="Retrieved KB chunks")
    formatted_chunks: List[str] = Field(
        ...,
        description="Formatted chunks with source citations"
    ) 