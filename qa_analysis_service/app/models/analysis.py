"""Pydantic models for QA analysis operations."""
from typing import List, Optional, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field


class QuestionAnswerPair(BaseModel):
    """A single question-answer pair extracted from a transcript."""
    
    question: str = Field(..., description="The customer question")
    agent_answer: str = Field(..., description="The agent's response")
    reference_answer: str = Field(..., description="AI-generated reference answer from RAG")
    accuracy_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Accuracy score between 0.0 and 1.0"
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence in the accuracy assessment"
    )
    feedback: Optional[str] = Field(
        None, 
        description="Detailed feedback on the agent's answer"
    )


class AnalysisRequest(BaseModel):
    """Request model for QA analysis."""
    
    transcript: str = Field(..., description="The chat or ticket transcript to analyze")
    transcript_id: Optional[str] = Field(None, description="Unique identifier for the transcript")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Additional metadata about the interaction"
    )


class AnalysisResponse(BaseModel):
    """Response model for QA analysis results."""
    
    transcript_id: Optional[str] = Field(None, description="Unique identifier for the transcript")
    qa_pairs: List[QuestionAnswerPair] = Field(
        ..., 
        description="List of question-answer pairs extracted and analyzed"
    )
    overall_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Overall quality score for the interaction"
    )
    total_questions: int = Field(..., description="Total number of questions identified")
    analysis_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the analysis was performed"
    )
    processing_time_ms: Optional[int] = Field(
        None, 
        description="Time taken to process the analysis in milliseconds"
    )


class RAGRequest(BaseModel):
    """Request model for RAG service calls."""
    
    question: str = Field(..., description="The question to get a reference answer for")
    context: Optional[str] = Field(None, description="Additional context for the question")


class RAGResponse(BaseModel):
    """Response model from RAG service."""
    
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The reference answer from knowledge base")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence in the answer quality"
    )
    sources: Optional[List[str]] = Field(
        default_factory=list,
        description="Sources used to generate the answer"
    ) 