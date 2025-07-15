"""Dummy RAG Service for testing QA Analysis Service."""
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    """Request model for RAG service."""
    
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
    sources: List[str] = Field(
        default_factory=list,
        description="Sources used to generate the answer"
    )


app = FastAPI(
    title="Dummy RAG Service",
    description="Provides reference answers from knowledge base for QA analysis",
    version="1.0.0"
)

# Knowledge base for generating reference answers
KNOWLEDGE_BASE = {
    "email notifications": {
        "answer": "To enable email notifications, navigate to Settings > Notifications > Email Settings and check the appropriate notification boxes. Ensure you have the required permissions and your browser cache is cleared.",
        "confidence": 0.95,
        "sources": ["User Manual - Notifications", "Admin Guide v2.1"]
    },
    "crm integration": {
        "answer": "Comm100 supports various CRM integrations including Salesforce, HubSpot, and Zendesk. Access integrations via Admin > Apps & Integrations > Marketplace. Admin privileges are required for installation.",
        "confidence": 0.92,
        "sources": ["Integration Guide", "CRM Setup Documentation"]
    },
    "chat widget": {
        "answer": "The chat widget code should be placed just before the closing </body> tag in your website's HTML. Ensure the code is not placed in the <head> section and clear browser cache after installation.",
        "confidence": 0.98,
        "sources": ["Installation Guide", "Widget Troubleshooting"]
    },
    "salesforce": {
        "answer": "To integrate Salesforce with Comm100, go to Admin > Apps & Integrations > Marketplace, search for Salesforce, and click Install. You'll need admin privileges and your Salesforce API credentials for the setup wizard.",
        "confidence": 0.94,
        "sources": ["Salesforce Integration Guide", "API Setup Manual"]
    },
    "troubleshooting": {
        "answer": "Common troubleshooting steps include: 1) Verify installation and configuration, 2) Check permissions and access rights, 3) Clear browser cache, 4) Refresh the page, 5) Contact support if issues persist.",
        "confidence": 0.88,
        "sources": ["General Troubleshooting Guide", "Support FAQ"]
    },
    "permissions": {
        "answer": "Admin privileges are required for most configuration changes including enabling notifications, installing integrations, and modifying system settings. Check with your organization's admin if you lack required permissions.",
        "confidence": 0.90,
        "sources": ["User Permissions Guide", "Admin Manual"]
    }
}


def find_best_answer(question: str) -> dict:
    """Find the best matching answer for a question using simple keyword matching."""
    question_lower = question.lower()
    best_match = None
    max_score = 0
    
    for topic, data in KNOWLEDGE_BASE.items():
        # Simple keyword matching
        if topic in question_lower:
            score = len(topic)
            if score > max_score:
                max_score = score
                best_match = data
    
    # If no direct match, provide a generic helpful response
    if not best_match:
        best_match = {
            "answer": "I'd be happy to help you with that. For specific technical issues, please refer to our documentation or contact our support team for personalized assistance.",
            "confidence": 0.70,
            "sources": ["General Support Guide"]
        }
    
    return best_match


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Dummy RAG Service", "status": "running"}


@app.post("/generate-answer", response_model=RAGResponse)
async def generate_answer(request: RAGRequest):
    """Generate a reference answer for the given question."""
    
    # Find the best matching answer
    answer_data = find_best_answer(request.question)
    
    return RAGResponse(
        question=request.question,
        answer=answer_data["answer"],
        confidence=answer_data["confidence"],
        sources=answer_data["sources"]
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "RAG Service"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 