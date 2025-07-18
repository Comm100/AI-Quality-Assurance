"""Dummy RAG Service for testing QA Analysis Service."""
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import random


class RAGRequest(BaseModel):
    """Request model for RAG service."""
    
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


app = FastAPI(
    title="Dummy RAG Service",
    description="Retrieves relevant KB chunks for QA analysis",
    version="2.0.0"
)

# Knowledge base chunks organized by topic
KNOWLEDGE_BASE_CHUNKS = {
    "invoices": [
        {
            "content": "On the Invoices page, use the Status dropdown to choose Unpaid.",
            "source": "billing/invoices.md",
            "confidence": 0.95
        },
        {
            "content": "To view unpaid invoices, navigate to Billing → Invoices and filter by status.",
            "source": "billing/overview.md",
            "confidence": 0.92
        },
        {
            "content": "The Billing Summary page provides an alternative way to filter unpaid invoices in older UI versions.",
            "source": "billing/legacy-ui.md",
            "confidence": 0.88
        }
    ],
    "display_name": [
        {
            "content": "To change your display name, go to People → Agents, click your name, then edit Display Name field.",
            "source": "agents/profile-settings.md",
            "confidence": 0.96
        },
        {
            "content": "Name changes propagate within 10 minutes across all chat interfaces.",
            "source": "agents/profile-sync.md",
            "confidence": 0.94
        },
        {
            "content": "Display names are visible to visitors in live chat sessions.",
            "source": "agents/visitor-experience.md",
            "confidence": 0.85
        }
    ],
    "2fa": [
        {
            "content": "Only administrators can disable 2FA requirements for agents.",
            "source": "security/2fa-management.md",
            "confidence": 0.98
        },
        {
            "content": "Agents cannot disable their own 2FA - admin permission is required.",
            "source": "security/agent-permissions.md",
            "confidence": 0.95
        },
        {
            "content": "2FA settings are managed in Security Settings by administrators.",
            "source": "security/admin-guide.md",
            "confidence": 0.90
        }
    ],
    "api_permissions": [
        {
            "content": "Use PUT to /global/agents/{id}/permissions with an empty array to remove all permissions.",
            "source": "api/agents-endpoints.md",
            "confidence": 0.97
        },
        {
            "content": "DELETE method is not supported on the permissions endpoint - use PUT instead.",
            "source": "api/methods-guide.md",
            "confidence": 0.93
        },
        {
            "content": "Agent permissions can be managed programmatically via the REST API.",
            "source": "api/overview.md",
            "confidence": 0.85
        }
    ],
    "routing": [
        {
            "content": "Time-based routing rules are configured under Live Chat → Settings → Triggers → Time-Based.",
            "source": "routing/time-rules.md",
            "confidence": 0.94
        },
        {
            "content": "Auto-assignment schedules can be set to route chats only during specific hours.",
            "source": "routing/auto-assignment.md",
            "confidence": 0.91
        },
        {
            "content": "Routing rules support business hours configuration (e.g., 9 AM-5 PM).",
            "source": "routing/business-hours.md",
            "confidence": 0.88
        }
    ],
    "credit_card": [
        {
            "content": "To update payment method, first remove the existing card, then add a new payment method.",
            "source": "billing/payment-methods.md",
            "confidence": 0.95
        },
        {
            "content": "Credit card updates require removing and re-adding payment information.",
            "source": "billing/card-management.md",
            "confidence": 0.92
        },
        {
            "content": "Payment methods are managed in the Billing Profile section.",
            "source": "billing/profile.md",
            "confidence": 0.87
        }
    ],
    "timestamps": [
        {
            "content": "Chat timestamps are permanent once recorded and cannot be retroactively adjusted.",
            "source": "chat/data-integrity.md",
            "confidence": 0.96
        },
        {
            "content": "Audit logs track edits but do not allow timestamp modifications.",
            "source": "chat/audit-trail.md",
            "confidence": 0.93
        },
        {
            "content": "Timezone settings affect new chats only, not historical records.",
            "source": "chat/timezone-settings.md",
            "confidence": 0.89
        }
    ],
    "api_docs": [
        {
            "content": "API documentation is available at https://developer.comm100.com/restful-api-guide.",
            "source": "developer/api-docs.md",
            "confidence": 0.98
        },
        {
            "content": "The docs.comm100.com/developer/api URL has been deprecated - use developer.comm100.com instead.",
            "source": "developer/url-changes.md",
            "confidence": 0.95
        },
        {
            "content": "REST API guides include authentication, endpoints, and code examples.",
            "source": "developer/resources.md",
            "confidence": 0.87
        }
    ],
    "default": [
        {
            "content": "For specific technical issues, please refer to our documentation or contact support.",
            "source": "support/general.md",
            "confidence": 0.70
        },
        {
            "content": "Our knowledge base contains detailed guides for all product features.",
            "source": "support/kb-overview.md",
            "confidence": 0.65
        }
    ]
}


def find_relevant_chunks(question: str, k: int = 6) -> List[KBChunk]:
    """Find the most relevant KB chunks for a question."""
    question_lower = question.lower()
    selected_chunks = []
    
    # Keywords mapping for topic matching
    topic_keywords = {
        "invoices": ["invoice", "unpaid", "billing", "bills", "payment", "filter"],
        "display_name": ["display name", "name", "profile", "visitor see", "agent name"],
        "2fa": ["2fa", "two factor", "authentication", "disable 2fa", "security"],
        "api_permissions": ["api", "permission", "delete", "remove permission", "agent permission"],
        "routing": ["routing", "auto-assign", "time", "9 am", "5 pm", "schedule"],
        "credit_card": ["credit card", "payment method", "card", "update card"],
        "timestamps": ["timestamp", "retroactive", "adjust time", "timezone", "audit"],
        "api_docs": ["api doc", "documentation", "developer", "restful"]
    }
    
    # Find matching topics
    matched_topics = []
    for topic, keywords in topic_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            matched_topics.append(topic)
    
    # If no specific match, use default
    if not matched_topics:
        matched_topics = ["default"]
    
    # Collect chunks from matched topics
    for topic in matched_topics:
        if topic in KNOWLEDGE_BASE_CHUNKS:
            selected_chunks.extend([
                KBChunk(**chunk) for chunk in KNOWLEDGE_BASE_CHUNKS[topic]
            ])
    
    # Sort by confidence and limit to k
    selected_chunks.sort(key=lambda x: x.confidence, reverse=True)
    selected_chunks = selected_chunks[:k]
    
    # If we don't have enough chunks, pad with defaults
    while len(selected_chunks) < k and "default" in KNOWLEDGE_BASE_CHUNKS:
        remaining = k - len(selected_chunks)
        default_chunks = [KBChunk(**chunk) for chunk in KNOWLEDGE_BASE_CHUNKS["default"]]
        selected_chunks.extend(default_chunks[:remaining])
    
    return selected_chunks


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Dummy RAG Service v2", "status": "running"}


@app.post("/retrieve-chunks", response_model=RAGResponse)
async def retrieve_chunks(request: RAGRequest):
    """Retrieve relevant KB chunks for the given question."""
    
    # Find relevant chunks
    chunks = find_relevant_chunks(request.question, request.k)
    
    # Format chunks with source citations
    formatted_chunks = [
        f"{chunk.content} (source: {chunk.source})"
        for chunk in chunks
    ]
    
    return RAGResponse(
        question=request.question,
        chunks=chunks,
        formatted_chunks=formatted_chunks
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "RAG Service v2"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 