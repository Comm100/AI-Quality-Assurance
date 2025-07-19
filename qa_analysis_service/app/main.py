"""Main FastAPI application for QA Analysis Service."""
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from app.models.analysis import AnalysisRequest, AnalysisResponse
from app.services.analysis_service import AnalysisService
from app.services.rag_client import RAGClient, RAGClientError
from app.services.llm_client import LLMClient, LLMClientError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# Global instances
analysis_service: Optional[AnalysisService] = None
rag_client: Optional[RAGClient] = None
llm_client: Optional[LLMClient] = None


# Security
security = HTTPBearer(auto_error=False)


async def verify_token(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify X-Token authentication."""
    # Check for X-Token header first (preferred method)
    x_token = request.headers.get("X-Token")
    if x_token:
        if x_token == settings.qa_service_token:
            return True
        else:
            raise HTTPException(status_code=401, detail="Invalid X-Token")
    
    # Fallback to Authorization header with Bearer token
    if credentials:
        if credentials.credentials == settings.qa_service_token:
            return True
        else:
            raise HTTPException(status_code=401, detail="Invalid Bearer token")
    
    # No token provided
    raise HTTPException(status_code=401, detail="Authentication required")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global analysis_service, rag_client, llm_client
    
    # Startup
    logger.info("Starting QA Analysis Service...")
    
    try:
        # Initialize services
        rag_client = RAGClient()
        
        # Initialize LLM client with config settings
        llm_client = LLMClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.openai_temperature
        )
        
        analysis_service = AnalysisService(rag_client=rag_client, llm_client=llm_client)
        
        # Check RAG service health
        if rag_client.health_check():
            logger.info("RAG service is healthy")
        else:
            logger.warning("RAG service is not available - service will use fallback mode")
        
        logger.info("QA Analysis Service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down QA Analysis Service...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "QA Analysis Service",
        "version": settings.api_version,
        "status": "running"
    }


@app.get("/health")
async def health_check(authenticated: bool = Depends(verify_token)):
    """Health check endpoint."""
    # Check if services are initialized
    if not analysis_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Check RAG service
    rag_healthy = rag_client.health_check() if rag_client else False
    
    return {
        "status": "healthy",
        "services": {
            "analysis": True,
            "rag": rag_healthy
        }
    }


@app.post("/aiqa/analysis/analyze", response_model=AnalysisResponse)
async def analyze_conversation(
    request: AnalysisRequest,
    authenticated: bool = Depends(verify_token)
):
    """Analyze a conversation and return QA analysis results.
    
    This endpoint implements the 3-stage analysis algorithm:
    1. Stage 1: Segment conversation into Q&A threads
    2. Stage 2: Generate AI reference answers from KB
    3. Stage 3: Score agent answers against AI references
    
    Args:
        request: The analysis request containing conversation and KB ID.
        
    Returns:
        AnalysisResponse: Comprehensive analysis results with ratings.
        
    Raises:
        HTTPException: If analysis fails.
    """
    if not analysis_service:
        raise HTTPException(status_code=503, detail="Analysis service not initialized")
    
    try:
        # Basic logging always
        logger.info(f"Starting analysis for conversation {request.conversation.id}")
        
        # Detailed logging only in debug mode
        if settings.debug:
            logger.info("=" * 80)
            logger.info("📥 INCOMING API REQUEST (DEBUG MODE)")
            logger.info("=" * 80)
            logger.info(f"Conversation ID: {request.conversation.id}")
            logger.info(f"Conversation Type: {request.conversation.type}")
            logger.info(f"Number of Messages: {len(request.conversation.messages)}")
            logger.info(f"Integrated KB ID: {request.integratedKbId}")
            
            for i, msg in enumerate(request.conversation.messages):
                logger.info(f"  Message {i+1} ({msg.role}): {msg.content[:100]}..." if len(msg.content) > 100 else f"  Message {i+1} ({msg.role}): {msg.content}")
        
        # Perform analysis
        response = analysis_service.analyze_conversation(request)
        
        # Basic logging always
        logger.info(f"Analysis completed for conversation {response.conversationId} - {len(response.questionRatings)} Q&A pairs analyzed")
        
        # Detailed logging only in debug mode
        if settings.debug:
            logger.info("=" * 80)
            logger.info("📤 OUTGOING API RESPONSE (DEBUG MODE)")
            logger.info("=" * 80)
            logger.info(f"Conversation ID: {response.conversationId}")
            logger.info(f"Conversation Type: {response.conversationType}")
            logger.info(f"Analysis Time: {response.analysisTime}")
            logger.info(f"Number of Question Ratings: {len(response.questionRatings)}")
            
            for i, rating in enumerate(response.questionRatings):
                logger.info(f"\n  Rating {i+1}:")
                logger.info(f"    Rewritten Question: {rating.aiRewrittenQuestion}")
                logger.info(f"    Agent Answer: {rating.agentAnswer}")
                logger.info(f"    AI Suggested Answer: {rating.aiSuggestedAnswer}")
                logger.info(f"    AI Score: {rating.aiScore}")
                logger.info(f"    AI Rationale: {rating.aiRationale}")
            
            logger.info("=" * 80)
        
        return response
        
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error for conversation {request.conversation.id}: {e}")
        raise HTTPException(status_code=400, detail="Invalid request parameters")
        
    except LLMClientError as e:
        # Handle LLM service errors
        logger.error(f"LLM service error for conversation {request.conversation.id}: {e}")
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable")
        
    except RAGClientError as e:
        # Handle RAG service errors (non-critical, continue with fallback)
        logger.warning(f"RAG service error for conversation {request.conversation.id}: {e}")
        # Analysis will continue with fallback mechanisms
        response = analysis_service.analyze_conversation(request)
        return response
        
    except Exception as e:
        # Handle all other unexpected errors
        logger.error(f"Unexpected error analyzing conversation {request.conversation.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed due to internal error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 