"""Main FastAPI application for QA Analysis Service."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.analysis import AnalysisRequest, AnalysisResponse
from app.services.analysis_service import AnalysisService
from app.services.rag_client import RAGClient
from app.services.llm_client import LLMClient


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# Global instances
analysis_service: AnalysisService = None
rag_client: RAGClient = None
llm_client: LLMClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global analysis_service, rag_client, llm_client
    
    # Startup
    logger.info("Starting QA Analysis Service...")
    
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
    
    yield
    
    # Shutdown
    logger.info("Shutting down QA Analysis Service...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
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
async def health_check():
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
async def analyze_conversation(request: AnalysisRequest):
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
        logger.info(f"Received analysis request for conversation: {request.conversation.id}")
        
        # Detailed logging only in debug mode
        if settings.debug:
            logger.info("=" * 80)
            logger.info("📥 INCOMING API REQUEST (DEBUG MODE)")
            logger.info("=" * 80)
            logger.info(f"Endpoint: POST /aiqa/analysis/analyze")
            logger.info(f"Conversation ID: {request.conversation.id}")
            logger.info(f"Conversation Type: {request.conversation.type}")
            logger.info(f"Integrated KB ID: {request.integratedKbId}")
            logger.info(f"Number of Messages: {len(request.conversation.messages)}")
            
            for i, msg in enumerate(request.conversation.messages):
                logger.info(f"  Message {i+1} ({msg.role}): {msg.content[:100]}..." if len(msg.content) > 100 else f"  Message {i+1} ({msg.role}): {msg.content}")
        
        # Perform analysis
        response = analysis_service.analyze_conversation(request)
        
        # Basic logging always
        logger.info(f"Analysis completed for conversation {response.conversationId} - Overall accuracy: {response.overallAccuracy}")
        
        # Detailed logging only in debug mode
        if settings.debug:
            logger.info("=" * 80)
            logger.info("📤 OUTGOING API RESPONSE (DEBUG MODE)")
            logger.info("=" * 80)
            logger.info(f"Conversation ID: {response.conversationId}")
            logger.info(f"Conversation Type: {response.conversationType}")
            logger.info(f"Analysis Time: {response.analysisTime}")
            logger.info(f"Overall Accuracy: {response.overallAccuracy}")
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
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Example endpoint for testing with dummy data
@app.post("/analyze-example")
async def analyze_example():
    """Analyze an example conversation for testing."""
    from datetime import datetime
    from app.models.analysis import Conversation, Message
    
    # Create example conversation
    example_conversation = Conversation(
        id=1,
        type="chat",
        messages=[
            Message(
                id="1",
                role="customer",
                content="Hey, I just saw an alert about unpaid invoices—where do I check them?",
                timestamp=datetime.utcnow()
            ),
            Message(
                id="2",
                role="agent",
                content="Sure—head over to Billing → Invoices and you'll see all your bills.",
                timestamp=datetime.utcnow()
            )
        ]
    )
    
    request = AnalysisRequest(
        conversation=example_conversation,
        integratedKbId="kb_001"
    )
    
    return await analyze_conversation(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 