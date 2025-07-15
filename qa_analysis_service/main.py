"""Main FastAPI application for QA Analysis Service."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.analysis import AnalysisRequest, AnalysisResponse
from app.services.analysis_service import AnalysisService
from app.services.rag_client import RAGClient


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# Global instances
analysis_service: AnalysisService = None
rag_client: RAGClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global analysis_service, rag_client
    
    # Startup
    logger.info("Starting QA Analysis Service...")
    
    # Initialize services
    rag_client = RAGClient()
    analysis_service = AnalysisService(rag_client=rag_client)
    
    # Check RAG service health
    if rag_client.health_check():
        logger.info("RAG service is healthy")
    else:
        logger.warning("RAG service is not available - service will use fallback mode")
    
    logger.info("QA Analysis Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down QA Analysis Service...")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description="Service for performing comprehensive QA analysis on chat transcripts",
    version=settings.api_version,
    debug=settings.debug,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "QA Analysis Service",
        "version": settings.api_version,
        "status": "running",
        "endpoints": {
            "analyze": "/analyze",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    rag_healthy = rag_client.health_check() if rag_client else False
    
    return {
        "status": "healthy",
        "service": "QA Analysis Service",
        "dependencies": {
            "rag_service": "healthy" if rag_healthy else "unavailable"
        }
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_transcript(request: AnalysisRequest):
    """Analyze a chat transcript and return QA analysis results.
    
    This endpoint performs comprehensive analysis including:
    - Segmentation of transcript into question-answer pairs
    - Generation of reference answers using RAG service
    - Accuracy assessment of agent responses
    - Overall quality scoring
    
    Args:
        request: Analysis request containing the transcript and metadata.
        
    Returns:
        AnalysisResponse: Comprehensive analysis results.
        
    Raises:
        HTTPException: If analysis fails.
    """
    try:
        logger.info(f"Received analysis request for transcript: {request.transcript_id}")
        
        # Validate input
        if not request.transcript or len(request.transcript.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Transcript cannot be empty"
            )
        
        # Perform analysis
        result = analysis_service.analyze_transcript(request)
        
        logger.info(f"Analysis completed for transcript: {request.transcript_id}")
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.error(f"Analysis failed for transcript {request.transcript_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info" if settings.debug else "warning"
    ) 