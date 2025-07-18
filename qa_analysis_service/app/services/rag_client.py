"""Client for communicating with the RAG service."""
import logging
from typing import Optional, List

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

from ..config import settings
from ..models.analysis import RAGRequest, RAGResponse


logger = logging.getLogger(__name__)


class RAGClientError(Exception):
    """Exception raised when RAG service communication fails."""
    pass


class RAGClient:
    """Client for interacting with the RAG service."""
    
    def __init__(self, base_url: Optional[str] = None, timeout: Optional[int] = None):
        """Initialize the RAG client.
        
        Args:
            base_url: Base URL for the RAG service. Defaults to config setting.
            timeout: Request timeout in seconds. Defaults to config setting.
        """
        self.settings = settings
        self.base_url = base_url or settings.rag_service_url
        self.timeout = timeout or getattr(settings, 'rag_service_timeout', 60)
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def retrieve_chunks(self, question: str, k: int = 6) -> RAGResponse:
        """Retrieve relevant KB chunks for the given question.
        
        Args:
            question: The question to get KB chunks for.
            k: Number of chunks to retrieve.
            
        Returns:
            RAGResponse: The response from the RAG service with KB chunks.
            
        Raises:
            RAGClientError: If the request fails or returns an error.
        """
        request_data = RAGRequest(question=question, k=k)
        
        try:
            # Log RAG request details (debug mode only)
            if self.settings.debug:
                logger.info("🔄 RAG SERVICE REQUEST (DEBUG MODE):")
                logger.info(f"  URL: {self.base_url}/retrieve-chunks")
                logger.info(f"  Method: POST")
                logger.info(f"  Request Data: {request_data.model_dump()}")
            
            response = self.session.post(
                f"{self.base_url}/retrieve-chunks",
                json=request_data.model_dump(),
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            # Log RAG response details (debug mode only)
            response_json = response.json()
            rag_response = RAGResponse(**response_json)
            
            if self.settings.debug:
                logger.info("✅ RAG SERVICE RESPONSE (DEBUG MODE):")
                logger.info(f"  Status Code: {response.status_code}")
                logger.info(f"  Response Size: {len(str(response_json))} characters")
                logger.info(f"  Number of chunks returned: {len(rag_response.chunks)}")
                logger.info(f"  Question processed: {rag_response.question}")
                
                for i, chunk in enumerate(rag_response.chunks):
                    logger.info(f"    Chunk {i+1} (confidence: {chunk.confidence:.2f}): {chunk.content[:80]}..." if len(chunk.content) > 80 else f"    Chunk {i+1} (confidence: {chunk.confidence:.2f}): {chunk.content}")
            else:
                logger.info(f"RAG service returned {len(rag_response.chunks)} chunks")
            
            return rag_response
            
        except ConnectionError as e:
            error_msg = f"Failed to connect to RAG service at {self.base_url}: {e}"
            logger.error(error_msg)
            raise RAGClientError(error_msg) from e
            
        except Timeout as e:
            error_msg = f"RAG service request timed out after {self.timeout} seconds: {e}"
            logger.error(error_msg)
            raise RAGClientError(error_msg) from e
            
        except RequestException as e:
            error_msg = f"RAG service request failed: {e}"
            logger.error(error_msg)
            raise RAGClientError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error when calling RAG service: {e}"
            logger.error(error_msg)
            raise RAGClientError(error_msg) from e
    
    def health_check(self) -> bool:
        """Check if the RAG service is healthy.
        
        Returns:
            bool: True if the service is healthy, False otherwise.
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5  # Short timeout for health checks
            )
            return response.status_code == 200
            
        except Exception as e:
            logger.warning(f"RAG service health check failed: {e}")
            return False 