"""Client for communicating with the RAG service."""
import logging
from typing import Optional

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
    
    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        """Initialize the RAG client.
        
        Args:
            base_url: Base URL for the RAG service. Defaults to config setting.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url or settings.rag_service_url
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def generate_answer(self, question: str, context: Optional[str] = None) -> RAGResponse:
        """Generate a reference answer for the given question.
        
        Args:
            question: The question to get a reference answer for.
            context: Optional additional context for the question.
            
        Returns:
            RAGResponse: The response from the RAG service.
            
        Raises:
            RAGClientError: If the request fails or returns an error.
        """
        request_data = RAGRequest(question=question, context=context)
        
        try:
            logger.info(f"Requesting answer for question: {question[:50]}...")
            
            response = self.session.post(
                f"{self.base_url}/generate-answer",
                json=request_data.model_dump(),
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            rag_response = RAGResponse(**response.json())
            
            logger.info(f"Received answer with confidence: {rag_response.confidence}")
            
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