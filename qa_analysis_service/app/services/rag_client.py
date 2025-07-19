"""Client for communicating with the real RAG service."""
import logging
from typing import Optional, List

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

from ..config import settings
from ..models.analysis import RAGRequest, RAGResponse, KBChunk


logger = logging.getLogger(__name__)


class RAGClientError(Exception):
    """Exception raised when RAG service communication fails."""
    pass


class RAGClient:
    """Client for interacting with the real RAG service."""
    
    def __init__(self, base_url: Optional[str] = None, timeout: Optional[int] = None, 
                 token: Optional[str] = None, site_id: Optional[str] = None):
        """Initialize the RAG client.
        
        Args:
            base_url: Base URL for the RAG service. Defaults to config setting.
            timeout: Request timeout in seconds. Defaults to config setting.
            token: Authentication token. Defaults to config setting.
            site_id: Site ID for the API. Defaults to config setting.
        """
        self.settings = settings
        self.base_url = base_url or settings.rag_service_url
        self.timeout = timeout or getattr(settings, 'rag_service_timeout', 60)
        self.token = token or getattr(settings, 'rag_service_token', 'cc9dfc7473d3486dac06e1634d4ce38e')
        self.site_id = site_id or getattr(settings, 'rag_service_site_id', '10001')
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Token": self.token
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
        try:
            # Prepare request payload for the real API
            payload = {
                "questions": [question]
            }
            
            # Construct the URL with site ID
            url = f"{self.base_url}/topSegments?siteId={self.site_id}"
            
            # Log RAG request details (debug mode only)
            if self.settings.debug:
                logger.info("🔄 RAG SERVICE REQUEST (DEBUG MODE):")
                logger.info(f"  URL: {url}")
                logger.info(f"  Method: POST")
                logger.info(f"  Token: {self.token[:8]}...")
                logger.info(f"  Site ID: {self.site_id}")
                logger.info(f"  Request Data: {payload}")
            
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            # Parse the real API response
            response_json = response.json()
            
            # Log RAG response details (debug mode only)
            if self.settings.debug:
                logger.info("✅ RAG SERVICE RESPONSE (DEBUG MODE):")
                logger.info(f"  Status Code: {response.status_code}")
                logger.info(f"  Response Size: {len(str(response_json))} characters")
                logger.info(f"  Response Structure: {type(response_json)}")
            
            # Transform the real API response to our internal format
            rag_response = self._transform_response(question, response_json)
            
            if self.settings.debug:
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
    
    def _transform_response(self, question: str, api_response: List[dict]) -> RAGResponse:
        """Transform the real API response to our internal format.
        
        Args:
            question: The original question.
            api_response: The response from the real API.
            
        Returns:
            RAGResponse: Transformed response in our internal format.
        """
        chunks = []
        formatted_chunks = []
        
        # Process the API response structure
        for item in api_response:
            if "topSegments" in item:
                for segment in item["topSegments"]:
                    # Create KBChunk from segment data
                    chunk = KBChunk(
                        content=segment.get("segment", ""),
                        source=segment.get("file", "Unknown"),
                        confidence=segment.get("score", 0.0)
                    )
                    chunks.append(chunk)
                    
                    # Create formatted chunk with source citation
                    formatted_chunk = f"{chunk.content} (source: {chunk.source})"
                    formatted_chunks.append(formatted_chunk)
        
        return RAGResponse(
            question=question,
            chunks=chunks,
            formatted_chunks=formatted_chunks
        )
    
    def health_check(self) -> bool:
        """Check if the RAG service is healthy.
        
        Returns:
            bool: True if the service is healthy, False otherwise.
        """
        try:
            # For the real API, we'll do a simple test request
            test_payload = {"questions": ["test"]}
            url = f"{self.base_url}/topSegments?siteId={self.site_id}"
            
            response = self.session.post(
                url,
                json=test_payload,
                timeout=5  # Short timeout for health checks
            )
            return response.status_code == 200
            
        except Exception as e:
            logger.warning(f"RAG service health check failed: {e}")
            return False 