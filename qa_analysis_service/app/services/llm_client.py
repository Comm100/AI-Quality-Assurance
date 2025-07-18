"""LLM client for OpenAI API calls with robust error handling."""
import json
import logging
import re
import time
from typing import Dict, List, Optional

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

from ..config import settings

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Exception raised when LLM API calls fail."""
    pass


class LLMClient:
    """Client for interacting with OpenAI API with retry logic and error handling."""
    
    def __init__(self, api_key: str, model: Optional[str] = None, temperature: float = 0.0):
        """Initialize the LLM client.
        
        Args:
            api_key: OpenAI API key (required).
            model: Model to use for completions.
            temperature: Temperature for sampling.
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")
            
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model = model if model is not None else settings.openai_model
        self.temperature = temperature
        self.timeout = getattr(settings, 'openai_timeout', 60)
        self.max_retries = getattr(settings, 'max_retries', 3)
        self.retry_delay = getattr(settings, 'retry_delay', 1.0)
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception to check.
            
        Returns:
            True if the request should be retried, False otherwise.
        """
        # Retry on rate limits, temporary server errors, and network issues
        if isinstance(exception, openai.RateLimitError):
            return True
        if isinstance(exception, openai.APITimeoutError):
            return True
        if isinstance(exception, openai.InternalServerError):
            return True
        if isinstance(exception, openai.APIConnectionError):
            return True
        
        return False
    
    def _get_retry_delay(self, attempt: int, exception: Exception) -> float:
        """Calculate delay before retry.
        
        Args:
            attempt: Current attempt number (0-based).
            exception: The exception that triggered the retry.
            
        Returns:
            Delay in seconds.
        """
        # Use exponential backoff for rate limits
        if isinstance(exception, openai.RateLimitError):
            return (2 ** attempt) * self.retry_delay
        
        # Fixed delay for other retryable errors
        return self.retry_delay
    
    def safe_json_parse(self, text: str) -> Dict:
        """Safely parse JSON from LLM response.
        
        Args:
            text: The text containing JSON.
            
        Returns:
            Parsed JSON as dictionary.
            
        Raises:
            LLMClientError: If JSON parsing fails.
        """
        try:
            # Try to find JSON object in the text
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in response")
            
            return json.loads(match.group(0))
        except Exception as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response text: {text}")
            raise LLMClientError(f"Failed to parse JSON: {e}") from e
    
    def chat_completion(self, messages: List[Dict], response_format: Optional[Dict] = None) -> str:
        """Get a chat completion from OpenAI with retry logic.
        
        Args:
            messages: List of message dictionaries.
            response_format: Optional response format specification.
            
        Returns:
            The completion text.
            
        Raises:
            LLMClientError: If the API call fails after all retries.
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Calling OpenAI API (attempt {attempt + 1}/{self.max_retries + 1}) with model: {self.model}")
                
                # Build completion arguments
                completion_args = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "timeout": self.timeout,
                }
                
                # Add response format if specified
                if response_format:
                    completion_args["response_format"] = response_format
                
                # Make API call
                response: ChatCompletion = self.client.chat.completions.create(**completion_args)
                
                content = response.choices[0].message.content
                logger.info("OpenAI API call successful")
                logger.debug(f"Response length: {len(content) if content else 0} characters")
                
                if content is None:
                    raise LLMClientError("OpenAI API returned empty response")
                
                return content
                
            except openai.AuthenticationError as e:
                # Don't retry authentication errors
                error_msg = f"OpenAI authentication failed: Invalid API key"
                logger.error(error_msg)
                raise LLMClientError(error_msg) from e
                
            except openai.PermissionDeniedError as e:
                # Don't retry permission errors
                error_msg = f"OpenAI permission denied: {e}"
                logger.error(error_msg)
                raise LLMClientError(error_msg) from e
                
            except openai.BadRequestError as e:
                # Don't retry bad request errors (model not found, invalid params)
                error_msg = f"OpenAI bad request: {e}"
                logger.error(error_msg)
                raise LLMClientError(error_msg) from e
                
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e):
                    # Don't retry non-retryable errors
                    error_msg = f"OpenAI API call failed (non-retryable): {e}"
                    logger.error(error_msg)
                    raise LLMClientError(error_msg) from e
                
                if attempt < self.max_retries:
                    # Calculate delay and retry
                    delay = self._get_retry_delay(attempt, e)
                    logger.warning(f"OpenAI API call failed (attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    # Max retries exceeded
                    error_msg = f"OpenAI API call failed after {self.max_retries + 1} attempts: {e}"
                    logger.error(error_msg)
                    raise LLMClientError(error_msg) from e
        
        # Should never reach here, but just in case
        error_msg = f"OpenAI API call failed unexpectedly: {last_exception}"
        logger.error(error_msg)
        raise LLMClientError(error_msg) from last_exception
    
    def chat_completion_json(self, messages: List[Dict]) -> Dict:
        """Get a JSON response from chat completion.
        
        Args:
            messages: List of message dictionaries.
            
        Returns:
            Parsed JSON response as dictionary.
            
        Raises:
            LLMClientError: If the API call or JSON parsing fails.
        """
        # Request JSON response format
        content = self.chat_completion(
            messages,
            response_format={"type": "json_object"}
        )
        
        # Parse and return JSON
        return self.safe_json_parse(content) 