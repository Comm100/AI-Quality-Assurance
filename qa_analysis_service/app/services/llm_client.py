"""LLM client for OpenAI API calls."""
import os
import re
import json
import logging
from typing import Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion


logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Exception raised when LLM API calls fail."""
    pass


class LLMClient:
    """Client for interacting with OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.0):
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
        self.model = model
        self.temperature = temperature
    
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
        """Get a chat completion from OpenAI.
        
        Args:
            messages: List of message dictionaries.
            response_format: Optional response format specification.
            
        Returns:
            The completion text.
            
        Raises:
            LLMClientError: If the API call fails.
        """
        try:
            logger.info(f"Calling OpenAI API with model: {self.model}")
            logger.debug(f"Messages: {len(messages)} messages")
            logger.debug(f"API Key present: {bool(self.api_key and len(self.api_key) > 10)}")
            
            # Build completion arguments
            completion_args = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            # Add response format if specified
            if response_format:
                completion_args["response_format"] = response_format
            
            # Make API call
            response: ChatCompletion = self.client.chat.completions.create(**completion_args)
            
            content = response.choices[0].message.content
            logger.info("OpenAI API call successful")
            logger.debug(f"Response length: {len(content) if content else 0} characters")
            
            return content
            
        except Exception as e:
            error_msg = f"OpenAI API call failed: {e}"
            logger.error(error_msg)
            logger.error(f"Model used: {self.model}")
            logger.error(f"API key length: {len(self.api_key) if self.api_key else 0}")
            raise LLMClientError(error_msg) from e
    
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