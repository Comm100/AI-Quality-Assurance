"""Configuration settings for QA Analysis Service."""
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # External Service URLs
    rag_service_url: str = Field(
        default="http://localhost:8002",
        description="URL for the RAG service that provides reference answers"
    )
    chat_data_service_url: str = Field(
        default="http://localhost:8001", 
        description="URL for the chat data service that provides transcripts"
    )
    
    # Service Configuration
    host: str = Field(default="0.0.0.0", description="Host to bind the service to")
    port: int = Field(default=8000, description="Port to run the service on")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # API Configuration
    api_title: str = Field(default="QA Analysis Service", description="API title")
    api_version: str = Field(default="1.0.0", description="API version")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings() 