"""Configuration settings for the QA Analysis Service."""
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Service configuration
    app_name: str = "QA Analysis Service"
    api_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # RAG service configuration
    rag_service_url: str = os.getenv("RAG_SERVICE_URL", "http://localhost:8002")
    rag_service_timeout: int = int(os.getenv("RAG_SERVICE_TIMEOUT", "60"))
    
    # Chat data service configuration
    chat_data_service_url: str = os.getenv("CHAT_DATA_SERVICE_URL", "http://localhost:8001")
    
    # OpenAI configuration - Set your API key in environment variables
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    openai_timeout: int = int(os.getenv("OPENAI_TIMEOUT", "60"))
    
    # Analysis configuration
    top_k_kb_chunks: int = int(os.getenv("TOP_K_KB_CHUNKS", "6"))
    
    # Retry configuration
    max_retries: int = int(os.getenv("MAX_RETRIES", "10"))
    retry_delay: float = float(os.getenv("RETRY_DELAY", "1.0"))
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create settings instance
settings = Settings() 