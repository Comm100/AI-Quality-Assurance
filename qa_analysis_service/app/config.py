"""Configuration settings for the QA Analysis Service."""
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Service configuration
    app_name: str = "QA Analysis Service"
    api_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # API Authentication
    qa_service_token: str = os.getenv("QA_SERVICE_TOKEN", "cc9dfc7473d3486dac06e1634d4ce38e")
    
    # RAG service configuration
    rag_service_url: str = os.getenv("RAG_SERVICE_URL", "https://mqapi.testing.comm100dev.io/vectorservice/aicopilots/05f11090-cd5d-4e3f-c131-08ddc57917f0")
    rag_service_timeout: int = int(os.getenv("RAG_SERVICE_TIMEOUT", "60"))
    rag_service_token: str = os.getenv("RAG_SERVICE_TOKEN", "cc9dfc7473d3486dac06e1634d4ce38e")
    rag_service_site_id: str = os.getenv("RAG_SERVICE_SITE_ID", "10001")
    
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