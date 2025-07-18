"""Configuration settings for the QA Analysis Service."""
import os

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Service configuration
    app_name: str = "QA Analysis Service"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # RAG service configuration
    rag_service_url: str = os.getenv("RAG_SERVICE_URL", "http://localhost:8002")
    
    # Chat data service configuration
    chat_data_service_url: str = os.getenv("CHAT_DATA_SERVICE_URL", "http://localhost:8001")
    
    # OpenAI configuration - Set your API key in environment variables
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    
    # Analysis configuration
    top_k_kb_chunks: int = int(os.getenv("TOP_K_KB_CHUNKS", "6"))
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create settings instance
settings = Settings() 