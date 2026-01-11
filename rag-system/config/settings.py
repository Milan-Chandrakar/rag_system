from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # LLM Configuration
    groq_api_key: str
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0
    
    # Embeddings Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector Store Configuration
    chroma_persist_dir: Path = Path("./data/vectordb")
    collection_name: str = "documents"
    
    # Retrieval Configuration
    top_k_results: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Gradio Configuration
    gradio_share: bool = True
    gradio_server_port: int = 7860
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
