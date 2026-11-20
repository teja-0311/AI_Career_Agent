"""Centralized, validated configuration system for AI Career Agent."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppSettings(BaseModel):
    # API Keys (read only from environment)
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    tavily_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    api_auth_secret: str = Field(default_factory=lambda: os.getenv("API_AUTH_SECRET", "super_secret_career_agent_jwt_key_32chars_min!"))
    jwt_algorithm: str = Field(default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256"))
    jwt_expiration_minutes: int = Field(default_factory=lambda: int(os.getenv("JWT_EXPIRATION_MINUTES", "120")))
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    allow_fallback_executor: bool = Field(default_factory=lambda: os.getenv("ALLOW_FALLBACK_EXECUTOR", "false").lower() in ("true", "1"))

    # Explicit Model Backends
    chat_model: str = Field(default_factory=lambda: os.getenv("CHAT_MODEL", "gemini-1.5-flash"))
    resume_parser_backend: str = Field(default_factory=lambda: os.getenv("RESUME_PARSER_BACKEND", "rule_based"))  # 'gemini' or 'rule_based'
    embedding_backend: str = Field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "mock"))  # 'gemini' or 'mock'
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"))
    embedding_dim: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768")))

    # Explicit Storage Backends
    vector_store_backend: str = Field(default_factory=lambda: os.getenv("VECTOR_STORE_BACKEND", "chroma"))  # 'chroma' or 'memory'
    chroma_dir: Path = Field(default_factory=lambda: Path(os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))))
    collection_name: str = Field(default_factory=lambda: os.getenv("COLLECTION_NAME", "job_postings"))
    sqlite_db_path: Path = Field(default_factory=lambda: Path(os.getenv("SQLITE_DB_PATH", str(PROJECT_ROOT / "career_agent_memory.db"))))

    # Retrieval & Ranking Parameters
    retriever_k: int = Field(default_factory=lambda: int(os.getenv("RETRIEVER_K", "10")))
    weight_semantic: float = Field(default_factory=lambda: float(os.getenv("WEIGHT_SEMANTIC", "0.45")))
    weight_skills: float = Field(default_factory=lambda: float(os.getenv("WEIGHT_SKILLS", "0.25")))
    weight_experience: float = Field(default_factory=lambda: float(os.getenv("WEIGHT_EXPERIENCE", "0.15")))
    weight_location: float = Field(default_factory=lambda: float(os.getenv("WEIGHT_LOCATION", "0.15")))

    # Security & Server
    cors_origins: str = Field(default_factory=lambda: os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000"))
    upload_max_bytes: int = Field(default_factory=lambda: int(os.getenv("UPLOAD_MAX_BYTES", str(10 * 1024 * 1024))))  # 10 MB

    def get_cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def is_gemini_active(self) -> bool:
        return bool(self.gemini_api_key and not self.gemini_api_key.startswith("your-"))


# Global validated configuration singleton
settings = AppSettings()
