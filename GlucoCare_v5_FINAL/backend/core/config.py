# core/config.py
"""
WHY: Centralising all configuration in one place with Pydantic's BaseSettings
means secrets are NEVER hardcoded. If you ever see a raw string like
"postgres://..." in application code, that is a critical bug.
Pydantic BaseSettings reads from environment variables AND .env files,
giving us one source of truth for all config.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    # App
    APP_NAME: str = "GlucoCare API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # Supabase
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_JWT_SECRET: str  # Found in Supabase dashboard > Settings > API

    # Database (async connection string for asyncpg)
    DATABASE_URL: str  # postgresql+asyncpg://user:pass@host:5432/dbname

    # Security
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Firebase (for push notifications)
    FIREBASE_CREDENTIALS_PATH: str = "firebase-adminsdk.json"
    FIREBASE_PROJECT_ID: str = ""

    # ML Model
    MODEL_PATH: str = "ml/xgboost_diabetes_model.joblib"
    SCALER_PATH: str = "ml/feature_scaler.joblib"

    # CORS - list of allowed frontend origins
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://glucocare.app",  # your production domain
    ]

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    WHY lru_cache: Settings are read from disk/env once and cached.
    Without this, every request would re-read environment variables —
    slow and wasteful at scale.
    """
    return Settings()
