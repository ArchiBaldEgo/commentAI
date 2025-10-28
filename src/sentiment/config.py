from __future__ import annotations

import os
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    model_dir: str = Field(default=os.getenv("MODEL_DIR", "models/online"))
    api_key: Optional[str] = Field(default=os.getenv("API_KEY"))
    cors_origins: List[str] = Field(default_factory=lambda: (os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"]))
    uvicorn_workers: int = int(os.getenv("UVICORN_WORKERS", "2"))
    uvicorn_timeout: int = int(os.getenv("UVICORN_TIMEOUT", "60"))
    app_name: str = "Sentiment Service"
    app_version: str = os.getenv("APP_VERSION", "1.0.0")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()
