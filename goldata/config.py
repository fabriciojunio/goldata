"""Configuração central do GolData via pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações do GolData carregadas do ambiente / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Projeto
    project_name: str = "GolData"
    version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///goldata_dev.db",
        description="URL async do banco (asyncpg ou aiosqlite)",
    )
    database_url_sync: str = Field(
        default="sqlite:///goldata_dev.db",
        description="URL síncrona do banco (para migrações)",
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # Segurança
    secret_key: str = Field(
        default="goldata-dev-secret-key-change-in-production-32chars!!",
        description="Chave secreta para JWT e Fernet",
    )
    api_key: str = Field(
        default="goldata-dev-api-key",
        description="API Key para autenticação dos endpoints",
    )

    # API
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
    )
    rate_limit_per_minute: int = 60
    debug: bool = False

    # Dados
    data_dir: str = "./data"
    log_level: str = "INFO"

    # ML
    model_cache_dir: str = "./data/models"
    random_seed: int = 42

    # LGPD
    data_retention_days: int = 365

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("SECRET_KEY deve ter pelo menos 32 caracteres")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Singleton de configurações: mesma instância em toda a aplicação."""
    return Settings()
