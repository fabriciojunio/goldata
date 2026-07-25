"""Configuração central do GolData via pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Valores padrão de desenvolvimento que jamais podem ir para produção.
# Não são segredos reais: o validador abaixo rejeita ambos quando ENVIRONMENT=production.
_DEFAULT_SECRET_KEY = "goldata-dev-secret-key-change-in-production-32chars!!"  # nosec B105
_DEFAULT_API_KEY = "goldata-dev-api-key"


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
        default=_DEFAULT_SECRET_KEY,
        description="Chave secreta para JWT e Fernet",
    )
    api_key: str = Field(
        default=_DEFAULT_API_KEY,
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

    @model_validator(mode="after")
    def reject_default_secrets_in_production(self) -> "Settings":
        """Impede subir em produção com os segredos padrão de desenvolvimento."""
        if self.environment == "production":
            if self.secret_key == _DEFAULT_SECRET_KEY:
                raise ValueError(
                    "SECRET_KEY padrão detectada em produção. Defina a variável "
                    "de ambiente SECRET_KEY com um valor aleatório (openssl rand -hex 32)."
                )
            if self.api_key == _DEFAULT_API_KEY:
                raise ValueError(
                    "API_KEY padrão detectada em produção. Defina a variável "
                    "de ambiente API_KEY com um valor aleatório."
                )
        return self

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
