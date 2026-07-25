"""Testes da configuração do GolData."""

import pytest

from goldata.config import Settings, get_settings


def test_settings_default_values():
    s = Settings()
    assert s.project_name == "GolData"
    assert s.version == "1.0.0"
    assert s.environment == "development"


_PROD_SECRET = "x" * 40
_PROD_API_KEY = "prod-api-key-aleatoria"


def test_settings_environment_values():
    s = Settings(
        environment="production",
        secret_key=_PROD_SECRET,
        api_key=_PROD_API_KEY,
    )
    assert s.is_production is True
    assert s.is_development is False


def test_production_rejects_default_secret_key():
    with pytest.raises(Exception):
        Settings(environment="production", api_key=_PROD_API_KEY)


def test_production_rejects_default_api_key():
    with pytest.raises(Exception):
        Settings(environment="production", secret_key=_PROD_SECRET)


def test_production_accepts_custom_secrets():
    s = Settings(
        environment="production",
        secret_key=_PROD_SECRET,
        api_key=_PROD_API_KEY,
    )
    assert s.secret_key == _PROD_SECRET
    assert s.api_key == _PROD_API_KEY


def test_development_allows_default_secrets():
    s = Settings(environment="development")
    assert s.is_development is True


def test_settings_development_flag():
    s = Settings(environment="development")
    assert s.is_development is True
    assert s.is_production is False


def test_settings_default_database_url():
    s = Settings()
    assert "sqlite" in s.database_url or "postgresql" in s.database_url


def test_settings_rate_limit_positive():
    s = Settings()
    assert s.rate_limit_per_minute > 0


def test_settings_retention_days():
    s = Settings()
    assert s.data_retention_days == 365


def test_settings_secret_key_minimum_length():
    with pytest.raises(Exception):
        Settings(secret_key="short")


def test_settings_random_seed():
    s = Settings()
    assert s.random_seed == 42


def test_get_settings_returns_same_instance():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_settings_cors_origins_is_list():
    s = Settings()
    assert isinstance(s.cors_origins, list)
    assert len(s.cors_origins) >= 1


def test_settings_data_dir():
    s = Settings()
    assert s.data_dir == "./data"
