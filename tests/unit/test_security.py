"""Testes do módulo de segurança e LGPD."""

import pytest
from goldata.security import (
    EncryptionManager,
    LGPDAuditLog,
    anonymize_player_data,
    cleanup_expired_data,
    delete_user_data,
    export_user_data,
    generate_consent_record,
    hash_personal_data,
    sanitize_dict,
    sanitize_string,
    validate_api_key,
)
from goldata.exceptions import SecurityError


SECRET = "goldata-test-secret-key-32chars!!"


# ── Criptografia ──────────────────────────────────────────────────────────────

def test_encrypt_decrypt_roundtrip():
    mgr = EncryptionManager(SECRET)
    original = "dado sensível do jogador"
    encrypted = mgr.encrypt(original)
    assert encrypted != original
    assert mgr.decrypt(encrypted) == original


def test_encrypt_produces_different_tokens_same_input():
    mgr = EncryptionManager(SECRET)
    t1 = mgr.encrypt("goldata")
    t2 = mgr.encrypt("goldata")
    # Fernet usa IV aleatório: tokens diferentes mesmo para mesmo input
    assert t1 != t2


def test_decrypt_invalid_token_raises_security_error():
    mgr = EncryptionManager(SECRET)
    with pytest.raises(SecurityError):
        mgr.decrypt("token-invalido-qualquer-coisa")


def test_different_keys_cannot_decrypt():
    mgr1 = EncryptionManager("goldata-test-secret-key-32chars!!")
    mgr2 = EncryptionManager("outra-chave-diferente-32chars!!!!")
    encrypted = mgr1.encrypt("segredo")
    with pytest.raises(SecurityError):
        mgr2.decrypt(encrypted)


# ── Sanitização ───────────────────────────────────────────────────────────────

def test_sanitize_removes_sql_injection():
    dirty = "'; DROP TABLE players; --"
    clean = sanitize_string(dirty)
    assert "DROP" not in clean
    assert "--" not in clean


def test_sanitize_removes_xss():
    dirty = "<script>alert('xss')</script>"
    clean = sanitize_string(dirty)
    assert "<script>" not in clean


def test_sanitize_respects_max_length():
    long_string = "a" * 1000
    clean = sanitize_string(long_string, max_length=100)
    assert len(clean) <= 100


def test_sanitize_normal_string_unchanged():
    normal = "Flamengo 2 x 1 Palmeiras"
    clean = sanitize_string(normal)
    assert "Flamengo" in clean
    assert "Palmeiras" in clean


def test_sanitize_dict_recursive():
    data = {"name": "<b>jogador</b>", "team": "Flamengo", "nested": {"value": "SELECT *"}}
    clean = sanitize_dict(data)
    assert "<b>" not in clean["name"]
    assert "Flamengo" in clean["team"]


# ── API Key ───────────────────────────────────────────────────────────────────

def test_validate_api_key_correct():
    assert validate_api_key("my-api-key", "my-api-key") is True


def test_validate_api_key_incorrect():
    assert validate_api_key("wrong-key", "correct-key") is False


def test_validate_api_key_empty():
    assert validate_api_key("", "some-key") is False


# ── LGPD ─────────────────────────────────────────────────────────────────────

def test_hash_personal_data_is_deterministic():
    h1 = hash_personal_data("Neymar Jr")
    h2 = hash_personal_data("Neymar Jr")
    assert h1 == h2


def test_hash_personal_data_different_inputs():
    h1 = hash_personal_data("Neymar Jr")
    h2 = hash_personal_data("Vinicius Jr")
    assert h1 != h2


def test_hash_personal_data_length():
    h = hash_personal_data("qualquer nome")
    assert len(h) == 64  # SHA-256 hex


def test_anonymize_player_data_hashes_name():
    data = {"name": "Neymar Jr", "display_name": "Neymar", "goals": 5}
    result = anonymize_player_data(data)
    assert result["name"] != "Neymar Jr"
    assert len(result["name"]) == 64
    assert result["goals"] == 5


def test_anonymize_player_data_removes_cpf():
    data = {"cpf": "123.456.789-00", "display_name": "Jogador"}
    result = anonymize_player_data(data)
    assert "cpf" not in result


def test_anonymize_player_data_birth_date_year_only():
    data = {"birth_date": "1992-02-05", "display_name": "Jogador"}
    result = anonymize_player_data(data)
    assert result["birth_date"] == "1992-01-01"


def test_generate_consent_record_fields():
    record = generate_consent_record(
        user_id="user123",
        purpose="analytics esportivo",
        data_categories=["performance", "position"],
    )
    assert "user_id_hash" in record
    assert record["purpose"] == "analytics esportivo"
    assert "timestamp" in record
    assert record["retention_days"] == 365
    assert "lgpd_article" in record


def test_export_user_data_structure():
    result = export_user_data("user456")
    assert "user_id_hash" in result
    assert "exported_at" in result
    assert "lgpd_article" in result


def test_delete_user_data_returns_true():
    assert delete_user_data("user789") is True


def test_cleanup_expired_data_returns_int():
    result = cleanup_expired_data(365)
    assert isinstance(result, int)


def test_lgpd_audit_log_runs_without_error():
    # Apenas verificar que não lança exceção
    LGPDAuditLog.log_access(
        action="read",
        entity_type="player",
        entity_id="player_123",
        purpose="analytics",
    )
