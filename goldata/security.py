"""Módulo de segurança do GolData: autenticação, LGPD, criptografia."""

import hashlib
import html
import re
from datetime import datetime, timezone
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from goldata.exceptions import SecurityError
from goldata.logging_config import get_logger

logger = get_logger(__name__)

# Constante LGPD
DATA_RETENTION_DAYS: int = 365

# Padrões de caracteres perigosos para sanitização
_SQL_INJECTION_PATTERNS = re.compile(
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b|--|;|/\*|\*/)",
    re.IGNORECASE,
)
_XSS_PATTERNS = re.compile(r"<[^>]*>", re.IGNORECASE)


# ── Criptografia ──────────────────────────────────────────────────────────────

class EncryptionManager:
    """Gerencia criptografia simétrica com Fernet."""

    def __init__(self, secret_key: str) -> None:
        # Derivar chave Fernet de 32 bytes a partir do secret_key
        key_bytes = hashlib.sha256(secret_key.encode()).digest()
        import base64
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        self._fernet = Fernet(fernet_key)

    def encrypt(self, data: str) -> str:
        """Encripta uma string e retorna token base64."""
        return self._fernet.encrypt(data.encode()).decode()

    def decrypt(self, token: str) -> str:
        """Decripta token. Lança SecurityError se inválido."""
        try:
            return self._fernet.decrypt(token.encode()).decode()
        except (InvalidToken, Exception) as e:
            raise SecurityError(f"Token inválido ou corrompido: {e}") from e


# ── Sanitização de Input ──────────────────────────────────────────────────────

def sanitize_string(value: str, max_length: int = 500) -> str:
    """Remove SQL injection, XSS e caracteres perigosos."""
    if not isinstance(value, str):
        return str(value)
    cleaned = html.escape(value)
    cleaned = _SQL_INJECTION_PATTERNS.sub("", cleaned)
    cleaned = _XSS_PATTERNS.sub("", cleaned)
    return cleaned[:max_length].strip()


def sanitize_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitiza recursivamente todos os valores string de um dict."""
    result = {}
    for k, v in data.items():
        if isinstance(v, str):
            result[k] = sanitize_string(v)
        elif isinstance(v, dict):
            result[k] = sanitize_dict(v)
        else:
            result[k] = v
    return result


# ── API Key ────────────────────────────────────────────────────────────────────

def validate_api_key(provided_key: str, expected_key: str) -> bool:
    """Valida API key usando comparação em tempo constante (anti-timing attack)."""
    import hmac
    return hmac.compare_digest(provided_key.encode(), expected_key.encode())


# ── LGPD Compliance ────────────────────────────────────────────────────────────

def hash_personal_data(value: str, salt: str = "goldata-lgpd") -> str:
    """Hash SHA-256 de dado pessoal para anonimização (Art. 12 LGPD)."""
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()


def anonymize_player_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Anonimiza dados pessoais de jogador conforme LGPD.
    - Nome real → hash
    - Data de nascimento exata → apenas ano
    - Documentos → removidos
    """
    anonymized = data.copy()
    pii_fields = ["name", "nome", "full_name", "birth_date", "cpf", "rg", "email", "phone"]

    for field in pii_fields:
        if field in anonymized:
            value = anonymized[field]
            if field in ("birth_date", "data_nascimento") and value:
                # Manter apenas o ano
                try:
                    year = str(value)[:4]
                    anonymized[field] = f"{year}-01-01"
                except Exception:
                    anonymized[field] = None
            elif field in ("cpf", "rg", "email", "phone"):
                del anonymized[field]
            elif isinstance(value, str) and value:
                anonymized[field] = hash_personal_data(value)

    return anonymized


def generate_consent_record(
    user_id: str,
    purpose: str,
    data_categories: list[str],
) -> dict[str, Any]:
    """
    Gera registro de consentimento conforme Art. 7 e 9 da LGPD.
    """
    return {
        "user_id_hash": hash_personal_data(user_id),
        "purpose": purpose,
        "data_categories": data_categories,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retention_days": DATA_RETENTION_DAYS,
        "legal_basis": "consent",
        "lgpd_article": "Art. 7, I - LGPD",
    }


def export_user_data(user_id: str) -> dict[str, Any]:
    """
    Gera estrutura de exportação de dados do usuário (Art. 18, V - LGPD).
    Em produção, buscar do banco de dados.
    """
    return {
        "user_id_hash": hash_personal_data(user_id),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "lgpd_article": "Art. 18, V - Portabilidade",
        "data": {},  # Em produção: buscar do DB
        "message": "Dados exportados conforme direito de portabilidade (LGPD Art. 18, V)",
    }


def delete_user_data(user_id: str) -> bool:
    """
    Marca dados do usuário para eliminação (Art. 18, VI - LGPD).
    Retorna True se operação foi registrada com sucesso.
    """
    logger.info(
        "lgpd_deletion_requested",
        user_hash=hash_personal_data(user_id),
        timestamp=datetime.now(timezone.utc).isoformat(),
        article="Art. 18, VI - Eliminação",
    )
    return True


def cleanup_expired_data(retention_days: int = DATA_RETENTION_DAYS) -> int:
    """
    Remove dados com retenção expirada.
    Retorna número de registros removidos.
    Em produção: executar query DELETE WHERE created_at < NOW() - retention_days.
    """
    logger.info("lgpd_cleanup_triggered", retention_days=retention_days)
    return 0  # Em produção: retornar contagem real


# ── Audit Log ─────────────────────────────────────────────────────────────────

class LGPDAuditLog:
    """Registra todo acesso a dados pessoais conforme LGPD."""

    @staticmethod
    def log_access(
        action: str,
        entity_type: str,
        entity_id: str,
        purpose: str,
        user_identifier: str = "system",
    ) -> None:
        """Registra acesso a dados pessoais."""
        logger.info(
            "lgpd_data_access",
            action=action,
            entity_type=entity_type,
            entity_id_hash=hash_personal_data(entity_id),
            purpose=purpose,
            user_hash=hash_personal_data(user_identifier),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
