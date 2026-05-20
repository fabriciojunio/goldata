"""Logging estruturado com structlog — NUNCA loga dados pessoais ou secrets."""

import logging
import sys
from typing import Any

import structlog

# Campos que NUNCA devem aparecer em logs (LGPD + segurança)
_SENSITIVE_FIELDS = {
    "password", "senha", "secret", "secret_key", "api_key", "token",
    "authorization", "cpf", "rg", "email", "phone", "telefone",
    "birth_date", "data_nascimento", "credit_card",
}


def _redact_sensitive(
    logger: Any, method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Processor que redacta campos sensíveis antes de logar."""
    for key in list(event_dict.keys()):
        if key.lower() in _SENSITIVE_FIELDS:
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    """Configura structlog para toda a aplicação."""

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _redact_sensitive,
    ]

    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Retorna logger configurado para o módulo dado."""
    return structlog.get_logger(name)
