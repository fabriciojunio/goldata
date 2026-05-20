"""Exceções customizadas do GolData."""

from typing import Any


class GolDataError(Exception):
    """Exceção base do GolData."""

    def __init__(
        self,
        message: str,
        error_code: str = "GOLDATA_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code}, msg={self.message!r})"


class DataNotFoundError(GolDataError):
    """Dados não encontrados (time, jogador, partida, etc.)."""

    def __init__(self, resource: str, identifier: Any) -> None:
        super().__init__(
            message=f"{resource} não encontrado: {identifier}",
            error_code="DATA_NOT_FOUND",
            details={"resource": resource, "identifier": str(identifier)},
        )


class ModelNotTrainedError(GolDataError):
    """Modelo não foi treinado antes de usar predict."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            message=f"Modelo '{model_name}' não foi treinado. Chame .train() primeiro.",
            error_code="MODEL_NOT_TRAINED",
            details={"model_name": model_name},
        )


class InvalidInputError(GolDataError):
    """Input inválido — fora de range, tipo errado, schema incorreto."""

    def __init__(self, field: str, value: Any, reason: str) -> None:
        super().__init__(
            message=f"Input inválido para '{field}': {reason}. Valor recebido: {value}",
            error_code="INVALID_INPUT",
            details={"field": field, "value": str(value), "reason": reason},
        )


class DatabaseError(GolDataError):
    """Erro de banco de dados."""

    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(
            message=f"Erro de banco de dados em '{operation}': {reason}",
            error_code="DATABASE_ERROR",
            details={"operation": operation, "reason": reason},
        )


class APIError(GolDataError):
    """Erro de API externa (StatsBomb, FBref, etc.)."""

    def __init__(self, source: str, status_code: int | None = None, reason: str = "") -> None:
        super().__init__(
            message=f"Erro na API '{source}': {reason}",
            error_code="API_ERROR",
            details={"source": source, "status_code": status_code, "reason": reason},
        )


class SecurityError(GolDataError):
    """Erro de segurança — autenticação, autorização, rate limit."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Erro de segurança: {reason}",
            error_code="SECURITY_ERROR",
            details={"reason": reason},
        )


class InsufficientDataError(GolDataError):
    """Dados insuficientes para treinar modelo ou gerar análise."""

    def __init__(self, minimum: int, received: int, context: str = "") -> None:
        super().__init__(
            message=f"Dados insuficientes. Mínimo: {minimum}, recebido: {received}. {context}",
            error_code="INSUFFICIENT_DATA",
            details={"minimum": minimum, "received": received, "context": context},
        )
