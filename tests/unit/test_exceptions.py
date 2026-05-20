"""Testes das exceções customizadas."""

import pytest
from goldata.exceptions import (
    APIError,
    DatabaseError,
    DataNotFoundError,
    GolDataError,
    InsufficientDataError,
    InvalidInputError,
    ModelNotTrainedError,
    SecurityError,
)


def test_goldata_error_base():
    err = GolDataError("erro base", "BASE_ERROR", {"key": "val"})
    assert err.message == "erro base"
    assert err.error_code == "BASE_ERROR"
    assert err.details == {"key": "val"}


def test_goldata_error_defaults():
    err = GolDataError("msg")
    assert err.error_code == "GOLDATA_ERROR"
    assert err.details == {}


def test_data_not_found_error():
    err = DataNotFoundError("Team", "Flamengo")
    assert "Flamengo" in err.message
    assert err.error_code == "DATA_NOT_FOUND"
    assert err.details["resource"] == "Team"


def test_model_not_trained_error():
    err = ModelNotTrainedError("BasicXGModel")
    assert "BasicXGModel" in err.message
    assert err.error_code == "MODEL_NOT_TRAINED"
    assert err.details["model_name"] == "BasicXGModel"


def test_invalid_input_error():
    err = InvalidInputError("xg", 1.5, "deve ser entre 0 e 1")
    assert "xg" in err.message
    assert err.error_code == "INVALID_INPUT"
    assert err.details["field"] == "xg"


def test_database_error():
    err = DatabaseError("INSERT", "unique constraint violated")
    assert "INSERT" in err.message
    assert err.error_code == "DATABASE_ERROR"


def test_api_error():
    err = APIError("StatsBomb", 503, "service unavailable")
    assert "StatsBomb" in err.message
    assert err.error_code == "API_ERROR"
    assert err.details["status_code"] == 503


def test_security_error():
    err = SecurityError("invalid api key")
    assert err.error_code == "SECURITY_ERROR"


def test_insufficient_data_error():
    err = InsufficientDataError(minimum=100, received=30, context="xG model training")
    assert "100" in err.message
    assert "30" in err.message
    assert err.error_code == "INSUFFICIENT_DATA"


def test_all_errors_are_goldata_error():
    errors = [
        DataNotFoundError("x", "y"),
        ModelNotTrainedError("model"),
        InvalidInputError("f", 0, "r"),
        DatabaseError("op", "reason"),
        APIError("src"),
        SecurityError("r"),
        InsufficientDataError(10, 5),
    ]
    for err in errors:
        assert isinstance(err, GolDataError)
        assert isinstance(err, Exception)


def test_error_repr():
    err = GolDataError("test", "TEST_CODE")
    r = repr(err)
    assert "TEST_CODE" in r
