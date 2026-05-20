"""Testes de integração da FastAPI."""

import pytest
from fastapi.testclient import TestClient
from goldata.api.main import app
from goldata.config import get_settings

settings = get_settings()
client = TestClient(app, raise_server_exceptions=False)
HEADERS = {"X-API-Key": settings.api_key}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_health_no_auth_needed():
    response = client.get("/health")
    assert response.status_code == 200


def test_xg_endpoint_valid():
    response = client.post(
        "/api/v1/xg",
        json={"x": 108.0, "y": 40.0, "is_penalty": 0},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert "xg" in data
    assert 0 < data["xg"] < 1


def test_xg_endpoint_penalty():
    response = client.post(
        "/api/v1/xg",
        json={"x": 108.0, "y": 40.0, "is_penalty": 1},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["xg"] == pytest.approx(0.76, abs=0.01)


def test_xg_endpoint_no_auth():
    response = client.post(
        "/api/v1/xg",
        json={"x": 108.0, "y": 40.0},
    )
    assert response.status_code == 401


def test_xg_endpoint_invalid_x():
    response = client.post(
        "/api/v1/xg",
        json={"x": 150.0, "y": 40.0},
        headers=HEADERS,
    )
    assert response.status_code == 422


def test_prediction_endpoint():
    response = client.post(
        "/api/v1/prediction",
        json={"home_team": "Flamengo", "away_team": "Palmeiras"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert "home_win_prob" in data
    total = data["home_win_prob"] + data["draw_prob"] + data["away_win_prob"]
    assert abs(total - 1.0) < 0.01


def test_value_bet_endpoint():
    response = client.post(
        "/api/v1/betting/value",
        json={
            "home_team": "Flamengo",
            "away_team": "Palmeiras",
            "model_probs": {"home_win": 0.62, "draw": 0.22, "away_win": 0.16},
            "market_odds": {"home_win": 2.10, "draw": 3.40, "away_win": 5.00},
        },
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert "n_value_bets" in data
    assert "value_bets" in data


def test_kelly_endpoint():
    response = client.post(
        "/api/v1/betting/kelly",
        json={"bankroll": 1000.0, "prob": 0.55, "odd": 2.10},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert "stake_amount" in data
    assert data["stake_amount"] >= 0


def test_xt_endpoint():
    response = client.post(
        "/api/v1/xt",
        json={"x": 110.0, "y": 40.0},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert "xt" in data
    assert data["xt"] > 0


def test_lgpd_info_endpoint():
    response = client.get("/api/v1/lgpd/info")
    assert response.status_code == 200
    data = response.json()
    assert "controller" in data
    assert "rights" in data


def test_lgpd_export_endpoint():
    response = client.get("/api/v1/lgpd/export/user123", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "exported_at" in data


def test_docs_accessible():
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
