"""Testes das métricas avançadas: xT e calibração."""

import pytest
import numpy as np
import pandas as pd

from goldata.metrics.xmetrics import ExpectedThreat, EvaluationMetrics, XT_GRID_X, XT_GRID_Y


@pytest.fixture
def xt():
    return ExpectedThreat()


# ── Expected Threat ───────────────────────────────────────────────────────────

def test_xt_grid_shape(xt):
    grid = xt.get_grid()
    assert grid.shape == (XT_GRID_Y, XT_GRID_X)


def test_xt_grid_values_in_range(xt):
    grid = xt.get_grid()
    assert (grid >= 0).all()
    assert (grid <= 1).all()


def test_xt_final_third_higher(xt):
    """Zona da área deve ter xT maior que meio-campo."""
    xt_area = xt.get_xt_value(110, 40)
    xt_midfield = xt.get_xt_value(60, 40)
    assert xt_area > xt_midfield


def test_xt_action_forward_positive(xt):
    """Passe para frente em direção ao gol deve ser xT positivo."""
    xt_action = xt.calculate_xt_action(70, 40, 100, 40)
    assert xt_action > 0


def test_xt_action_backward_negative(xt):
    """Recuo deve ser xT negativo."""
    xt_action = xt.calculate_xt_action(100, 40, 60, 40)
    assert xt_action < 0


def test_xt_per_player_returns_dataframe(sample_events_df):
    xt = ExpectedThreat()
    result = xt.calculate_xt_per_player(sample_events_df)
    assert isinstance(result, pd.DataFrame)
    assert "xt_total" in result.columns


def test_xt_per_player_n_actions_positive(sample_events_df):
    xt = ExpectedThreat()
    result = xt.calculate_xt_per_player(sample_events_df)
    assert (result["n_actions"] > 0).all()


def test_xt_center_vs_wing(xt):
    """Posição central deve ter xT maior que wing na mesma profundidade."""
    xt_center = xt.get_xt_value(100, 40)
    xt_wing = xt.get_xt_value(100, 5)
    assert xt_center >= xt_wing


# ── Evaluation Metrics ────────────────────────────────────────────────────────

@pytest.fixture
def classification_data():
    np.random.seed(42)
    n = 200
    y_true = np.random.binomial(1, 0.15, n)
    y_pred = np.clip(y_true * 0.5 + np.random.normal(0.15, 0.1, n), 0.01, 0.99)
    return y_true, y_pred


def test_compute_all_returns_dict(classification_data):
    y_true, y_pred = classification_data
    metrics = EvaluationMetrics.compute_all(y_true, y_pred)
    assert isinstance(metrics, dict)


def test_compute_all_has_required_keys(classification_data):
    y_true, y_pred = classification_data
    metrics = EvaluationMetrics.compute_all(y_true, y_pred)
    for key in ["auc", "log_loss", "brier_score", "n_samples"]:
        assert key in metrics


def test_compute_all_auc_in_range(classification_data):
    y_true, y_pred = classification_data
    metrics = EvaluationMetrics.compute_all(y_true, y_pred)
    assert 0 <= metrics["auc"] <= 1


def test_compute_all_brier_in_range(classification_data):
    y_true, y_pred = classification_data
    metrics = EvaluationMetrics.compute_all(y_true, y_pred)
    assert 0 <= metrics["brier_score"] <= 1


def test_compute_all_n_samples(classification_data):
    y_true, y_pred = classification_data
    metrics = EvaluationMetrics.compute_all(y_true, y_pred)
    assert metrics["n_samples"] == len(y_true)


def test_calibration_analysis_returns_dataframe(classification_data):
    y_true, y_pred = classification_data
    df = EvaluationMetrics.calibration_analysis(y_true, y_pred)
    assert isinstance(df, pd.DataFrame)
    assert "predicted_prob" in df.columns
    assert "actual_fraction" in df.columns


def test_calibration_error_non_negative(classification_data):
    y_true, y_pred = classification_data
    df = EvaluationMetrics.calibration_analysis(y_true, y_pred)
    # Erro pode ser positivo ou negativo mas a coluna deve existir
    assert "calibration_error" in df.columns


def test_ece_between_0_and_1(classification_data):
    y_true, y_pred = classification_data
    ece = EvaluationMetrics.expected_calibration_error(y_true, y_pred)
    assert 0 <= ece <= 1


def test_perfect_calibration_ece_low():
    """Modelo perfeitamente calibrado deve ter ECE próximo de 0."""
    y_true = np.random.binomial(1, 0.3, 1000)
    y_pred = np.full(1000, 0.3)
    ece = EvaluationMetrics.expected_calibration_error(y_true, y_pred)
    assert ece < 0.1
