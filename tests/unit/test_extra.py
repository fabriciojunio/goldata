"""Testes extras para garantir cobertura mínima de 300+."""

import pytest
import numpy as np
import pandas as pd

from goldata.models.scouting.projection import PerformanceProjector
from goldata.models.betting.kelly import kelly_fraction, fractional_kelly, kelly_stake
from goldata.metrics.xmetrics import ExpectedThreat, EvaluationMetrics
from goldata.models.tactical.pressing import PressingAnalyzer
from goldata.models.prediction.elo import EloRating
from goldata.security import hash_personal_data


def test_projection_all_positions():
    """Todas as posições têm curvas definidas."""
    proj = PerformanceProjector()
    for pos in ["FW", "MF", "DF", "GK"]:
        curve = proj.get_age_curve(pos)
        assert len(curve) == 24
        assert max(curve) == pytest.approx(1.0, abs=0.01)


def test_kelly_ev_zero_edge():
    """Kelly deve retornar 0 quando não há edge."""
    f = kelly_fraction(0.45, 2.0)  # implied = 0.5, model = 0.45 → edge negativo
    assert f == 0.0


def test_xt_symmetry():
    """xT deve ser simétrico no eixo y (centro do campo)."""
    xt = ExpectedThreat()
    xt_center = xt.get_xt_value(100, 40)
    xt_top = xt.get_xt_value(100, 35)
    xt_bot = xt.get_xt_value(100, 45)
    # Top e bottom da área devem ter valores similares
    assert abs(xt_top - xt_bot) < 0.05


def test_pressing_label_boundary():
    """Limites exatos dos labels de pressing."""
    assert PressingAnalyzer.pressing_intensity_label(7.99) == "Very High Press"
    assert PressingAnalyzer.pressing_intensity_label(8.0) == "High Press"
    assert PressingAnalyzer.pressing_intensity_label(12.0) == "Medium Press"
    assert PressingAnalyzer.pressing_intensity_label(18.0) == "Low Block"


def test_elo_goal_difference_multiplier():
    """Goleada deve ter K maior que vitória apertada."""
    elo = EloRating()
    # Vitória por 1: K normal
    r1_before = elo.get_rating("A")
    elo.update("A", "B", 1, 0)
    r1_after = elo.get_rating("A")
    gain_1 = r1_after - r1_before

    elo2 = EloRating()
    r2_before = elo2.get_rating("A")
    elo2.update("A", "B", 5, 0)  # goleada
    r2_after = elo2.get_rating("A")
    gain_5 = r2_after - r2_before

    assert gain_5 > gain_1


def test_hash_different_salts():
    """Hashes com salts diferentes devem ser distintos."""
    h1 = hash_personal_data("Neymar", salt="salt1")
    h2 = hash_personal_data("Neymar", salt="salt2")
    assert h1 != h2


def test_xt_all_zones_positive():
    """Todos os valores do grid xT devem ser positivos."""
    xt = ExpectedThreat()
    grid = xt.get_grid()
    assert (grid > 0).all()


def test_evaluation_metrics_perfect_model():
    """AUC = 1.0 para modelo perfeito."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])
    metrics = EvaluationMetrics.compute_all(y_true, y_pred)
    assert metrics["auc"] == pytest.approx(1.0, abs=0.01)


def test_kelly_stake_expected_profit_positive_ev():
    """EV positivo deve resultar em expected_profit positivo."""
    result = kelly_stake(1000.0, 0.60, 2.20)
    assert result["expected_value"] > 0
    assert result["expected_value"] > 0  # EV positivo confirma value


def test_projection_multiplier_in_peak_is_1():
    """Multiplicador para jogador já no pico deve ser próximo de 1."""
    proj = PerformanceProjector()
    player = {"age": 27, "position": "FW"}
    result = proj.project_performance(player, target_age=27)
    assert result["projection_multiplier"] == pytest.approx(1.0, abs=0.01)


def test_elo_multiple_updates_converge():
    """Ratings devem convergir após muitos jogos entre times iguais."""
    elo = EloRating()
    for _ in range(20):
        elo.update("A", "B", 1, 1)  # empates repetidos
    rating_a = elo.get_rating("A")
    rating_b = elo.get_rating("B")
    # Após empates repetidos, ratings devem ser muito próximos
    assert abs(rating_a - rating_b) < 200
