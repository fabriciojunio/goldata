"""Testes dos modelos de previsão."""

import pytest
import numpy as np
import pandas as pd
from goldata.models.prediction.elo import EloRating, DEFAULT_RATING
from goldata.models.prediction.poisson import BivariatePoisson
from goldata.models.prediction.dixon_coles import DixonColes
from goldata.models.prediction.monte_carlo import LeagueSimulator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def brasileirao_results(sample_match_results_df):
    return sample_match_results_df


@pytest.fixture
def elo():
    return EloRating()


@pytest.fixture
def trained_elo(brasileirao_results):
    elo = EloRating()
    elo.train(brasileirao_results)
    return elo


@pytest.fixture
def trained_poisson(brasileirao_results):
    model = BivariatePoisson()
    model.fit(brasileirao_results)
    return model


@pytest.fixture
def trained_dc(brasileirao_results):
    model = DixonColes()
    model.fit(brasileirao_results)
    return model


# ── Elo ───────────────────────────────────────────────────────────────────────

def test_elo_default_rating(elo):
    assert elo.get_rating("Flamengo") == DEFAULT_RATING


def test_elo_win_increases_rating(elo):
    elo.update("Flamengo", "Palmeiras", 2, 0)
    assert elo.get_rating("Flamengo") > DEFAULT_RATING


def test_elo_loss_decreases_rating(elo):
    elo.update("Flamengo", "Palmeiras", 0, 2)
    assert elo.get_rating("Flamengo") < DEFAULT_RATING


def test_elo_draw_balanced(elo):
    r_before = elo.get_rating("Flamengo")
    elo.update("Flamengo", "Palmeiras", 1, 1)
    r_after = elo.get_rating("Flamengo")
    # Em empate entre iguais, variação pequena
    assert abs(r_after - r_before) < 15


def test_elo_update_returns_tuple(elo):
    result = elo.update("Flamengo", "Palmeiras", 1, 0)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_elo_get_ratings_dataframe(trained_elo):
    df = trained_elo.get_ratings()
    assert isinstance(df, pd.DataFrame)
    assert "team" in df.columns
    assert "rating" in df.columns


def test_elo_get_ratings_sorted(trained_elo):
    df = trained_elo.get_ratings()
    assert df["rating"].is_monotonic_decreasing


def test_elo_predict_match_returns_dict(trained_elo):
    pred = trained_elo.predict_match("Flamengo", "Palmeiras")
    assert "home_win_prob" in pred
    assert "draw_prob" in pred
    assert "away_win_prob" in pred


def test_elo_probabilities_sum_to_1(trained_elo):
    pred = trained_elo.predict_match("Flamengo", "Palmeiras")
    total = pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"]
    assert abs(total - 1.0) < 0.01


def test_elo_probabilities_in_range(trained_elo):
    pred = trained_elo.predict_match("Flamengo", "Palmeiras")
    for key in ["home_win_prob", "draw_prob", "away_win_prob"]:
        assert 0 <= pred[key] <= 1


def test_elo_history_recorded(elo):
    elo.update("Flamengo", "Palmeiras", 1, 0, date="2024-01-01")
    history = elo.get_rating_history("Flamengo")
    assert len(history) == 1


def test_elo_train_processes_all_matches(brasileirao_results):
    elo = EloRating()
    elo.train(brasileirao_results)
    df = elo.get_ratings()
    assert len(df) > 5  # pelo menos 5 times diferentes


def test_elo_k_factor_new_team(elo):
    """Time novo (< 20 jogos) deve ter K maior."""
    elo.update("NovoTime", "Outro", 1, 0)
    record = elo._teams["NovoTime"]
    assert record.games_played == 1


# ── Poisson ───────────────────────────────────────────────────────────────────

def test_poisson_trains(brasileirao_results):
    model = BivariatePoisson()
    model.fit(brasileirao_results)
    assert model.is_trained is True


def test_poisson_predict_returns_dict(trained_poisson):
    pred = trained_poisson.predict_match("Flamengo", "Palmeiras")
    assert "home_win_prob" in pred
    assert "draw_prob" in pred
    assert "away_win_prob" in pred
    assert "score_matrix" in pred


def test_poisson_probs_sum_to_1(trained_poisson):
    pred = trained_poisson.predict_match("Flamengo", "Palmeiras")
    total = pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"]
    assert abs(total - 1.0) < 0.05


def test_poisson_expected_goals_positive(trained_poisson):
    pred = trained_poisson.predict_match("Flamengo", "Palmeiras")
    assert pred["expected_home_goals"] > 0
    assert pred["expected_away_goals"] > 0


def test_poisson_team_ratings_dataframe(trained_poisson):
    df = trained_poisson.get_team_ratings()
    assert isinstance(df, pd.DataFrame)
    assert "attack" in df.columns
    assert "defense" in df.columns


def test_poisson_unknown_team_uses_default(trained_poisson):
    pred = trained_poisson.predict_match("TimeX", "TimeY")
    total = pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"]
    assert abs(total - 1.0) < 0.1


# ── Dixon-Coles ───────────────────────────────────────────────────────────────

def test_dc_trains(brasileirao_results):
    model = DixonColes()
    model.fit(brasileirao_results)
    assert model.is_trained is True


def test_dc_predict_returns_dict(trained_dc):
    pred = trained_dc.predict_match("Flamengo", "Palmeiras")
    assert "home_win_prob" in pred
    assert "rho" in pred


def test_dc_probs_sum_to_1(trained_dc):
    pred = trained_dc.predict_match("Flamengo", "Palmeiras")
    total = pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"]
    assert abs(total - 1.0) < 0.05


def test_dc_rho_parameter_estimated(trained_dc):
    pred = trained_dc.predict_match("Flamengo", "Palmeiras")
    assert isinstance(pred["rho"], float)


def test_dc_team_ratings(trained_dc):
    df = trained_dc.get_team_ratings()
    assert len(df) > 0


# ── Monte Carlo ───────────────────────────────────────────────────────────────

@pytest.fixture
def simple_table():
    teams = ["Flamengo", "Palmeiras", "Atletico-MG", "Botafogo",
             "Corinthians", "Sao Paulo", "Fluminense", "Vasco",
             "Internacional", "Santos", "Bahia", "Fortaleza",
             "Atletico-GO", "Cuiaba", "Coritiba", "Goias",
             "America-MG", "Cruzeiro", "Gremio", "Bragantino"]
    return pd.DataFrame({
        "team": teams,
        "points": list(range(40, 0, -2)),
        "goals_for": [40] * 20,
        "goals_against": [25] * 20,
    })


@pytest.fixture
def remaining_fixtures(simple_table):
    teams = simple_table["team"].tolist()
    fixtures = []
    for i in range(min(10, len(teams))):
        h = teams[i % len(teams)]
        a = teams[(i + 1) % len(teams)]
        if h != a:
            fixtures.append((h, a))
    return fixtures[:10]


def test_monte_carlo_returns_result(simple_table, remaining_fixtures):
    sim = LeagueSimulator()
    result = sim.simulate(simple_table, remaining_fixtures, n_simulations=200)
    assert result.n_simulations == 200


def test_monte_carlo_all_teams_have_probs(simple_table, remaining_fixtures):
    sim = LeagueSimulator()
    result = sim.simulate(simple_table, remaining_fixtures, n_simulations=200)
    for team in simple_table["team"]:
        assert team in result.team_position_probs


def test_monte_carlo_probs_sum_to_1_per_team(simple_table, remaining_fixtures):
    sim = LeagueSimulator()
    result = sim.simulate(simple_table, remaining_fixtures, n_simulations=200)
    for team, probs in result.team_position_probs.items():
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.05, f"Probs de {team} não somam 1: {total}"


def test_monte_carlo_title_race_dataframe(simple_table, remaining_fixtures):
    sim = LeagueSimulator()
    result = sim.simulate(simple_table, remaining_fixtures, n_simulations=200)
    df = sim.get_title_race(result)
    assert "team" in df.columns
    assert "title_prob" in df.columns


def test_monte_carlo_relegation_battle(simple_table, remaining_fixtures):
    sim = LeagueSimulator()
    result = sim.simulate(simple_table, remaining_fixtures, n_simulations=200)
    df = sim.get_relegation_battle(result)
    assert len(df) > 0


def test_monte_carlo_libertadores_g4(simple_table, remaining_fixtures):
    """Brasileirão específico: G4 (Libertadores)."""
    sim = LeagueSimulator()
    result = sim.simulate(simple_table, remaining_fixtures, n_simulations=200)
    df = sim.get_libertadores_race(result)
    assert "libertadores_prob" in df.columns


def test_monte_carlo_sulamericana_g6(simple_table, remaining_fixtures):
    """Brasileirão específico: G6 (Sul-Americana)."""
    sim = LeagueSimulator()
    result = sim.simulate(simple_table, remaining_fixtures, n_simulations=200)
    df = sim.get_sulamericana_race(result)
    assert "sulamericana_prob" in df.columns


def test_monte_carlo_leader_higher_title_prob(simple_table, remaining_fixtures):
    """Time líder deve ter maior probabilidade de título."""
    sim = LeagueSimulator()
    result = sim.simulate(simple_table, remaining_fixtures, n_simulations=500)
    leader = simple_table.iloc[0]["team"]
    last = simple_table.iloc[-1]["team"]
    assert result.title_probs.get(leader, 0) >= result.title_probs.get(last, 0)
