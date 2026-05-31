"""Testes do cliente de dados do Brasileirão."""

import pytest
import pandas as pd
from goldata.data.brasileirao import BrasileiraoDataClient, BRASILEIRAO_QUALITY_FACTOR


@pytest.fixture
def client(tmp_path):
    return BrasileiraoDataClient(data_dir=str(tmp_path))


def test_client_creates_sample_data(tmp_path):
    client = BrasileiraoDataClient(data_dir=str(tmp_path))
    standings = client.get_serie_a_standings(2024)
    assert len(standings) == 20


def test_standings_has_required_columns(client):
    df = client.get_serie_a_standings()
    for col in ["team", "points", "wins", "draws", "losses"]:
        assert col in df.columns


def test_standings_20_teams(client):
    df = client.get_serie_a_standings()
    assert len(df) == 20


def test_standings_points_positive(client):
    df = client.get_serie_a_standings()
    assert (df["points"] >= 0).all()


def test_player_stats_has_required_columns(client):
    df = client.get_serie_a_player_stats()
    for col in ["display_name", "position", "minutes_played", "goals", "assists"]:
        assert col in df.columns


def test_player_stats_has_players(client):
    df = client.get_serie_a_player_stats()
    assert len(df) > 0


def test_player_stats_minutes_positive(client):
    df = client.get_serie_a_player_stats()
    assert (df["minutes_played"] > 0).all()


def test_matches_has_required_columns(client):
    df = client.get_serie_a_matches()
    for col in ["home_team", "away_team", "home_goals", "away_goals"]:
        assert col in df.columns


def test_matches_goals_non_negative(client):
    df = client.get_serie_a_matches()
    assert (df["home_goals"] >= 0).all()
    assert (df["away_goals"] >= 0).all()


def test_match_history_returns_dataframe(client):
    df = client.get_match_history("Flamengo")
    assert isinstance(df, pd.DataFrame)


def test_match_history_unknown_team_empty(client):
    df = client.get_match_history("Time Inexistente XYZ")
    assert len(df) == 0


def test_home_advantage_factor_returns_float(client):
    factor = client.get_home_advantage_factor("Flamengo")
    assert isinstance(factor, float)
    assert factor > 0


def test_home_advantage_unknown_team_uses_default(client):
    from goldata.data.brasileirao import HOME_ADVANTAGE_DEFAULT
    factor = client.get_home_advantage_factor("Time Desconhecido XYZ")
    assert factor == HOME_ADVANTAGE_DEFAULT


def test_quality_factor_correct(client):
    assert client.quality_factor == BRASILEIRAO_QUALITY_FACTOR
    assert 0 < client.quality_factor < 1


def test_caching_works(client):
    df1 = client.get_serie_a_standings()
    df2 = client.get_serie_a_standings()
    # Segundo call usa cache: não lança erro
    assert len(df1) == len(df2)
