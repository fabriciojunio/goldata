"""Fixtures compartilhadas para toda a suíte de testes do GolData."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_shots_df() -> pd.DataFrame:
    """DataFrame de chutes sample para testes de xG."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "x": np.random.uniform(80, 120, n),
        "y": np.random.uniform(15, 65, n),
        "distance_to_goal": np.random.uniform(3, 35, n),
        "angle_to_goal": np.random.uniform(0.1, 1.5, n),
        "is_header": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "is_foot_right": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "is_foot_left": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "is_penalty": np.random.choice([0, 1], n, p=[0.92, 0.08]),
        "is_direct_freekick": np.random.choice([0, 1], n, p=[0.88, 0.12]),
        "is_open_play": np.random.choice([0, 1], n, p=[0.2, 0.8]),
        "is_goal": np.random.choice([0, 1], n, p=[0.9, 0.1]),
    })


@pytest.fixture
def sample_player_stats_df() -> pd.DataFrame:
    """DataFrame de stats de jogadores sample."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "player_id": [f"player_{i}" for i in range(n)],
        "display_name": [f"Jogador {i}" for i in range(n)],
        "position": np.random.choice(["FW", "MF", "DF", "GK"], n),
        "minutes_played": np.random.uniform(500, 3000, n),
        "goals": np.random.uniform(0, 25, n),
        "assists": np.random.uniform(0, 15, n),
        "xg": np.random.uniform(0, 20, n),
        "xa": np.random.uniform(0, 12, n),
        "shots": np.random.uniform(10, 120, n),
        "passes": np.random.uniform(500, 3000, n),
        "passes_completed": np.random.uniform(400, 2800, n),
        "key_passes": np.random.uniform(5, 80, n),
        "progressive_passes": np.random.uniform(10, 200, n),
        "tackles": np.random.uniform(10, 100, n),
        "interceptions": np.random.uniform(5, 60, n),
        "pressures": np.random.uniform(50, 500, n),
        "dribbles": np.random.uniform(5, 100, n),
        "dribbles_completed": np.random.uniform(3, 80, n),
    })


@pytest.fixture
def sample_match_results_df() -> pd.DataFrame:
    """DataFrame de resultados de partidas sample."""
    np.random.seed(42)
    teams = ["Flamengo", "Palmeiras", "Atletico MG", "Corinthians",
             "Botafogo", "Vasco", "Fluminense", "Santos",
             "Sao Paulo", "Internacional"]
    matches = []
    for i in range(100):
        home = np.random.choice(teams)
        away = np.random.choice([t for t in teams if t != home])
        home_goals = np.random.poisson(1.5)
        away_goals = np.random.poisson(1.1)
        matches.append({
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_xg": np.random.uniform(0.5, 3.0),
            "away_xg": np.random.uniform(0.3, 2.5),
            "season": "2024",
        })
    return pd.DataFrame(matches)


@pytest.fixture
def sample_events_df() -> pd.DataFrame:
    """DataFrame de eventos de partida sample."""
    np.random.seed(42)
    n = 500
    event_types = ["pass", "shot", "dribble", "tackle", "interception", "foul", "clearance"]
    return pd.DataFrame({
        "id": range(n),
        "match_id": [1] * n,
        "team_id": np.random.choice(["team_home", "team_away"], n),
        "player_id": [f"player_{np.random.randint(1, 23)}" for _ in range(n)],
        "event_type": np.random.choice(event_types, n, p=[0.6, 0.05, 0.1, 0.08, 0.07, 0.05, 0.05]),
        "minute": np.random.randint(1, 91, n),
        "second": np.random.randint(0, 60, n),
        "x": np.random.uniform(0, 120, n),
        "y": np.random.uniform(0, 80, n),
        "end_x": np.random.uniform(0, 120, n),
        "end_y": np.random.uniform(0, 80, n),
        "outcome": np.random.choice(["success", "failure"], n, p=[0.65, 0.35]),
        "period": np.random.choice([1, 2], n),
    })
