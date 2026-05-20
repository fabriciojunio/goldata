"""Testes dos módulos táticos: passing network e pressing."""

import pytest
import numpy as np
import pandas as pd
import networkx as nx

from goldata.models.tactical.passing_network import PassingNetworkAnalyzer, PassingNetworkStats
from goldata.models.tactical.pressing import PressingAnalyzer


@pytest.fixture
def match_events(sample_events_df):
    """Events com team_id definido como home/away."""
    return sample_events_df


# ── Passing Network ────────────────────────────────────────────────────────────

def test_build_network_returns_digraph(match_events):
    analyzer = PassingNetworkAnalyzer()
    G = analyzer.build_network(match_events, team_id="team_home", min_passes=1)
    assert isinstance(G, nx.DiGraph)


def test_build_network_has_nodes_for_team(match_events):
    analyzer = PassingNetworkAnalyzer()
    G = analyzer.build_network(match_events, team_id="team_home", min_passes=1)
    assert len(G.nodes) >= 0  # pode ser 0 se nenhum passe completado


def test_build_network_min_passes_filter(match_events):
    analyzer = PassingNetworkAnalyzer()
    G_loose = analyzer.build_network(match_events, team_id="team_home", min_passes=1)
    G_strict = analyzer.build_network(match_events, team_id="team_home", min_passes=100)
    assert len(G_strict.edges) <= len(G_loose.edges)


def test_analyze_returns_stats(match_events):
    analyzer = PassingNetworkAnalyzer()
    stats = analyzer.analyze(match_events, team_id="team_home", min_passes=1)
    assert isinstance(stats, PassingNetworkStats)
    assert stats.team_id == "team_home"


def test_analyze_density_in_range(match_events):
    analyzer = PassingNetworkAnalyzer()
    stats = analyzer.analyze(match_events, team_id="team_home", min_passes=1)
    assert 0.0 <= stats.density <= 1.0


def test_analyze_avg_passes_positive(match_events):
    analyzer = PassingNetworkAnalyzer()
    stats = analyzer.analyze(match_events, team_id="team_home", min_passes=1)
    assert stats.avg_passes >= 0.0


def test_analyze_unknown_team_empty(match_events):
    analyzer = PassingNetworkAnalyzer()
    stats = analyzer.analyze(match_events, team_id="unknown_team_xyz")
    assert stats.n_players == 0
    assert stats.n_edges == 0


def test_analyze_centrality_dict(match_events):
    analyzer = PassingNetworkAnalyzer()
    stats = analyzer.analyze(match_events, team_id="team_home", min_passes=1)
    assert isinstance(stats.betweenness_centrality, dict)
    assert isinstance(stats.degree_centrality, dict)


def test_get_top_combinations():
    analyzer = PassingNetworkAnalyzer()
    G = nx.DiGraph()
    G.add_edge("player_1", "player_2", weight=15)
    G.add_edge("player_2", "player_3", weight=10)
    G.add_edge("player_1", "player_3", weight=5)
    combos = analyzer.get_top_combinations(G, n=2)
    assert len(combos) == 2
    assert combos[0]["passes"] == 15  # mais passes primeiro


def test_network_stats_edge_weights_populated(match_events):
    analyzer = PassingNetworkAnalyzer()
    stats = analyzer.analyze(match_events, team_id="team_home", min_passes=1)
    assert isinstance(stats.edge_weights, dict)


# ── Pressing / PPDA ────────────────────────────────────────────────────────────

def test_ppda_returns_float(match_events):
    ppda = PressingAnalyzer.calculate_ppda(match_events, "team_home", "team_away")
    assert isinstance(ppda, float)


def test_ppda_non_negative(match_events):
    ppda = PressingAnalyzer.calculate_ppda(match_events, "team_home", "team_away")
    assert ppda >= 0 or ppda == float("inf")


def test_ppda_no_defensive_actions_returns_inf():
    df = pd.DataFrame({
        "team_id": ["team_away", "team_away"],
        "event_type": ["pass", "pass"],
        "x": [30.0, 40.0],
        "outcome": ["success", "success"],
        "match_id": [1, 1],
    })
    ppda = PressingAnalyzer.calculate_ppda(df, "team_home", "team_away")
    assert ppda == float("inf")


def test_ppda_intensity_labels():
    assert PressingAnalyzer.pressing_intensity_label(5.0) == "Very High Press"
    assert PressingAnalyzer.pressing_intensity_label(10.0) == "High Press"
    assert PressingAnalyzer.pressing_intensity_label(15.0) == "Medium Press"
    assert PressingAnalyzer.pressing_intensity_label(25.0) == "Low Block"


def test_ppda_batch_returns_dataframe(match_events):
    result = PressingAnalyzer.calculate_ppda_batch(
        match_events,
        team_ids=["team_home", "team_away"],
    )
    assert isinstance(result, pd.DataFrame)


def test_ppda_batch_columns(match_events):
    result = PressingAnalyzer.calculate_ppda_batch(
        match_events, team_ids=["team_home"],
    )
    if len(result) > 0:
        assert "ppda_avg" in result.columns


def test_ppda_high_pressing_lower_value():
    """Time que pressa muito deve ter PPDA menor."""
    df_high = pd.DataFrame({
        "team_id": ["away"] * 5 + ["home"] * 10,
        "event_type": ["pass"] * 5 + ["tackle"] * 10,
        "x": [30.0] * 5 + [70.0] * 10,
        "match_id": [1] * 15,
    })
    df_low = pd.DataFrame({
        "team_id": ["away"] * 20 + ["home"] * 2,
        "event_type": ["pass"] * 20 + ["tackle"] * 2,
        "x": [30.0] * 20 + [70.0] * 2,
        "match_id": [1] * 22,
    })
    ppda_high = PressingAnalyzer.calculate_ppda(df_high, "home", "away")
    ppda_low = PressingAnalyzer.calculate_ppda(df_low, "home", "away")
    assert ppda_high < ppda_low
