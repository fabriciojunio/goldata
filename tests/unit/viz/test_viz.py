"""Testes de visualizações — verifica que as funções retornam fig, ax sem erros."""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # sem display no CI
import matplotlib.pyplot as plt

from goldata.viz.pitch import draw_pitch, plot_shot_map, plot_heatmap
from goldata.viz.radar import plot_player_radar, plot_comparison_radar


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ── Pitch ─────────────────────────────────────────────────────────────────────

def test_draw_pitch_returns_fig_ax():
    fig, ax = draw_pitch()
    assert fig is not None
    assert ax is not None


def test_draw_pitch_custom_color():
    fig, ax = draw_pitch(color="#2d572c")
    assert fig is not None


def test_draw_pitch_with_existing_ax():
    fig_ext, ax_ext = plt.subplots()
    fig, ax = draw_pitch(ax=ax_ext)
    assert ax is ax_ext


def test_plot_shot_map_empty_returns():
    df = pd.DataFrame({"x": [], "y": []})
    fig, ax = plot_shot_map(df, title="Empty Shots")
    assert fig is not None


def test_plot_shot_map_with_goals(sample_shots_df):
    shots = sample_shots_df[["x", "y", "is_goal"]].copy()
    shots["xg"] = 0.15
    fig, ax = plot_shot_map(shots, title="Test Shot Map")
    assert fig is not None


def test_plot_shot_map_no_xg_column(sample_shots_df):
    shots = sample_shots_df[["x", "y", "is_goal"]].copy()
    fig, ax = plot_shot_map(shots)
    assert fig is not None


def test_plot_heatmap_returns():
    df = pd.DataFrame({
        "x": np.random.uniform(0, 120, 50),
        "y": np.random.uniform(0, 80, 50),
    })
    fig, ax = plot_heatmap(df, title="Heatmap Test")
    assert fig is not None


def test_plot_heatmap_empty():
    df = pd.DataFrame({"z": [1, 2, 3]})  # sem x, y
    fig, ax = plot_heatmap(df)
    assert fig is not None


# ── Radar ─────────────────────────────────────────────────────────────────────

def test_radar_basic():
    stats = {
        "Gols/90": 0.7, "Assistências/90": 0.5, "xG/90": 0.6,
        "Passes-chave/90": 0.8, "Dribles/90": 0.4,
    }
    fig, ax = plot_player_radar(stats, title="Radar Test")
    assert fig is not None


def test_radar_normalized_values():
    stats = {"A": 1.5, "B": -0.2, "C": 0.8, "D": 0.5}
    fig, ax = plot_player_radar(stats)
    assert fig is not None


def test_radar_too_few_metrics():
    stats = {"A": 0.5, "B": 0.7}
    fig, ax = plot_player_radar(stats)
    assert fig is not None


def test_radar_custom_metrics_subset():
    stats = {"Gols/90": 0.7, "Assistências/90": 0.5, "xG/90": 0.6, "Extra": 0.3}
    fig, ax = plot_player_radar(stats, metrics=["Gols/90", "Assistências/90", "xG/90"])
    assert fig is not None


def test_comparison_radar():
    player_a = {"Gols/90": 0.8, "xG/90": 0.7, "Passes-chave/90": 0.4, "Dribles/90": 0.6}
    player_b = {"Gols/90": 0.5, "xG/90": 0.6, "Passes-chave/90": 0.8, "Dribles/90": 0.3}
    fig, ax = plot_comparison_radar(player_a, player_b, name_a="Neymar", name_b="Vinicius")
    assert fig is not None


def test_comparison_radar_returns_legend():
    player_a = {"A": 0.5, "B": 0.7, "C": 0.6}
    player_b = {"A": 0.4, "B": 0.8, "C": 0.5}
    fig, ax = plot_comparison_radar(player_a, player_b)
    # Verifica que plotou sem erros
    assert ax is not None
