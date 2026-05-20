"""Testes de feature engineering."""

import math
import pytest
import numpy as np
import pandas as pd
from goldata.data.features import FeatureEngineer, GOAL_X, GOAL_Y_CENTER


fe = FeatureEngineer()


# ── Distância e Ângulo ────────────────────────────────────────────────────────

def test_distance_penalty_spot():
    # Pênalti: x=108, y=40 no StatsBomb → ~12 metros do gol
    dist = fe.calculate_distance_to_goal(108, 40)
    assert 10 < dist < 15


def test_distance_center_of_field():
    dist = fe.calculate_distance_to_goal(60, 40)
    assert dist > 50  # longe do gol


def test_distance_on_goal_line():
    dist = fe.calculate_distance_to_goal(120, 40)
    assert dist == pytest.approx(0.0)


def test_angle_penalty_center():
    angle = fe.calculate_angle_to_goal(108, 40)
    assert angle > 0.3  # bom ângulo


def test_angle_corner_position():
    angle = fe.calculate_angle_to_goal(115, 2)
    assert angle < 0.3  # ângulo ruim (posição lateral extrema)


def test_angle_center_better_than_corner():
    center_angle = fe.calculate_angle_to_goal(108, 40)
    corner_angle = fe.calculate_angle_to_goal(115, 5)
    assert center_angle > corner_angle


def test_angle_non_negative():
    for x in [80, 90, 100, 110, 120]:
        for y in [10, 20, 40, 60, 70]:
            assert fe.calculate_angle_to_goal(x, y) >= 0


# ── extract_shot_features ─────────────────────────────────────────────────────

def test_extract_shot_features_returns_dict():
    shot = {"x": 110, "y": 40, "body_part": "right foot"}
    result = fe.extract_shot_features(shot)
    assert isinstance(result, dict)


def test_extract_shot_features_all_keys_present():
    shot = {"x": 110, "y": 40}
    result = fe.extract_shot_features(shot)
    required_keys = ["distance_to_goal", "angle_to_goal", "is_header",
                     "is_foot_right", "is_foot_left", "is_penalty",
                     "is_direct_freekick", "is_open_play"]
    for key in required_keys:
        assert key in result, f"Chave ausente: {key}"


def test_extract_shot_features_header_detected():
    shot = {"x": 108, "y": 40, "body_part": "head"}
    result = fe.extract_shot_features(shot)
    assert result["is_header"] == 1


def test_extract_shot_features_penalty_detected():
    shot = {"x": 108, "y": 40, "technique": "penalty"}
    result = fe.extract_shot_features(shot)
    assert result["is_penalty"] == 1


def test_extract_shot_features_freekick_detected():
    shot = {"x": 95, "y": 40, "technique": "free kick"}
    result = fe.extract_shot_features(shot)
    assert result["is_direct_freekick"] == 1


def test_extract_shot_features_penalty_larger_than_distance():
    """Pênalti (distância ~12m) vs chute de longe (~35m) — distância menor no pênalti."""
    penalty = fe.extract_shot_features({"x": 108, "y": 40, "technique": "penalty"})
    long_shot = fe.extract_shot_features({"x": 85, "y": 40})
    assert penalty["distance_to_goal"] < long_shot["distance_to_goal"]


# ── normalize_player_stats_per90 ──────────────────────────────────────────────

def test_normalize_per90_basic(sample_player_stats_df):
    result = fe.normalize_player_stats_per90(sample_player_stats_df)
    assert "goals_per_90" in result.columns
    assert "assists_per_90" in result.columns


def test_normalize_per90_values_reasonable(sample_player_stats_df):
    result = fe.normalize_player_stats_per90(sample_player_stats_df)
    # Nenhum jogador deve ter mais de 5 gols por 90 min
    assert result["goals_per_90"].max() < 5.0


def test_normalize_per90_does_not_modify_original(sample_player_stats_df):
    original_goals = sample_player_stats_df["goals"].copy()
    fe.normalize_player_stats_per90(sample_player_stats_df)
    assert sample_player_stats_df["goals"].equals(original_goals)


def test_normalize_per90_pass_completion_rate(sample_player_stats_df):
    result = fe.normalize_player_stats_per90(sample_player_stats_df)
    if "pass_completion_rate" in result.columns:
        assert (result["pass_completion_rate"] >= 0).all()


def test_normalize_per90_dribble_success_rate(sample_player_stats_df):
    result = fe.normalize_player_stats_per90(sample_player_stats_df)
    if "dribble_success_rate" in result.columns:
        assert (result["dribble_success_rate"] >= 0).all()


# ── extract_team_form_features ────────────────────────────────────────────────

def test_team_form_features_returns_dict(sample_match_results_df):
    result = fe.extract_team_form_features(sample_match_results_df, "Flamengo")
    assert isinstance(result, dict)


def test_team_form_features_contains_points(sample_match_results_df):
    result = fe.extract_team_form_features(sample_match_results_df, "Flamengo")
    assert "points_last_5" in result


def test_team_form_features_points_valid(sample_match_results_df):
    result = fe.extract_team_form_features(sample_match_results_df, "Flamengo")
    pts = result.get("points_last_5", 0)
    assert 0 <= pts <= 15  # max 5 vitórias = 15 pontos


def test_team_form_features_unknown_team(sample_match_results_df):
    result = fe.extract_team_form_features(sample_match_results_df, "Time Inexistente")
    assert result["games_analyzed"] == 0


def test_team_form_features_n_games_respected(sample_match_results_df):
    result = fe.extract_team_form_features(sample_match_results_df, "Flamengo", n_games=10)
    assert "points_last_10" in result


def test_team_form_averages_are_positive(sample_match_results_df):
    result = fe.extract_team_form_features(sample_match_results_df, "Palmeiras")
    if result["games_analyzed"] > 0:
        assert result.get("goals_scored_avg_last_5", 0) >= 0
        assert result.get("goals_conceded_avg_last_5", 0) >= 0


# ── extract_shot_features_batch ───────────────────────────────────────────────

def test_batch_features_adds_distance_column(sample_shots_df):
    result = fe.extract_shot_features_batch(sample_shots_df)
    assert "distance_to_goal" in result.columns


def test_batch_features_adds_angle_column(sample_shots_df):
    result = fe.extract_shot_features_batch(sample_shots_df)
    assert "angle_to_goal" in result.columns


def test_batch_features_distances_positive(sample_shots_df):
    result = fe.extract_shot_features_batch(sample_shots_df)
    assert (result["distance_to_goal"] >= 0).all()
