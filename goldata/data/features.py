"""Feature engineering para modelos de xG, scouting e previsão."""

import math

import numpy as np
import pandas as pd

from goldata.logging_config import get_logger

logger = get_logger(__name__)

# Constantes do campo StatsBomb (120x80)
GOAL_X = 120.0
GOAL_Y_CENTER = 40.0
GOAL_POST_Y_LEFT = 36.0
GOAL_POST_Y_RIGHT = 44.0


class FeatureEngineer:
    """Extrai e transforma features para uso nos modelos de ML."""

    # ── Features de Chute (xG) ────────────────────────────────────────────────

    @staticmethod
    def calculate_distance_to_goal(x: float, y: float) -> float:
        """Distância euclidiana do chute ao centro do gol."""
        return math.sqrt((GOAL_X - x) ** 2 + (GOAL_Y_CENTER - y) ** 2)

    @staticmethod
    def calculate_angle_to_goal(x: float, y: float) -> float:
        """
        Ângulo de visão do gol em radianos.
        Maior ângulo = melhor posição de chute.
        """
        # Vetor para o poste esquerdo e direito
        a = math.sqrt((GOAL_X - x) ** 2 + (GOAL_POST_Y_LEFT - y) ** 2)
        b = math.sqrt((GOAL_X - x) ** 2 + (GOAL_POST_Y_RIGHT - y) ** 2)
        c = abs(GOAL_POST_Y_RIGHT - GOAL_POST_Y_LEFT)  # largura do gol = 8 metros

        # Lei dos cossenos para calcular o ângulo
        try:
            cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # clip para evitar erro numérico
            return math.acos(cos_angle)
        except (ZeroDivisionError, ValueError):
            return 0.0

    @staticmethod
    def extract_shot_features(shot: dict) -> dict:
        """
        Extrai todas as features de um chute para uso no xG.

        Args:
            shot: dict com campos x, y, body_part, technique, etc.

        Returns:
            dict com todas as features numéricas
        """
        x = float(shot.get("x", 60.0))
        y = float(shot.get("y", 40.0))
        body_part = str(shot.get("body_part", "foot")).lower()
        technique = str(shot.get("technique", "normal")).lower()

        return {
            "distance_to_goal": FeatureEngineer.calculate_distance_to_goal(x, y),
            "angle_to_goal": FeatureEngineer.calculate_angle_to_goal(x, y),
            "x": x,
            "y": y,
            "is_header": int("head" in body_part),
            "is_foot_right": int("right" in body_part),
            "is_foot_left": int("left" in body_part),
            "is_penalty": int("penalty" in technique or shot.get("is_penalty", False)),
            "is_direct_freekick": int(
                "free" in technique or shot.get("is_direct_freekick", False)
            ),
            "is_corner": int(shot.get("is_corner", False)),
            "is_open_play": int(shot.get("is_open_play", True)),
            "prev_action_type": shot.get("prev_action_type", "pass"),
            "shot_sequence_length": int(shot.get("shot_sequence_length", 3)),
        }

    @staticmethod
    def extract_shot_features_batch(shots_df: pd.DataFrame) -> pd.DataFrame:
        """Extrai features de chute de um DataFrame inteiro."""
        df = shots_df.copy()

        if "x" not in df.columns:
            df["x"] = 100.0
        if "y" not in df.columns:
            df["y"] = 40.0

        df["distance_to_goal"] = df.apply(
            lambda r: FeatureEngineer.calculate_distance_to_goal(r["x"], r["y"]), axis=1
        )
        df["angle_to_goal"] = df.apply(
            lambda r: FeatureEngineer.calculate_angle_to_goal(r["x"], r["y"]), axis=1
        )

        for col in ["is_header", "is_foot_right", "is_foot_left", "is_penalty",
                    "is_direct_freekick", "is_open_play"]:
            if col not in df.columns:
                df[col] = 0

        return df

    # ── Features de Jogador (Scouting) ────────────────────────────────────────

    @staticmethod
    def normalize_player_stats_per90(
        stats: pd.DataFrame,
        minutes_col: str = "minutes_played",
        count_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Normaliza estatísticas de jogador para por 90 minutos.

        Args:
            stats: DataFrame com estatísticas
            minutes_col: nome da coluna de minutos jogados
            count_columns: colunas a normalizar (auto-detecta se None)

        Returns:
            DataFrame com colunas _per_90 adicionadas
        """
        df = stats.copy()
        minutes = df[minutes_col].clip(lower=1)  # evitar divisão por zero

        default_count_cols = [
            "goals", "assists", "xg", "xa", "shots", "shots_on_target",
            "passes", "passes_completed", "key_passes", "progressive_passes",
            "tackles", "interceptions", "blocks", "pressures",
            "dribbles", "dribbles_completed", "touches",
        ]
        cols_to_normalize = count_columns or [
            c for c in default_count_cols if c in df.columns
        ]

        for col in cols_to_normalize:
            df[f"{col}_per_90"] = (df[col] / minutes * 90).round(3)

        # Taxas derivadas
        if "passes" in df.columns and "passes_completed" in df.columns:
            df["pass_completion_rate"] = (df["passes_completed"] / df["passes"].clip(1)).round(3)

        if "shots" in df.columns and "shots_on_target" in df.columns:
            df["shot_accuracy"] = (df["shots_on_target"] / df["shots"].clip(1)).round(3)

        if "dribbles" in df.columns and "dribbles_completed" in df.columns:
            df["dribble_success_rate"] = (
                df["dribbles_completed"] / df["dribbles"].clip(1)
            ).round(3)

        return df

    # ── Features de Time (Previsão) ────────────────────────────────────────────

    @staticmethod
    def extract_team_form_features(
        results: pd.DataFrame,
        team: str,
        n_games: int = 5,
        reference_date: str | None = None,
    ) -> dict:
        """
        Extrai features de forma recente de um time.

        Args:
            results: DataFrame com colunas home_team, away_team, home_goals, away_goals
            team: nome do time
            n_games: número de jogos recentes a considerar
            reference_date: data de referência (ISO string)

        Returns:
            dict com features de forma
        """
        # Filtrar jogos do time
        home_mask = results["home_team"] == team
        away_mask = results["away_team"] == team
        team_games = results[home_mask | away_mask].tail(n_games)

        if len(team_games) == 0:
            return _empty_form_features(team, n_games)

        points = []
        goals_scored = []
        goals_conceded = []
        xg_for = []
        xg_against = []

        for _, row in team_games.iterrows():
            is_home = row["home_team"] == team
            if is_home:
                gf = row.get("home_goals", 0)
                ga = row.get("away_goals", 0)
                xg_f = row.get("home_xg", gf)
                xg_a = row.get("away_xg", ga)
            else:
                gf = row.get("away_goals", 0)
                ga = row.get("home_goals", 0)
                xg_f = row.get("away_xg", gf)
                xg_a = row.get("home_xg", ga)

            goals_scored.append(gf)
            goals_conceded.append(ga)
            xg_for.append(xg_f)
            xg_against.append(xg_a)

            if gf > ga:
                points.append(3)
            elif gf == ga:
                points.append(1)
            else:
                points.append(0)

        n = len(points)
        return {
            "team": team,
            "games_analyzed": n,
            f"points_last_{n_games}": sum(points),
            f"wins_last_{n_games}": sum(1 for p in points if p == 3),
            f"draws_last_{n_games}": sum(1 for p in points if p == 1),
            f"losses_last_{n_games}": sum(1 for p in points if p == 0),
            f"goals_scored_avg_last_{n_games}": round(np.mean(goals_scored), 3),
            f"goals_conceded_avg_last_{n_games}": round(np.mean(goals_conceded), 3),
            f"xg_for_avg_last_{n_games}": round(np.mean(xg_for), 3),
            f"xg_against_avg_last_{n_games}": round(np.mean(xg_against), 3),
        }


def _empty_form_features(team: str, n_games: int) -> dict:
    """Retorna features zeradas quando não há dados."""
    return {
        "team": team,
        "games_analyzed": 0,
        f"points_last_{n_games}": 0,
        f"wins_last_{n_games}": 0,
        f"draws_last_{n_games}": 0,
        f"losses_last_{n_games}": 0,
        f"goals_scored_avg_last_{n_games}": 0.0,
        f"goals_conceded_avg_last_{n_games}": 0.0,
        f"xg_for_avg_last_{n_games}": 0.0,
        f"xg_against_avg_last_{n_games}": 0.0,
    }
