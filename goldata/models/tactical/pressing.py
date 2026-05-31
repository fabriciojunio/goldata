"""Análise de pressing usando PPDA (Passes Allowed per Defensive Action)."""

import numpy as np
import pandas as pd

from goldata.logging_config import get_logger

logger = get_logger(__name__)

# Zonas de campo para PPDA (StatsBomb: 120x80)
# PPDA mede pressing na metade de campo adversária (x > 60)
PPDA_PRESS_ZONE_X = 60.0


class PressingAnalyzer:
    """
    Analisa o pressing de um time usando PPDA.

    PPDA = Passes do adversário permitidos na própria metade /
           Ações defensivas do time que pressa na metade adversária

    Referência: StatsBomb, Spielverlagerung.
    Menor PPDA = pressing mais intenso.
    """

    @staticmethod
    def calculate_ppda(
        events_df: pd.DataFrame,
        pressing_team: str,
        opponent_team: str,
    ) -> float:
        """
        Calcula PPDA para um time em uma partida.

        Args:
            events_df: DataFrame de eventos com team_id, event_type, x, outcome
            pressing_team: time que está pressionando (avaliando pressing)
            opponent_team: time adversário

        Returns:
            PPDA (float): menor = pressing mais intenso
        """
        # Passes do adversário na própria metade de campo (x < 60)
        opponent_passes = events_df[
            (events_df["team_id"] == opponent_team)
            & (events_df["event_type"] == "pass")
            & (events_df.get("x", pd.Series([0.0] * len(events_df))) < PPDA_PRESS_ZONE_X)
        ]

        # Ações defensivas do time pressionador na metade adversária (x > 60)
        defensive_actions = events_df[
            (events_df["team_id"] == pressing_team)
            & (events_df["event_type"].isin(["tackle", "interception", "foul"]))
            & (events_df.get("x", pd.Series([0.0] * len(events_df))) > PPDA_PRESS_ZONE_X)
        ]

        n_opponent_passes = len(opponent_passes)
        n_defensive_actions = len(defensive_actions)

        if n_defensive_actions == 0:
            return float("inf")

        return round(n_opponent_passes / n_defensive_actions, 3)

    @staticmethod
    def calculate_ppda_batch(
        events_df: pd.DataFrame,
        team_ids: list[str],
    ) -> pd.DataFrame:
        """
        Calcula PPDA para múltiplos times a partir de múltiplas partidas.

        Args:
            events_df: DataFrame com match_id, team_id, event_type, x, outcome
            team_ids: lista de IDs dos times

        Returns:
            DataFrame com team_id, ppda_avg, ppda_min, ppda_max, matches_analyzed
        """
        results = []
        unique_teams = set(events_df["team_id"].unique()) if "team_id" in events_df.columns else set()

        for team in team_ids:
            if team not in unique_teams:
                continue

            match_ppda_values = []
            match_ids = events_df["match_id"].unique() if "match_id" in events_df.columns else [None]

            for match_id in match_ids:
                if match_id is not None:
                    match_events = events_df[events_df["match_id"] == match_id]
                else:
                    match_events = events_df

                opponents_in_match = [
                    t for t in match_events["team_id"].unique() if t != team
                ]
                for opponent in opponents_in_match:
                    ppda = PressingAnalyzer.calculate_ppda(match_events, team, opponent)
                    if ppda != float("inf"):
                        match_ppda_values.append(ppda)

            if match_ppda_values:
                results.append({
                    "team_id": team,
                    "ppda_avg": round(float(np.mean(match_ppda_values)), 3),
                    "ppda_min": round(float(min(match_ppda_values)), 3),
                    "ppda_max": round(float(max(match_ppda_values)), 3),
                    "matches_analyzed": len(match_ppda_values),
                })

        return pd.DataFrame(results)

    @staticmethod
    def pressing_intensity_label(ppda: float) -> str:
        """
        Converte PPDA em rótulo de intensidade de pressing.

        < 8:  Pressing muito alto (tipo Klopp/City)
        8-12: Pressing alto
        12-18: Pressing médio
        > 18: Baixo pressing / bloco defensivo
        """
        if ppda < 8:
            return "Very High Press"
        elif ppda < 12:
            return "High Press"
        elif ppda < 18:
            return "Medium Press"
        else:
            return "Low Block"
