"""Análise de padrões de arbitragem: cartões, pênaltis e home bias."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from goldata.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RefereeProfile:
    referee_id: str
    name: str
    matches_analyzed: int
    yellow_cards_per_game: float
    red_cards_per_game: float
    penalties_per_game: float
    home_team_advantage_index: float  # > 1 = favorece mandante
    avg_added_time: float
    strictness_score: float          # 0-1, maior = mais rigoroso


class RefereeAnalyzer:
    """
    Analisa padrões de arbitragem para uso estratégico.

    Usos:
    - Prever número de cartões em uma partida (para apostas totais)
    - Identificar árbitros com home bias
    - Ajustar previsões baseado no árbitro escalado
    """

    def __init__(self) -> None:
        self._referee_profiles: dict[str, RefereeProfile] = {}

    def build_profile(
        self,
        matches_df: pd.DataFrame,
        referee_col: str = "referee_id",
    ) -> dict[str, RefereeProfile]:
        """
        Constrói perfis de árbitros a partir do histórico de partidas.

        Args:
            matches_df: DataFrame com referee_id, yellow_cards_home,
                        yellow_cards_away, red_cards, penalties_home,
                        penalties_away, home_goals, away_goals, added_time

        Returns:
            dict {referee_id: RefereeProfile}
        """
        profiles = {}
        for ref_id, group in matches_df.groupby(referee_col):
            n = len(group)

            yellow_home = group.get("yellow_cards_home", pd.Series(dtype=float)).fillna(0).sum()
            yellow_away = group.get("yellow_cards_away", pd.Series(dtype=float)).fillna(0).sum()
            red_total = group.get("red_cards", pd.Series(dtype=float)).fillna(0).sum()
            pen_home = group.get("penalties_home", pd.Series(dtype=float)).fillna(0).sum()
            pen_away = group.get("penalties_away", pd.Series(dtype=float)).fillna(0).sum()
            added_time = group.get("added_time", pd.Series(dtype=float)).fillna(4.0).mean()

            yellow_per_game = (yellow_home + yellow_away) / n
            red_per_game = red_total / n
            pen_per_game = (pen_home + pen_away) / n

            # Home advantage index: pênaltis e cartões a favor do mandante
            home_decisions = pen_home + yellow_away  # favorece mandante
            away_decisions = pen_away + yellow_home  # favorece visitante
            total_decisions = home_decisions + away_decisions
            home_adv_idx = (home_decisions / total_decisions) / 0.5 if total_decisions > 0 else 1.0

            # Strictness: baseado em cartões totais normalizado
            strictness = min(1.0, yellow_per_game / 6.0 + red_per_game / 2.0)

            profile = RefereeProfile(
                referee_id=str(ref_id),
                name=str(group.get("referee_name", pd.Series([str(ref_id)])).iloc[0]),
                matches_analyzed=n,
                yellow_cards_per_game=round(yellow_per_game, 3),
                red_cards_per_game=round(red_per_game, 3),
                penalties_per_game=round(pen_per_game, 3),
                home_team_advantage_index=round(home_adv_idx, 3),
                avg_added_time=round(float(added_time), 1),
                strictness_score=round(strictness, 3),
            )
            profiles[str(ref_id)] = profile
            self._referee_profiles[str(ref_id)] = profile

        logger.info("referee_profiles_built", n_referees=len(profiles))
        return profiles

    def get_match_adjustment(
        self,
        referee_id: str,
        base_home_win_prob: float,
    ) -> dict:
        """
        Ajusta probabilidades de uma partida baseado no perfil do árbitro.

        Args:
            referee_id: ID do árbitro escalado
            base_home_win_prob: probabilidade base de vitória do mandante

        Returns:
            dict com probabilidades ajustadas e previsão de cartões
        """
        profile = self._referee_profiles.get(referee_id)
        if profile is None:
            return {
                "home_win_prob_adjusted": round(base_home_win_prob, 4),
                "referee_found": False,
                "expected_yellow_cards": 3.5,
                "expected_red_cards": 0.1,
            }

        # Ajuste: árbitros com home bias fortalecem probabilidade do mandante
        bias_adj = (profile.home_team_advantage_index - 1.0) * 0.05
        adj_prob = min(0.95, max(0.05, base_home_win_prob + bias_adj))

        return {
            "home_win_prob_adjusted": round(adj_prob, 4),
            "referee_found": True,
            "referee_id": referee_id,
            "home_advantage_index": profile.home_team_advantage_index,
            "expected_yellow_cards": round(profile.yellow_cards_per_game, 1),
            "expected_red_cards": round(profile.red_cards_per_game, 2),
            "expected_penalties": round(profile.penalties_per_game, 2),
            "strictness_score": profile.strictness_score,
        }

    def get_top_strict_referees(self, n: int = 5) -> list[RefereeProfile]:
        """Retorna os árbitros mais rigorosos (mais cartões)."""
        return sorted(
            self._referee_profiles.values(),
            key=lambda r: -r.strictness_score,
        )[:n]

    def get_profiles_dataframe(self) -> pd.DataFrame:
        """Retorna todos os perfis como DataFrame."""
        if not self._referee_profiles:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "referee_id": p.referee_id,
                "matches": p.matches_analyzed,
                "yellow_per_game": p.yellow_cards_per_game,
                "red_per_game": p.red_cards_per_game,
                "penalties_per_game": p.penalties_per_game,
                "home_adv_index": p.home_team_advantage_index,
                "strictness": p.strictness_score,
            }
            for p in self._referee_profiles.values()
        ]).sort_values("strictness", ascending=False).reset_index(drop=True)
