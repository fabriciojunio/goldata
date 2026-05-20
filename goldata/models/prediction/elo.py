"""Sistema de rating Elo para times de futebol."""

import math
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from goldata.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_RATING = 1500.0
K_FACTOR_NEW = 40.0      # Times com < 20 jogos
K_FACTOR_STABLE = 20.0   # Times estabelecidos
HOME_ADVANTAGE = 100.0   # Pontos de vantagem em casa


@dataclass
class EloTeamRecord:
    team: str
    rating: float = DEFAULT_RATING
    games_played: int = 0
    history: list[tuple[str, float]] = field(default_factory=list)


class EloRating:
    """
    Sistema de rating Elo adaptativo para futebol.

    Características:
    - K-factor adaptativo (maior para times novos)
    - Vantagem em casa integrada
    - Goal difference bonus
    - Histórico de ratings por time
    """

    def __init__(
        self,
        default_rating: float = DEFAULT_RATING,
        home_advantage: float = HOME_ADVANTAGE,
        k_new: float = K_FACTOR_NEW,
        k_stable: float = K_FACTOR_STABLE,
    ) -> None:
        self.default_rating = default_rating
        self.home_advantage = home_advantage
        self.k_new = k_new
        self.k_stable = k_stable
        self._teams: dict[str, EloTeamRecord] = {}

    def _get_or_create(self, team: str) -> EloTeamRecord:
        if team not in self._teams:
            self._teams[team] = EloTeamRecord(team=team, rating=self.default_rating)
        return self._teams[team]

    def _k_factor(self, team: str) -> float:
        record = self._get_or_create(team)
        return self.k_new if record.games_played < 20 else self.k_stable

    @staticmethod
    def _expected_score(rating_a: float, rating_b: float) -> float:
        """Probabilidade esperada de vitória de A sobre B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def _goal_difference_multiplier(home_goals: int, away_goals: int) -> float:
        """Multiplicador baseado na diferença de gols (máx 1.75)."""
        diff = abs(home_goals - away_goals)
        if diff <= 1:
            return 1.0
        elif diff == 2:
            return 1.5
        else:
            return min(1.75, 1.75 + (diff - 3) * 0.04)

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        date: str | None = None,
    ) -> tuple[float, float]:
        """
        Atualiza ratings após uma partida.

        Returns:
            (new_home_rating, new_away_rating)
        """
        home_record = self._get_or_create(home_team)
        away_record = self._get_or_create(away_team)

        # Ajustar rating do time da casa com vantagem
        home_adj = home_record.rating + self.home_advantage
        expected_home = self._expected_score(home_adj, away_record.rating)
        expected_away = 1.0 - expected_home

        # Resultado real: 1=vitória, 0.5=empate, 0=derrota
        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals == away_goals:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0

        gdm = self._goal_difference_multiplier(home_goals, away_goals)

        k_home = self._k_factor(home_team)
        k_away = self._k_factor(away_team)

        new_home = home_record.rating + k_home * gdm * (actual_home - expected_home)
        new_away = away_record.rating + k_away * gdm * (actual_away - expected_away)

        # Salvar histórico
        event = date or datetime.now().strftime("%Y-%m-%d")
        home_record.history.append((event, new_home))
        away_record.history.append((event, new_away))

        home_record.rating = new_home
        away_record.rating = new_away
        home_record.games_played += 1
        away_record.games_played += 1

        return new_home, new_away

    def train(self, results_df: pd.DataFrame) -> "EloRating":
        """
        Treina o sistema com histórico de resultados.

        Args:
            results_df: DataFrame com colunas home_team, away_team, home_goals, away_goals
        """
        for _, row in results_df.iterrows():
            self.update(
                home_team=str(row["home_team"]),
                away_team=str(row["away_team"]),
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
                date=str(row.get("date", "")),
            )
        logger.info("elo_trained", teams=len(self._teams))
        return self

    def predict_match(self, home_team: str, away_team: str) -> dict:
        """
        Prediz probabilidades para uma partida.

        Returns:
            dict com home_win_prob, draw_prob, away_win_prob, expected_home_goals
        """
        home_record = self._get_or_create(home_team)
        away_record = self._get_or_create(away_team)

        home_adj = home_record.rating + self.home_advantage
        p_home_win = self._expected_score(home_adj, away_record.rating)

        # Estimar empate baseado na diferença de rating
        rating_diff = abs(home_adj - away_record.rating)
        p_draw = max(0.15, 0.30 - rating_diff / 3000.0)
        p_draw = min(p_draw, min(p_home_win, 1 - p_home_win) * 0.8)

        p_home_win_adj = p_home_win * (1 - p_draw)
        p_away_win = max(0.0, 1.0 - p_home_win_adj - p_draw)

        # Normalizar para somar 1
        total = p_home_win_adj + p_draw + p_away_win
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": round(p_home_win_adj / total, 4),
            "draw_prob": round(p_draw / total, 4),
            "away_win_prob": round(p_away_win / total, 4),
            "home_elo": round(home_record.rating, 1),
            "away_elo": round(away_record.rating, 1),
        }

    def get_ratings(self) -> pd.DataFrame:
        """Retorna DataFrame com todos os ratings ranqueados."""
        if not self._teams:
            return pd.DataFrame(columns=["team", "rating", "games_played"])
        records = [
            {"team": t, "rating": r.rating, "games_played": r.games_played}
            for t, r in self._teams.items()
        ]
        return pd.DataFrame(records).sort_values("rating", ascending=False).reset_index(drop=True)

    def get_rating(self, team: str) -> float:
        """Retorna rating atual de um time."""
        return self._get_or_create(team).rating

    def get_rating_history(self, team: str) -> list[tuple[str, float]]:
        """Retorna histórico de ratings de um time."""
        record = self._get_or_create(team)
        return record.history.copy()
