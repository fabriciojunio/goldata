"""Simulador de liga via Monte Carlo com suporte ao Brasileirão (G4/G6/Z4)."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from goldata.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SimulationResult:
    """Resultado de N simulações de uma liga."""

    team_position_probs: dict[str, dict[int, float]] = field(default_factory=dict)
    title_probs: dict[str, float] = field(default_factory=dict)
    top4_probs: dict[str, float] = field(default_factory=dict)   # Libertadores (Brasileirão)
    top6_probs: dict[str, float] = field(default_factory=dict)   # Sul-Americana
    relegation_probs: dict[str, float] = field(default_factory=dict)
    n_simulations: int = 0
    teams: list[str] = field(default_factory=list)


class LeagueSimulator:
    """
    Simulador de liga usando Monte Carlo.

    Simula todos os jogos restantes N vezes usando um modelo de previsão
    e calcula a distribuição de probabilidades de posição final.

    Suporte especial ao Brasileirão:
    - G4: Libertadores (posições 1-4)
    - G6: Sul-Americana (posições 5-6)
    - Z4: Rebaixamento (últimas 4 posições)
    """

    def __init__(self, prediction_model: Any = None) -> None:
        """
        Args:
            prediction_model: modelo com método predict_match() (Poisson, DC, Elo).
                              Se None, usa modelo randômico.
        """
        self._model = prediction_model

    def _predict_match_outcome(
        self, home_team: str, away_team: str
    ) -> tuple[int, int]:
        """Simula um jogo e retorna (home_goals, away_goals)."""
        if self._model is None:
            # Fallback: Poisson simples com médias padrão
            lh, la = 1.5, 1.1
        else:
            pred = self._model.predict_match(home_team, away_team)
            lh = pred.get("expected_home_goals", 1.5)
            la = pred.get("expected_away_goals", 1.1)

        return int(np.random.poisson(lh)), int(np.random.poisson(la))

    def _simulate_once(
        self,
        current_table: pd.DataFrame,
        remaining_fixtures: list[tuple[str, str]],
    ) -> pd.DataFrame:
        """Executa uma simulação completa e retorna tabela final."""
        table = current_table.copy()
        table = table.set_index("team") if "team" in table.columns else table

        for home_team, away_team in remaining_fixtures:
            hg, ag = self._predict_match_outcome(home_team, away_team)

            if home_team in table.index:
                table.loc[home_team, "goals_for"] = table.loc[home_team].get("goals_for", 0) + hg
                table.loc[home_team, "goals_against"] = table.loc[home_team].get("goals_against", 0) + ag
                if hg > ag:
                    table.loc[home_team, "points"] = table.loc[home_team].get("points", 0) + 3
                elif hg == ag:
                    table.loc[home_team, "points"] = table.loc[home_team].get("points", 0) + 1

            if away_team in table.index:
                table.loc[away_team, "goals_for"] = table.loc[away_team].get("goals_for", 0) + ag
                table.loc[away_team, "goals_against"] = table.loc[away_team].get("goals_against", 0) + hg
                if ag > hg:
                    table.loc[away_team, "points"] = table.loc[away_team].get("points", 0) + 3
                elif ag == hg:
                    table.loc[away_team, "points"] = table.loc[away_team].get("points", 0) + 1

        table["goal_diff"] = table.get("goals_for", 0) - table.get("goals_against", 0)
        table_sorted = table.sort_values(
            ["points", "goal_diff", "goals_for"],
            ascending=[False, False, False],
        )
        return table_sorted.reset_index()

    def simulate(
        self,
        current_table: pd.DataFrame,
        remaining_fixtures: list[tuple[str, str]],
        n_simulations: int = 5000,
        n_jobs: int = 1,
    ) -> SimulationResult:
        """
        Executa N simulações da liga.

        Args:
            current_table: DataFrame com colunas team, points, goals_for, goals_against
            remaining_fixtures: lista de (home_team, away_team)
            n_simulations: número de simulações (default 5000)
            n_jobs: paralelismo com joblib (-1 = usar todos os cores)

        Returns:
            SimulationResult com distribuição de posições
        """
        teams = current_table["team"].tolist() if "team" in current_table.columns else list(current_table.index)
        n_teams = len(teams)
        position_counts: dict[str, dict[int, int]] = {t: {p: 0 for p in range(1, n_teams + 1)} for t in teams}

        def run_one(_: int) -> list[str]:
            sim_table = self._simulate_once(current_table, remaining_fixtures)
            if "team" in sim_table.columns:
                return sim_table["team"].tolist()
            return list(sim_table.index)

        if n_jobs != 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_one)(i) for i in range(n_simulations)
            )
        else:
            results = [run_one(i) for i in range(n_simulations)]

        for sim_teams_order in results:
            for pos, team in enumerate(sim_teams_order, start=1):
                if team in position_counts:
                    position_counts[team][pos] = position_counts[team].get(pos, 0) + 1

        # Converter para probabilidades
        team_position_probs: dict[str, dict[int, float]] = {}
        for team in teams:
            team_position_probs[team] = {
                pos: round(count / n_simulations, 4)
                for pos, count in position_counts[team].items()
            }

        result = SimulationResult(
            team_position_probs=team_position_probs,
            n_simulations=n_simulations,
            teams=teams,
        )

        # Calcular probs específicas
        for team in teams:
            result.title_probs[team] = team_position_probs[team].get(1, 0.0)
            result.top4_probs[team] = sum(
                team_position_probs[team].get(p, 0) for p in range(1, 5)
            )
            result.top6_probs[team] = sum(
                team_position_probs[team].get(p, 0) for p in range(1, 7)
            )
            result.relegation_probs[team] = sum(
                team_position_probs[team].get(p, 0) for p in range(n_teams - 3, n_teams + 1)
            )

        logger.info(
            "monte_carlo_completed",
            n_simulations=n_simulations,
            n_teams=n_teams,
            remaining_fixtures=len(remaining_fixtures),
        )
        return result

    def get_title_race(self, result: SimulationResult) -> pd.DataFrame:
        """Retorna DataFrame com probabilidade de título por time."""
        records = [{"team": t, "title_prob": p} for t, p in result.title_probs.items()]
        return pd.DataFrame(records).sort_values("title_prob", ascending=False).reset_index(drop=True)

    def get_relegation_battle(self, result: SimulationResult) -> pd.DataFrame:
        """Retorna DataFrame com probabilidade de rebaixamento por time."""
        records = [{"team": t, "relegation_prob": p} for t, p in result.relegation_probs.items()]
        return pd.DataFrame(records).sort_values("relegation_prob", ascending=False).reset_index(drop=True)

    def get_libertadores_race(self, result: SimulationResult) -> pd.DataFrame:
        """G4 — Classificação para a Libertadores (Top 4 do Brasileirão)."""
        records = [{"team": t, "libertadores_prob": p} for t, p in result.top4_probs.items()]
        return pd.DataFrame(records).sort_values("libertadores_prob", ascending=False).reset_index(drop=True)

    def get_sulamericana_race(self, result: SimulationResult) -> pd.DataFrame:
        """G6 — Classificação para a Sul-Americana (Top 6 do Brasileirão)."""
        records = [{"team": t, "sulamericana_prob": p} for t, p in result.top6_probs.items()]
        return pd.DataFrame(records).sort_values("sulamericana_prob", ascending=False).reset_index(drop=True)
