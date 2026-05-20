"""Modelo de previsão de resultados usando Poisson bivariado com MLE."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from goldata.logging_config import get_logger

logger = get_logger(__name__)


class BivariatePoisson:
    """
    Modelo de previsão usando distribuição de Poisson bivariada.

    Estima força de ataque e defesa de cada time + vantagem em casa
    usando Maximum Likelihood Estimation.

    Referência: Dixon & Coles (1997), Dixon & Pope (2004)
    """

    def __init__(self) -> None:
        self._teams: list[str] = []
        self._params: dict[str, float] = {}
        self.is_trained: bool = False

    def _get_lambda(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Retorna taxa esperada de gols para home e away."""
        home_attack = self._params.get(f"attack_{home_team}", 1.0)
        home_defense = self._params.get(f"defense_{home_team}", 1.0)
        away_attack = self._params.get(f"attack_{away_team}", 1.0)
        away_defense = self._params.get(f"defense_{away_team}", 1.0)
        home_adv = self._params.get("home_advantage", 1.2)
        intercept = self._params.get("intercept", 1.0)

        lambda_home = intercept * home_attack * away_defense * home_adv
        lambda_away = intercept * away_attack * home_defense

        return max(0.01, lambda_home), max(0.01, lambda_away)

    def fit(self, results_df: pd.DataFrame) -> "BivariatePoisson":
        """
        Estima parâmetros por MLE.

        Args:
            results_df: DataFrame com colunas home_team, away_team, home_goals, away_goals
        """
        teams = sorted(set(results_df["home_team"]) | set(results_df["away_team"]))
        self._teams = teams
        n_teams = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}

        def neg_log_likelihood(params: np.ndarray) -> float:
            """Log-verossimilhança negativa do modelo Poisson."""
            intercept = params[0]
            home_adv = params[1]
            attack = params[2: 2 + n_teams]
            defense = params[2 + n_teams: 2 + 2 * n_teams]

            ll = 0.0
            for _, row in results_df.iterrows():
                hi = team_idx[row["home_team"]]
                ai = team_idx[row["away_team"]]

                lh = intercept * attack[hi] * defense[ai] * home_adv
                la = intercept * attack[ai] * defense[hi]
                lh = max(0.01, lh)
                la = max(0.01, la)

                ll += poisson.logpmf(int(row["home_goals"]), lh)
                ll += poisson.logpmf(int(row["away_goals"]), la)

            return -ll if np.isfinite(ll) else 1e10

        # Inicialização
        x0 = np.concatenate([
            [1.0, 1.2],           # intercept, home_advantage
            np.ones(n_teams),     # attack
            np.ones(n_teams),     # defense
        ])
        bounds = (
            [(0.1, 5.0), (1.0, 2.0)]
            + [(0.1, 5.0)] * n_teams
            + [(0.1, 5.0)] * n_teams
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                neg_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-8},
            )

        params = result.x
        self._params["intercept"] = float(params[0])
        self._params["home_advantage"] = float(params[1])
        for i, team in enumerate(teams):
            self._params[f"attack_{team}"] = float(params[2 + i])
            self._params[f"defense_{team}"] = float(params[2 + n_teams + i])

        self.is_trained = True
        logger.info("poisson_fitted", teams=n_teams, matches=len(results_df))
        return self

    def predict_match(self, home_team: str, away_team: str, max_goals: int = 7) -> dict:
        """
        Prediz probabilidades de cada placar e de resultado.

        Args:
            home_team: nome do time da casa
            away_team: nome do time visitante
            max_goals: máximo de gols por time a considerar

        Returns:
            dict com score_matrix, home_win_prob, draw_prob, away_win_prob,
                      expected_home_goals, expected_away_goals
        """
        lambda_home, lambda_away = self._get_lambda(home_team, away_team)

        # Matriz de probabilidades de placar (max_goals x max_goals)
        score_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                score_matrix[h, a] = (
                    poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
                )

        home_win = float(np.tril(score_matrix, -1).sum())  # home > away
        draw = float(np.diag(score_matrix).sum())
        away_win = float(np.triu(score_matrix, 1).sum())

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": round(home_win, 4),
            "draw_prob": round(draw, 4),
            "away_win_prob": round(away_win, 4),
            "expected_home_goals": round(lambda_home, 3),
            "expected_away_goals": round(lambda_away, 3),
            "score_matrix": score_matrix.tolist(),
        }

    def get_team_ratings(self) -> pd.DataFrame:
        """Retorna DataFrame com ratings de ataque e defesa por time."""
        records = []
        for team in self._teams:
            records.append({
                "team": team,
                "attack": round(self._params.get(f"attack_{team}", 1.0), 4),
                "defense": round(self._params.get(f"defense_{team}", 1.0), 4),
            })
        return pd.DataFrame(records).sort_values("attack", ascending=False).reset_index(drop=True)
